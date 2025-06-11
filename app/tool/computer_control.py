"""
Enhanced Computer Control Tools for ParManus AI
Provides comprehensive system control capabilities including file management, app control,
screen capture, mouse/keyboard automation, and computer vision-based UI interaction
"""

import base64
import ctypes
import json
import os
import shutil
import subprocess
import tempfile
import time
import winreg
from io import BytesIO
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import cv2
import mss
import numpy as np
import psutil
import pyautogui
import pygetwindow as gw
import pyperclip
from PIL import Image
from screeninfo import get_monitors

from app.logger import logger
from app.tool.base import BaseTool, ToolResult

# Configure PyAutoGUI safety settings
pyautogui.FAILSAFE = True
pyautogui.PAUSE = 0.1


class ComputerControlTool(BaseTool):
    """Enhanced tool for comprehensive computer system control"""

    name: str = "computer_control"
    description: str = """Control the computer system with advanced capabilities including:
    - Launch/close applications and manage processes
    - Take screenshots and capture specific screen regions
    - Control mouse (click, move, drag, scroll) with pixel precision
    - Control keyboard (type text, send keys, shortcuts)
    - Manage windows (move, resize, minimize, maximize, focus)
    - Find and interact with UI elements using computer vision
    - Manage clipboard operations
    - Execute system commands and scripts
    - Monitor system information and performance    - Automate complex multi-step workflows"""

    class Config:
        arbitrary_types_allowed = True
        extra = "allow"

    # Action name mapping for common incorrect variations
    ACTION_NAME_MAP: Dict[str, str] = {
        # Screenshot variations
        "capture_screenshot": "screenshot",
        "take_screenshot": "screenshot",
        "get_screenshot": "screenshot",
        "screen_capture": "screenshot",
        # Mouse action variations
        "click_button": "mouse_click",
        "click_mouse": "mouse_click",
        "mouse_click_button": "mouse_click",
        "click": "mouse_click",
        "move_mouse": "mouse_move",
        "mouse_movement": "mouse_move",
        "drag_mouse": "mouse_drag",
        "scroll_mouse": "mouse_scroll",
        "get_mouse_pos": "get_mouse_position",
        "mouse_pos": "get_mouse_position",
        # Keyboard variations
        "type": "type_text",
        "input_text": "type_text",
        "keyboard_input": "type_text",
        "send_key": "send_keys",
        "key_press": "send_keys",
        "keyboard_press": "send_keys",
        "key_combo": "key_combination",
        "hotkey": "key_combination",
        # Application variations
        "launch_application": "launch_app",
        "start_app": "launch_app",
        "open_app": "launch_app",
        "run_app": "launch_app",
        "close_application": "close_app",
        "stop_app": "close_app",
        "terminate_app": "close_app",
        "list_running_processes": "list_processes",
        "get_processes": "list_processes",
        "show_processes": "list_processes",
        "terminate_process": "kill_process",
        "stop_process": "kill_process",
        # Window variations
        "get_windows": "list_windows",
        "show_windows": "list_windows",
        "switch_window": "focus_window",
        "activate_window": "focus_window",
        "set_window_focus": "focus_window",
        "window_move": "move_window",
        "window_resize": "resize_window",
        "window_minimize": "minimize_window",
        "window_maximize": "maximize_window",
        "window_close": "close_window",
    }

    parameters: dict = {
        "type": "object",
        "properties": {
            "action": {
                "type": "string",
                "enum": [
                    "screenshot",
                    "screenshot_region",
                    "mouse_click",
                    "mouse_move",
                    "mouse_drag",
                    "mouse_scroll",
                    "type_text",
                    "send_keys",
                    "key_combination",
                    "launch_app",
                    "close_app",
                    "list_processes",
                    "kill_process",
                    "list_windows",
                    "focus_window",
                    "move_window",
                    "resize_window",
                    "minimize_window",
                    "maximize_window",
                    "close_window",
                    "find_ui_element",
                    "click_ui_element",
                    "get_clipboard",
                    "set_clipboard",
                    "execute_command",
                    "get_system_info",
                    "get_mouse_position",
                    "get_screen_info",
                    "wait",
                ],
                "description": "The computer control action to perform",
            },
            "target": {
                "type": "string",
                "description": "Target app, process, window, or element name",
            },
            "x": {
                "type": "integer",
                "description": "X coordinate for mouse actions or region capture",
            },
            "y": {
                "type": "integer",
                "description": "Y coordinate for mouse actions or region capture",
            },
            "width": {
                "type": "integer",
                "description": "Width for region capture or window resize",
            },
            "height": {
                "type": "integer",
                "description": "Height for region capture or window resize",
            },
            "text": {
                "type": "string",
                "description": "Text to type or search for",
            },
            "keys": {
                "type": "string",
                "description": "Keys to send (e.g., 'ctrl+c', 'alt+tab', 'enter')",
            },
            "button": {
                "type": "string",
                "enum": ["left", "right", "middle"],
                "description": "Mouse button for click actions",
            },
            "clicks": {
                "type": "integer",
                "description": "Number of clicks (default: 1)",
            },
            "scroll_direction": {
                "type": "string",
                "enum": ["up", "down"],
                "description": "Scroll direction for mouse scroll",
            },
            "scroll_amount": {
                "type": "integer",
                "description": "Amount to scroll (default: 3)",
            },
            "confidence": {
                "type": "number",
                "description": "Confidence threshold for UI element detection (0.0-1.0)",
            },
            "timeout": {
                "type": "integer",
                "description": "Timeout in seconds for operations",
            },
            "parameters": {
                "type": "object",
                "description": "Additional parameters for specific actions",
            },
        },
        "required": ["action"],
    }

    def __init__(self):
        super().__init__()
        self.screenshot_sct = mss.mss()

    def _map_action_name(self, action: str) -> str:
        """Map common incorrect action name variations to correct ones"""
        mapped_action = self.ACTION_NAME_MAP.get(action.lower(), action)
        if mapped_action != action:
            logger.info(f"🔄 Mapped action '{action}' to '{mapped_action}'")
        return mapped_action

    def _check_existing_app_instances(self, app_name: str) -> List[Dict[str, Any]]:
        """Check for existing instances of an application"""
        existing_instances = []  # Map app name to process name
        app_process_map = {
            "calculator": "CalculatorApp.exe",
            "calc": "CalculatorApp.exe",
            "notepad": "notepad.exe",
            "paint": "mspaint.exe",
            "chrome": "chrome.exe",
            "firefox": "firefox.exe",
            "edge": "msedge.exe",
        }

        process_name = app_process_map.get(app_name.lower(), f"{app_name.lower()}.exe")

        for proc in psutil.process_iter(["pid", "name", "create_time"]):
            try:
                if process_name.lower() in proc.info["name"].lower():
                    existing_instances.append(
                        {
                            "pid": proc.info["pid"],
                            "name": proc.info["name"],
                            "create_time": proc.info["create_time"],
                        }
                    )
            except (psutil.NoSuchProcess, psutil.AccessDenied):
                continue

        return existing_instances

    def _focus_existing_calculator(self, instances: List[Dict[str, Any]]) -> bool:
        """Focus on existing calculator window if available"""
        try:
            # Try to find calculator windows and focus them
            calculator_windows = []
            for window in gw.getAllWindows():
                if (
                    "calculator" in window.title.lower()
                    or "calc" in window.title.lower()
                ):
                    calculator_windows.append(window)

            if calculator_windows:
                # Focus the most recent calculator window
                calculator_windows[0].activate()
                logger.info(
                    f"✅ Focused existing calculator window: {calculator_windows[0].title}"
                )
                return True
            return False
        except Exception as e:
            logger.warning(f"Could not focus existing calculator: {e}")
            return False

    async def execute(
        self,
        action: str,
        target: Optional[str] = None,
        x: Optional[int] = None,
        y: Optional[int] = None,
        width: Optional[int] = None,
        height: Optional[int] = None,
        text: Optional[str] = None,
        keys: Optional[str] = None,
        button: str = "left",
        clicks: int = 1,
        scroll_direction: str = "down",
        scroll_amount: int = 3,
        confidence: float = 0.8,
        timeout: int = 10,
        parameters: Optional[Dict[str, Any]] = None,
        **kwargs,
    ) -> ToolResult:
        """Execute enhanced computer control actions with smart workflows"""
        try:
            if parameters is None:
                parameters = {}

            # Map action name to handle common variations
            original_action = action
            action = self._map_action_name(action)

            logger.info(f"🖥️ Computer control action: {action}")

            # Smart Calculator Workflow - Check for existing instances before launching
            if (
                action == "launch_app"
                and target
                and target.lower() in ["calculator", "calc"]
            ):
                existing_instances = self._check_existing_app_instances(target)

                if existing_instances:
                    logger.info(
                        f"📱 Found {len(existing_instances)} existing calculator instance(s)"
                    )

                    # Try to focus existing calculator
                    if self._focus_existing_calculator(existing_instances):
                        return ToolResult(
                            output=f"✅ Used existing calculator instance (PID: {existing_instances[0]['pid']}) instead of launching new one"
                        )
                    else:
                        logger.warning(
                            "Could not focus existing calculator, will launch new one"
                        )
                else:
                    logger.info(
                        "🆕 No existing calculator found, launching new instance"
                    )

            # Screenshot actions
            if action == "screenshot":
                return await self._take_screenshot()
            elif action == "screenshot_region":
                return await self._take_screenshot_region(x, y, width, height)

            # Mouse actions
            elif action == "mouse_click":
                return await self._mouse_click(x, y, button, clicks)
            elif action == "mouse_move":
                return await self._mouse_move(x, y)
            elif action == "mouse_drag":
                return await self._mouse_drag(
                    x, y, parameters.get("end_x"), parameters.get("end_y")
                )
            elif action == "mouse_scroll":
                return await self._mouse_scroll(x, y, scroll_direction, scroll_amount)
            elif action == "get_mouse_position":
                return await self._get_mouse_position()

            # Keyboard actions
            elif action == "type_text":
                return await self._type_text(text)
            elif action == "send_keys":
                return await self._send_keys(keys)
            elif action == "key_combination":
                return await self._key_combination(keys)

            # Application actions
            elif action == "launch_app":
                return await self._launch_app(target, parameters)
            elif action == "close_app":
                return await self._close_app(target)
            elif action == "list_processes":
                return await self._list_processes()
            elif action == "kill_process":
                return await self._kill_process(target)

            # Window management
            elif action == "list_windows":
                return await self._list_windows()
            elif action == "focus_window":
                return await self._focus_window(target)
            elif action == "move_window":
                return await self._move_window(target, x, y)
            elif action == "resize_window":
                return await self._resize_window(target, width, height)
            elif action == "minimize_window":
                return await self._minimize_window(target)
            elif action == "maximize_window":
                return await self._maximize_window(target)
            elif action == "close_window":
                return await self._close_window(target)

            # UI element detection and interaction
            elif action == "find_ui_element":
                return await self._find_ui_element(target, confidence, timeout)
            elif action == "click_ui_element":
                return await self._click_ui_element(target, confidence, timeout)

            # Clipboard operations
            elif action == "get_clipboard":
                return await self._get_clipboard()
            elif action == "set_clipboard":
                return await self._set_clipboard(text)

            # System operations
            elif action == "execute_command":
                return await self._execute_command(target, parameters)
            elif action == "get_system_info":
                return await self._get_system_info()
            elif action == "get_screen_info":
                return await self._get_screen_info()
            elif action == "wait":
                return await self._wait(timeout)

            else:
                return ToolResult(error=f"Unknown computer control action: {action}")

        except Exception as e:
            logger.error(f"Computer control error: {e}")
            return ToolResult(error=f"Computer control failed: {str(e)}")

    # Screenshot methods
    async def _take_screenshot(self) -> ToolResult:
        """Take a full screenshot"""
        try:
            # Get primary monitor
            monitor = self.screenshot_sct.monitors[1]  # 0 is all monitors, 1 is primary

            # Take screenshot
            screenshot = self.screenshot_sct.grab(monitor)

            # Convert to PIL Image
            img = Image.frombytes(
                "RGB", screenshot.size, screenshot.bgra, "raw", "BGRX"
            )

            # Convert to base64
            buffer = BytesIO()
            img.save(buffer, format="PNG")
            screenshot_b64 = base64.b64encode(buffer.getvalue()).decode("utf-8")

            return ToolResult(
                output=f"Screenshot taken: {img.size[0]}x{img.size[1]}",
                base64_image=screenshot_b64,
            )

        except Exception as e:
            return ToolResult(error=f"Failed to take screenshot: {str(e)}")

    async def _take_screenshot_region(
        self, x: int, y: int, width: int, height: int
    ) -> ToolResult:
        """Take a screenshot of a specific region"""
        try:
            if not all(v is not None for v in [x, y, width, height]):
                return ToolResult(
                    error="Region coordinates (x, y, width, height) are required"
                )

            # Define region
            region = {"top": y, "left": x, "width": width, "height": height}

            # Take screenshot
            screenshot = self.screenshot_sct.grab(region)

            # Convert to PIL Image
            img = Image.frombytes(
                "RGB", screenshot.size, screenshot.bgra, "raw", "BGRX"
            )

            # Convert to base64
            buffer = BytesIO()
            img.save(buffer, format="PNG")
            screenshot_b64 = base64.b64encode(buffer.getvalue()).decode("utf-8")

            return ToolResult(
                output=f"Region screenshot taken: {width}x{height} at ({x}, {y})",
                base64_image=screenshot_b64,
            )

        except Exception as e:
            return ToolResult(error=f"Failed to take region screenshot: {str(e)}")

    # Mouse control methods
    async def _mouse_click(
        self, x: Optional[int], y: Optional[int], button: str, clicks: int
    ) -> ToolResult:
        """Click mouse at specified coordinates"""
        try:
            if x is None or y is None:
                # Click at current position
                x, y = pyautogui.position()

            pyautogui.click(x, y, clicks=clicks, button=button)

            return ToolResult(
                output=f"Clicked {button} button {clicks} time(s) at ({x}, {y})"
            )

        except Exception as e:
            return ToolResult(error=f"Failed to click mouse: {str(e)}")

    async def _mouse_move(self, x: int, y: int) -> ToolResult:
        """Move mouse to specified coordinates"""
        try:
            if x is None or y is None:
                return ToolResult(
                    error="Coordinates (x, y) are required for mouse move"
                )

            pyautogui.moveTo(x, y)

            return ToolResult(output=f"Moved mouse to ({x}, {y})")

        except Exception as e:
            return ToolResult(error=f"Failed to move mouse: {str(e)}")

    async def _mouse_drag(
        self, start_x: int, start_y: int, end_x: int, end_y: int
    ) -> ToolResult:
        """Drag mouse from start to end coordinates"""
        try:
            if not all(v is not None for v in [start_x, start_y, end_x, end_y]):
                return ToolResult(
                    error="Start and end coordinates are required for drag"
                )

            pyautogui.drag(end_x - start_x, end_y - start_y, duration=0.5)

            return ToolResult(
                output=f"Dragged from ({start_x}, {start_y}) to ({end_x}, {end_y})"
            )

        except Exception as e:
            return ToolResult(error=f"Failed to drag mouse: {str(e)}")

    async def _mouse_scroll(
        self, x: Optional[int], y: Optional[int], direction: str, amount: int
    ) -> ToolResult:
        """Scroll mouse wheel"""
        try:
            if x is not None and y is not None:
                pyautogui.moveTo(x, y)

            scroll_amount = amount if direction == "up" else -amount
            pyautogui.scroll(scroll_amount)

            return ToolResult(
                output=f"Scrolled {direction} by {amount} at ({x or 'current'}, {y or 'current'})"
            )

        except Exception as e:
            return ToolResult(error=f"Failed to scroll: {str(e)}")

    async def _get_mouse_position(self) -> ToolResult:
        """Get current mouse position"""
        try:
            x, y = pyautogui.position()
            return ToolResult(output=f"Mouse position: ({x}, {y})")

        except Exception as e:
            return ToolResult(error=f"Failed to get mouse position: {str(e)}")

    # Keyboard control methods
    async def _type_text(self, text: str) -> ToolResult:
        """Type text"""
        try:
            if not text:
                return ToolResult(error="Text is required for typing")

            pyautogui.typewrite(text)

            return ToolResult(
                output=f"Typed text: '{text[:50]}{'...' if len(text) > 50 else ''}'"
            )

        except Exception as e:
            return ToolResult(error=f"Failed to type text: {str(e)}")

    async def _send_keys(self, keys: str) -> ToolResult:
        """Send special keys"""
        try:
            if not keys:
                return ToolResult(error="Keys are required")

            pyautogui.press(keys)

            return ToolResult(output=f"Sent keys: {keys}")

        except Exception as e:
            return ToolResult(error=f"Failed to send keys: {str(e)}")

    async def _key_combination(self, keys: str) -> ToolResult:
        """Send key combination (e.g., 'ctrl+c')"""
        try:
            if not keys:
                return ToolResult(error="Key combination is required")

            # Parse key combination
            key_list = [key.strip() for key in keys.split("+")]

            if len(key_list) == 1:
                pyautogui.press(key_list[0])
            else:
                pyautogui.hotkey(*key_list)

            return ToolResult(output=f"Sent key combination: {keys}")

        except Exception as e:
            return ToolResult(error=f"Failed to send key combination: {str(e)}")

    # Application control methods
    async def _launch_app(
        self, app_name: str, parameters: Dict[str, Any]
    ) -> ToolResult:
        """Launch an application"""
        try:
            if not app_name:
                return ToolResult(error="Application name is required")

            # Common applications mapping
            app_paths = {
                "notepad": "notepad.exe",
                "calculator": "calc.exe",
                "paint": "mspaint.exe",
                "chrome": r"C:\Program Files\Google\Chrome\Application\chrome.exe",
                "firefox": r"C:\Program Files\Mozilla Firefox\firefox.exe",
                "edge": r"C:\Program Files (x86)\Microsoft\Edge\Application\msedge.exe",
                "code": r"C:\Users\{}\AppData\Local\Programs\Microsoft VS Code\Code.exe".format(
                    os.getenv("USERNAME")
                ),
                "cmd": "cmd.exe",
                "powershell": "powershell.exe",
                "explorer": "explorer.exe",
            }

            # Get app path
            app_path = app_paths.get(app_name.lower(), app_name)

            # Add arguments if provided
            args = parameters.get("args", [])
            if args:
                command = [app_path] + args
            else:
                command = app_path

            # Launch application
            process = subprocess.Popen(command, shell=True)

            return ToolResult(
                output=f"Launched application: {app_name} (PID: {process.pid})"
            )

        except Exception as e:
            return ToolResult(error=f"Failed to launch app {app_name}: {str(e)}")

    async def _close_app(self, app_name: str) -> ToolResult:
        """Close an application by name"""
        try:
            if not app_name:
                return ToolResult(error="Application name is required")

            closed_count = 0

            for proc in psutil.process_iter(["pid", "name"]):
                try:
                    if app_name.lower() in proc.info["name"].lower():
                        proc.terminate()
                        closed_count += 1
                except (psutil.NoSuchProcess, psutil.AccessDenied):
                    continue

            if closed_count > 0:
                return ToolResult(
                    output=f"Closed {closed_count} instance(s) of {app_name}"
                )
            else:
                return ToolResult(output=f"No running instances of {app_name} found")

        except Exception as e:
            return ToolResult(error=f"Failed to close app {app_name}: {str(e)}")

    async def _list_processes(self) -> ToolResult:
        """List running processes"""
        try:
            processes = []
            for proc in psutil.process_iter(
                ["pid", "name", "memory_info", "cpu_percent"]
            ):
                try:
                    processes.append(
                        {
                            "pid": proc.info["pid"],
                            "name": proc.info["name"],
                            "memory_mb": round(
                                proc.info["memory_info"].rss / 1024 / 1024, 1
                            ),
                            "cpu_percent": proc.info["cpu_percent"],
                        }
                    )
                except (psutil.NoSuchProcess, psutil.AccessDenied):
                    continue

            # Sort by memory usage
            processes.sort(key=lambda x: x["memory_mb"], reverse=True)

            # Return top 20 processes
            top_processes = processes[:20]
            output = "🖥️ Top 20 Processes by Memory Usage:\n"
            for proc in top_processes:
                output += (
                    f"  {proc['name']} (PID: {proc['pid']}) - {proc['memory_mb']} MB\n"
                )

            return ToolResult(output=output)

        except Exception as e:
            return ToolResult(error=f"Failed to list processes: {str(e)}")

    async def _kill_process(self, target: str) -> ToolResult:
        """Kill a process by name or PID"""
        try:
            if not target:
                return ToolResult(error="Process name or PID is required")

            killed_count = 0

            # Try to kill by PID first
            try:
                pid = int(target)
                proc = psutil.Process(pid)
                proc.terminate()
                killed_count = 1
                return ToolResult(output=f"Killed process with PID {pid}")
            except ValueError:
                # Not a PID, try by name
                pass
            except (psutil.NoSuchProcess, psutil.AccessDenied) as e:
                return ToolResult(error=f"Failed to kill process {pid}: {str(e)}")

            # Kill by name
            for proc in psutil.process_iter(["pid", "name"]):
                try:
                    if target.lower() in proc.info["name"].lower():
                        proc.terminate()
                        killed_count += 1
                except (psutil.NoSuchProcess, psutil.AccessDenied):
                    continue

            if killed_count > 0:
                return ToolResult(
                    output=f"Killed {killed_count} process(es) matching '{target}'"
                )
            else:
                return ToolResult(output=f"No processes found matching '{target}'")

        except Exception as e:
            return ToolResult(error=f"Failed to kill process: {str(e)}")

    # Window management methods
    async def _list_windows(self) -> ToolResult:
        """List all open windows"""
        try:
            windows = gw.getAllWindows()

            output = "🪟 Open Windows:\n"
            for i, window in enumerate(windows):
                if window.title:  # Only show windows with titles
                    output += f"  {i+1}. {window.title} - {window.width}x{window.height} at ({window.left}, {window.top})\n"

            return ToolResult(output=output)

        except Exception as e:
            return ToolResult(error=f"Failed to list windows: {str(e)}")

    async def _focus_window(self, window_title: str) -> ToolResult:
        """Focus on a specific window"""
        try:
            if not window_title:
                return ToolResult(error="Window title is required")

            windows = gw.getWindowsWithTitle(window_title)

            if not windows:
                return ToolResult(
                    error=f"No window found with title containing '{window_title}'"
                )

            window = windows[0]
            window.activate()

            return ToolResult(output=f"Focused window: {window.title}")

        except Exception as e:
            return ToolResult(error=f"Failed to focus window: {str(e)}")

    async def _move_window(self, window_title: str, x: int, y: int) -> ToolResult:
        """Move a window to specified coordinates"""
        try:
            if not window_title:
                return ToolResult(error="Window title is required")
            if x is None or y is None:
                return ToolResult(error="Coordinates (x, y) are required")

            windows = gw.getWindowsWithTitle(window_title)

            if not windows:
                return ToolResult(
                    error=f"No window found with title containing '{window_title}'"
                )

            window = windows[0]
            window.moveTo(x, y)

            return ToolResult(output=f"Moved window '{window.title}' to ({x}, {y})")

        except Exception as e:
            return ToolResult(error=f"Failed to move window: {str(e)}")

    async def _resize_window(
        self, window_title: str, width: int, height: int
    ) -> ToolResult:
        """Resize a window"""
        try:
            if not window_title:
                return ToolResult(error="Window title is required")
            if width is None or height is None:
                return ToolResult(error="Width and height are required")

            windows = gw.getWindowsWithTitle(window_title)

            if not windows:
                return ToolResult(
                    error=f"No window found with title containing '{window_title}'"
                )

            window = windows[0]
            window.resizeTo(width, height)

            return ToolResult(
                output=f"Resized window '{window.title}' to {width}x{height}"
            )

        except Exception as e:
            return ToolResult(error=f"Failed to resize window: {str(e)}")

    async def _minimize_window(self, window_title: str) -> ToolResult:
        """Minimize a window"""
        try:
            if not window_title:
                return ToolResult(error="Window title is required")

            windows = gw.getWindowsWithTitle(window_title)

            if not windows:
                return ToolResult(
                    error=f"No window found with title containing '{window_title}'"
                )

            window = windows[0]
            window.minimize()

            return ToolResult(output=f"Minimized window: {window.title}")

        except Exception as e:
            return ToolResult(error=f"Failed to minimize window: {str(e)}")

    async def _maximize_window(self, window_title: str) -> ToolResult:
        """Maximize a window"""
        try:
            if not window_title:
                return ToolResult(error="Window title is required")

            windows = gw.getWindowsWithTitle(window_title)

            if not windows:
                return ToolResult(
                    error=f"No window found with title containing '{window_title}'"
                )

            window = windows[0]
            window.maximize()

            return ToolResult(output=f"Maximized window: {window.title}")

        except Exception as e:
            return ToolResult(error=f"Failed to maximize window: {str(e)}")

    async def _close_window(self, window_title: str) -> ToolResult:
        """Close a window"""
        try:
            if not window_title:
                return ToolResult(error="Window title is required")

            windows = gw.getWindowsWithTitle(window_title)

            if not windows:
                return ToolResult(
                    error=f"No window found with title containing '{window_title}'"
                )

            window = windows[0]
            window.close()

            return ToolResult(output=f"Closed window: {window.title}")

        except Exception as e:
            return ToolResult(error=f"Failed to close window: {str(e)}")

    # UI element detection methods
    async def _find_ui_element(
        self, image_path: str, confidence: float, timeout: int
    ) -> ToolResult:
        """Find UI element using image recognition"""
        try:
            if not image_path:
                return ToolResult(error="Image path or description is required")

            # Try to locate the element on screen
            try:
                location = pyautogui.locateOnScreen(image_path, confidence=confidence)

                if location:
                    center = pyautogui.center(location)
                    return ToolResult(
                        output=f"Found UI element at ({center.x}, {center.y}), "
                        f"region: {location.left}, {location.top}, {location.width}, {location.height}"
                    )
                else:
                    return ToolResult(output="UI element not found on screen")

            except pyautogui.ImageNotFoundException:
                return ToolResult(output="UI element not found on screen")

        except Exception as e:
            return ToolResult(error=f"Failed to find UI element: {str(e)}")

    async def _click_ui_element(
        self, image_path: str, confidence: float, timeout: int
    ) -> ToolResult:
        """Click on UI element using image recognition"""
        try:
            if not image_path:
                return ToolResult(error="Image path or description is required")

            # Try to locate and click the element
            try:
                location = pyautogui.locateOnScreen(image_path, confidence=confidence)

                if location:
                    center = pyautogui.center(location)
                    pyautogui.click(center)
                    return ToolResult(
                        output=f"Clicked UI element at ({center.x}, {center.y})"
                    )
                else:
                    return ToolResult(error="UI element not found on screen")

            except pyautogui.ImageNotFoundException:
                return ToolResult(error="UI element not found on screen")

        except Exception as e:
            return ToolResult(error=f"Failed to click UI element: {str(e)}")

    # Clipboard methods
    async def _get_clipboard(self) -> ToolResult:
        """Get clipboard content"""
        try:
            content = pyperclip.paste()
            return ToolResult(
                output=f"Clipboard content: {content[:200]}{'...' if len(content) > 200 else ''}"
            )

        except Exception as e:
            return ToolResult(error=f"Failed to get clipboard: {str(e)}")

    async def _set_clipboard(self, text: str) -> ToolResult:
        """Set clipboard content"""
        try:
            if not text:
                return ToolResult(error="Text is required for clipboard")

            pyperclip.copy(text)
            return ToolResult(
                output=f"Copied to clipboard: {text[:100]}{'...' if len(text) > 100 else ''}"
            )

        except Exception as e:
            return ToolResult(error=f"Failed to set clipboard: {str(e)}")

    # System information methods
    async def _execute_command(
        self, command: str, parameters: Dict[str, Any]
    ) -> ToolResult:
        """Execute system command"""
        try:
            if not command:
                return ToolResult(error="Command is required")

            # Security check - only allow safe commands
            safe_commands = [
                "dir",
                "ls",
                "pwd",
                "whoami",
                "date",
                "time",
                "hostname",
                "systeminfo",
                "tasklist",
                "ipconfig",
                "ping",
                "nslookup",
                "echo",
                "type",
                "cat",
                "head",
                "tail",
                "find",
                "grep",
            ]

            command_base = command.split()[0].lower()
            if command_base not in safe_commands:
                return ToolResult(
                    error=f"Command '{command_base}' not allowed for security reasons"
                )

            # Execute command
            result = subprocess.run(
                command,
                shell=True,
                capture_output=True,
                text=True,
                timeout=parameters.get("timeout", 10),
            )

            output = result.stdout
            if result.stderr:
                output += f"\nErrors: {result.stderr}"

            return ToolResult(output=f"Command executed: {command}\n{output}")

        except subprocess.TimeoutExpired:
            return ToolResult(error=f"Command timed out: {command}")
        except Exception as e:
            return ToolResult(error=f"Failed to execute command: {str(e)}")

    async def _get_system_info(self) -> ToolResult:
        """Get system information"""
        try:
            import platform

            # Get basic system info
            info = {
                "system": platform.system(),
                "release": platform.release(),
                "version": platform.version(),
                "machine": platform.machine(),
                "processor": platform.processor(),
                "python_version": platform.python_version(),
            }

            # Get memory info
            memory = psutil.virtual_memory()
            info["memory_total_gb"] = round(memory.total / (1024**3), 2)
            info["memory_available_gb"] = round(memory.available / (1024**3), 2)
            info["memory_used_percent"] = memory.percent

            # Get CPU info
            info["cpu_count"] = psutil.cpu_count()
            info["cpu_percent"] = psutil.cpu_percent()

            # Get disk info
            disk = psutil.disk_usage("/")
            info["disk_total_gb"] = round(disk.total / (1024**3), 2)
            info["disk_free_gb"] = round(disk.free / (1024**3), 2)
            info["disk_used_percent"] = round((disk.used / disk.total) * 100, 1)

            output = "💻 System Information:\n"
            for key, value in info.items():
                output += f"  {key}: {value}\n"

            return ToolResult(output=output)

        except Exception as e:
            return ToolResult(error=f"Failed to get system info: {str(e)}")

    async def _get_screen_info(self) -> ToolResult:
        """Get screen/monitor information"""
        try:
            monitors = get_monitors()

            output = "🖥️ Monitor Information:\n"
            for i, monitor in enumerate(monitors):
                output += f"  Monitor {i+1}: {monitor.width}x{monitor.height} at ({monitor.x}, {monitor.y})\n"
                if hasattr(monitor, "is_primary") and monitor.is_primary:
                    output += "    (Primary Monitor)\n"

            return ToolResult(output=output)

        except Exception as e:
            return ToolResult(error=f"Failed to get screen info: {str(e)}")

    async def _wait(self, seconds: int) -> ToolResult:
        """Wait for specified number of seconds"""
        try:
            if seconds is None or seconds <= 0:
                return ToolResult(error="Valid wait time in seconds is required")

            time.sleep(seconds)
            return ToolResult(output=f"Waited for {seconds} seconds")

        except Exception as e:
            return ToolResult(error=f"Failed to wait: {str(e)}")
