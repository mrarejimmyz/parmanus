"""
Advanced Automation Tool for ParManus AI
Provides comprehensive automation capabilities for complex task execution,
workflow management, and intelligent task scheduling
"""

import asyncio
import json
import os
import subprocess
import threading
import time
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import pyautogui
import schedule

from app.logger import logger
from app.tool.base import BaseTool, ToolResult


class AutomationTool(BaseTool):
    """Advanced automation tool for complex task sequences and workflows"""

    name: str = "automation"
    description: str = """Execute advanced automation sequences including:
    - Multi-step workflow creation and execution
    - Intelligent task scheduling and automation
    - Batch file processing and data operations
    - Screen automation with computer vision
    - System maintenance and cleanup tasks
    - Integration with external tools and APIs
    - Conditional logic and decision trees
    - Error handling and recovery workflows
    - Performance monitoring and optimization"""

    class Config:
        arbitrary_types_allowed = True
        extra = "allow"

    parameters: dict = {
        "type": "object",
        "properties": {
            "action": {
                "type": "string",
                "enum": [
                    "create_workflow",
                    "run_workflow",
                    "schedule_task",
                    "cancel_scheduled_task",
                    "list_scheduled_tasks",
                    "batch_process_files",
                    "system_cleanup",
                    "monitor_system",
                    "create_automation_script",
                    "run_automation_script",
                    "screen_automation",
                    "data_processing_workflow",
                    "backup_automation",
                    "maintenance_routine",
                    "performance_optimization",
                    "error_recovery_workflow",
                ],
                "description": "Automation action to perform",
            },
            "workflow_name": {
                "type": "string",
                "description": "Name of the workflow to create or execute",
            },
            "workflow_definition": {
                "type": "object",
                "description": "Workflow definition with steps and conditions",
            },
            "schedule_time": {
                "type": "string",
                "description": "Schedule time in format 'HH:MM' or cron-like expression",
            },
            "schedule_frequency": {
                "type": "string",
                "enum": ["once", "daily", "weekly", "monthly", "hourly"],
                "description": "Frequency for scheduled tasks",
            },
            "file_patterns": {
                "type": "array",
                "items": {"type": "string"},
                "description": "File patterns for batch processing",
            },
            "source_directory": {
                "type": "string",
                "description": "Source directory for file operations",
            },
            "target_directory": {
                "type": "string",
                "description": "Target directory for file operations",
            },
            "script_content": {
                "type": "string",
                "description": "Script content for automation",
            },
            "conditions": {
                "type": "object",
                "description": "Conditions for conditional automation",
            },
            "error_handling": {
                "type": "object",
                "description": "Error handling configuration",
            },
            "parameters": {
                "type": "object",
                "description": "Additional parameters for automation actions",
            },
        },
        "required": ["action"],
    }

    def __init__(self):
        super().__init__()
        self.workflows = {}
        self.scheduled_tasks = {}
        self.automation_scripts = {}
        self.task_scheduler = None
        self._init_scheduler()

    def _init_scheduler(self):
        """Initialize the task scheduler"""
        try:
            # Start scheduler in background thread
            self.task_scheduler = threading.Thread(
                target=self._run_scheduler, daemon=True
            )
            self.task_scheduler.start()
            logger.info("Automation scheduler initialized")
        except Exception as e:
            logger.error(f"Failed to initialize scheduler: {e}")

    def _run_scheduler(self):
        """Run the task scheduler"""
        while True:
            try:
                schedule.run_pending()
                time.sleep(1)
            except Exception as e:
                logger.error(f"Scheduler error: {e}")
                time.sleep(5)

    async def execute(
        self,
        action: str,
        workflow_name: Optional[str] = None,
        workflow_definition: Optional[Dict[str, Any]] = None,
        schedule_time: Optional[str] = None,
        schedule_frequency: str = "once",
        file_patterns: Optional[List[str]] = None,
        source_directory: Optional[str] = None,
        target_directory: Optional[str] = None,
        script_content: Optional[str] = None,
        conditions: Optional[Dict[str, Any]] = None,
        error_handling: Optional[Dict[str, Any]] = None,
        parameters: Optional[Dict[str, Any]] = None,
        **kwargs,
    ) -> ToolResult:
        """Execute automation actions"""
        try:
            if parameters is None:
                parameters = {}

            logger.info(f"🤖 Automation action: {action}")

            # Workflow management
            if action == "create_workflow":
                return await self._create_workflow(workflow_name, workflow_definition)
            elif action == "run_workflow":
                return await self._run_workflow(workflow_name, parameters)

            # Task scheduling
            elif action == "schedule_task":
                return await self._schedule_task(
                    workflow_name, schedule_time, schedule_frequency, parameters
                )
            elif action == "cancel_scheduled_task":
                return await self._cancel_scheduled_task(workflow_name)
            elif action == "list_scheduled_tasks":
                return await self._list_scheduled_tasks()

            # File operations
            elif action == "batch_process_files":
                return await self._batch_process_files(
                    file_patterns, source_directory, target_directory, parameters
                )

            # System operations
            elif action == "system_cleanup":
                return await self._system_cleanup(parameters)
            elif action == "monitor_system":
                return await self._monitor_system(parameters)
            elif action == "maintenance_routine":
                return await self._maintenance_routine(parameters)
            elif action == "performance_optimization":
                return await self._performance_optimization(parameters)

            # Script automation
            elif action == "create_automation_script":
                return await self._create_automation_script(
                    workflow_name, script_content, parameters
                )
            elif action == "run_automation_script":
                return await self._run_automation_script(workflow_name, parameters)

            # Advanced automation
            elif action == "screen_automation":
                return await self._screen_automation(workflow_definition, parameters)
            elif action == "data_processing_workflow":
                return await self._data_processing_workflow(
                    workflow_definition, parameters
                )
            elif action == "backup_automation":
                return await self._backup_automation(
                    source_directory, target_directory, parameters
                )
            elif action == "error_recovery_workflow":
                return await self._error_recovery_workflow(
                    workflow_definition, parameters
                )

            else:
                return ToolResult(error=f"Unknown automation action: {action}")

        except Exception as e:
            logger.error(f"Automation error: {e}")
            return ToolResult(error=f"Automation failed: {str(e)}")

    # Workflow management methods
    async def _create_workflow(
        self, workflow_name: str, workflow_definition: Dict[str, Any]
    ) -> ToolResult:
        """Create a new workflow"""
        try:
            if not workflow_name:
                return ToolResult(error="Workflow name is required")
            if not workflow_definition:
                return ToolResult(error="Workflow definition is required")

            # Validate workflow definition
            if "steps" not in workflow_definition:
                return ToolResult(error="Workflow must have 'steps' defined")

            # Store workflow
            self.workflows[workflow_name] = {
                "definition": workflow_definition,
                "created_at": datetime.now().isoformat(),
                "runs": 0,
                "last_run": None,
                "status": "ready",
            }

            # Save to file for persistence
            workflow_dir = Path("workflows")
            workflow_dir.mkdir(exist_ok=True)

            workflow_file = workflow_dir / f"{workflow_name}.json"
            with open(workflow_file, "w") as f:
                json.dump(self.workflows[workflow_name], f, indent=2)

            return ToolResult(
                output=f"Created workflow '{workflow_name}' with {len(workflow_definition['steps'])} steps"
            )

        except Exception as e:
            return ToolResult(error=f"Failed to create workflow: {str(e)}")

    async def _run_workflow(
        self, workflow_name: str, parameters: Dict[str, Any]
    ) -> ToolResult:
        """Run a workflow"""
        try:
            if not workflow_name:
                return ToolResult(error="Workflow name is required")

            # Load workflow if not in memory
            if workflow_name not in self.workflows:
                workflow_file = Path("workflows") / f"{workflow_name}.json"
                if workflow_file.exists():
                    with open(workflow_file, "r") as f:
                        self.workflows[workflow_name] = json.load(f)
                else:
                    return ToolResult(error=f"Workflow '{workflow_name}' not found")

            workflow = self.workflows[workflow_name]
            definition = workflow["definition"]
            steps = definition["steps"]

            # Update workflow status
            workflow["status"] = "running"
            workflow["last_run"] = datetime.now().isoformat()
            workflow["runs"] += 1

            results = []

            for i, step in enumerate(steps):
                step_name = step.get("name", f"Step {i+1}")
                step_type = step.get("type", "command")
                step_params = step.get("parameters", {})

                # Merge global parameters
                step_params.update(parameters)

                logger.info(f"Executing workflow step: {step_name}")

                try:
                    # Execute step based on type
                    if step_type == "command":
                        result = await self._execute_command_step(step_params)
                    elif step_type == "file_operation":
                        result = await self._file_operation_step(step_params)
                    elif step_type == "mouse_action":
                        result = await self._mouse_action_step(step_params)
                    elif step_type == "keyboard_action":
                        result = await self._keyboard_action_step(step_params)
                    elif step_type == "wait":
                        result = await self._wait_step(step_params)
                    elif step_type == "condition":
                        result = await self._condition_step(step_params)
                    elif step_type == "screenshot":
                        result = await self._screenshot_step(step_params)
                    else:
                        result = f"❌ Unknown step type: {step_type}"

                    results.append(f"✅ {step_name}: {result}")

                except Exception as e:
                    error_msg = f"❌ {step_name}: {str(e)}"
                    results.append(error_msg)

                    # Check error handling
                    if step.get("continue_on_error", False):
                        continue
                    else:
                        break

            # Update workflow status
            workflow["status"] = "completed"

            return ToolResult(
                output=f"Workflow '{workflow_name}' completed:\n" + "\n".join(results)
            )

        except Exception as e:
            if workflow_name in self.workflows:
                self.workflows[workflow_name]["status"] = "failed"
            return ToolResult(error=f"Failed to run workflow: {str(e)}")

    # Step execution methods
    async def _execute_command_step(self, params: Dict[str, Any]) -> str:
        """Execute a command step"""
        command = params.get("command")
        if not command:
            raise ValueError("Command is required")

        result = subprocess.run(
            command,
            shell=True,
            capture_output=True,
            text=True,
            timeout=params.get("timeout", 30),
        )

        if result.returncode == 0:
            return f"Command executed successfully: {result.stdout.strip()}"
        else:
            raise Exception(f"Command failed: {result.stderr.strip()}")

    async def _file_operation_step(self, params: Dict[str, Any]) -> str:
        """Execute a file operation step"""
        operation = params.get("operation")
        source = params.get("source")
        target = params.get("target")

        if operation == "copy":
            import shutil

            shutil.copy2(source, target)
            return f"Copied {source} to {target}"
        elif operation == "move":
            import shutil

            shutil.move(source, target)
            return f"Moved {source} to {target}"
        elif operation == "delete":
            os.remove(source)
            return f"Deleted {source}"
        elif operation == "create_dir":
            os.makedirs(source, exist_ok=True)
            return f"Created directory {source}"
        else:
            raise ValueError(f"Unknown file operation: {operation}")

    async def _mouse_action_step(self, params: Dict[str, Any]) -> str:
        """Execute a mouse action step"""
        action = params.get("action")
        x = params.get("x", 0)
        y = params.get("y", 0)

        if action == "click":
            pyautogui.click(x, y)
            return f"Clicked at ({x}, {y})"
        elif action == "move":
            pyautogui.moveTo(x, y)
            return f"Moved mouse to ({x}, {y})"
        elif action == "scroll":
            amount = params.get("amount", 3)
            pyautogui.scroll(amount, x, y)
            return f"Scrolled {amount} at ({x}, {y})"
        else:
            raise ValueError(f"Unknown mouse action: {action}")

    async def _keyboard_action_step(self, params: Dict[str, Any]) -> str:
        """Execute a keyboard action step"""
        action = params.get("action")

        if action == "type":
            text = params.get("text", "")
            pyautogui.typewrite(text)
            return f"Typed: {text[:50]}..."
        elif action == "press":
            key = params.get("key", "")
            pyautogui.press(key)
            return f"Pressed key: {key}"
        elif action == "hotkey":
            keys = params.get("keys", [])
            pyautogui.hotkey(*keys)
            return f"Pressed hotkey: {'+'.join(keys)}"
        else:
            raise ValueError(f"Unknown keyboard action: {action}")

    async def _wait_step(self, params: Dict[str, Any]) -> str:
        """Execute a wait step"""
        duration = params.get("duration", 1)
        await asyncio.sleep(duration)
        return f"Waited {duration} seconds"

    async def _condition_step(self, params: Dict[str, Any]) -> str:
        """Execute a conditional step"""
        condition_type = params.get("condition_type")

        if condition_type == "file_exists":
            file_path = params.get("file_path")
            exists = os.path.exists(file_path)
            return f"File {file_path} exists: {exists}"
        elif condition_type == "process_running":
            process_name = params.get("process_name")
            import psutil

            running = any(proc.name() == process_name for proc in psutil.process_iter())
            return f"Process {process_name} running: {running}"
        else:
            raise ValueError(f"Unknown condition type: {condition_type}")

    async def _screenshot_step(self, params: Dict[str, Any]) -> str:
        """Execute a screenshot step"""
        filename = params.get("filename", f"screenshot_{int(time.time())}.png")
        screenshot = pyautogui.screenshot()
        screenshot.save(filename)
        return f"Screenshot saved: {filename}"

    # Task scheduling methods
    async def _schedule_task(
        self,
        task_name: str,
        schedule_time: str,
        frequency: str,
        parameters: Dict[str, Any],
    ) -> ToolResult:
        """Schedule a task"""
        try:
            if not task_name:
                return ToolResult(error="Task name is required")
            if not schedule_time:
                return ToolResult(error="Schedule time is required")

            # Create scheduled task
            task_info = {
                "name": task_name,
                "schedule_time": schedule_time,
                "frequency": frequency,
                "parameters": parameters,
                "created_at": datetime.now().isoformat(),
                "next_run": None,
                "runs": 0,
            }

            def run_scheduled_task():
                try:
                    # Run the workflow or task
                    asyncio.create_task(self._run_workflow(task_name, parameters))
                    task_info["runs"] += 1
                    task_info["last_run"] = datetime.now().isoformat()
                    logger.info(f"Scheduled task '{task_name}' executed")
                except Exception as e:
                    logger.error(f"Scheduled task '{task_name}' failed: {e}")

            # Schedule based on frequency
            if frequency == "daily":
                schedule.every().day.at(schedule_time).do(run_scheduled_task)
            elif frequency == "weekly":
                schedule.every().week.at(schedule_time).do(run_scheduled_task)
            elif frequency == "hourly":
                schedule.every().hour.do(run_scheduled_task)
            elif frequency == "once":
                # Parse time and schedule for today or tomorrow
                target_time = datetime.strptime(schedule_time, "%H:%M").time()
                now = datetime.now()
                target_datetime = datetime.combine(now.date(), target_time)

                if target_datetime <= now:
                    target_datetime += timedelta(days=1)

                def run_once():
                    run_scheduled_task()
                    schedule.cancel_job(task_info["job"])

                task_info["job"] = schedule.every().day.at(schedule_time).do(run_once)
            else:
                return ToolResult(error=f"Unknown frequency: {frequency}")

            self.scheduled_tasks[task_name] = task_info

            return ToolResult(
                output=f"Scheduled task '{task_name}' for {frequency} at {schedule_time}"
            )

        except Exception as e:
            return ToolResult(error=f"Failed to schedule task: {str(e)}")

    async def _cancel_scheduled_task(self, task_name: str) -> ToolResult:
        """Cancel a scheduled task"""
        try:
            if task_name not in self.scheduled_tasks:
                return ToolResult(error=f"Scheduled task '{task_name}' not found")

            # Cancel the scheduled job
            task_info = self.scheduled_tasks[task_name]
            if "job" in task_info:
                schedule.cancel_job(task_info["job"])

            del self.scheduled_tasks[task_name]

            return ToolResult(output=f"Cancelled scheduled task '{task_name}'")

        except Exception as e:
            return ToolResult(error=f"Failed to cancel scheduled task: {str(e)}")

    async def _list_scheduled_tasks(self) -> ToolResult:
        """List all scheduled tasks"""
        try:
            if not self.scheduled_tasks:
                return ToolResult(output="No scheduled tasks found")

            output = "📅 Scheduled Tasks:\n"
            for name, info in self.scheduled_tasks.items():
                output += f"  {name}: {info['frequency']} at {info['schedule_time']} (runs: {info['runs']})\n"

            return ToolResult(output=output)

        except Exception as e:
            return ToolResult(error=f"Failed to list scheduled tasks: {str(e)}")

    # File processing methods
    async def _batch_process_files(
        self,
        patterns: List[str],
        source_dir: str,
        target_dir: str,
        parameters: Dict[str, Any],
    ) -> ToolResult:
        """Batch process files matching patterns"""
        try:
            if not patterns:
                return ToolResult(error="File patterns are required")
            if not source_dir:
                return ToolResult(error="Source directory is required")

            import glob

            processed_files = []
            operation = parameters.get("operation", "copy")

            for pattern in patterns:
                search_pattern = os.path.join(source_dir, pattern)
                files = glob.glob(search_pattern, recursive=True)

                for file_path in files:
                    try:
                        if operation == "copy" and target_dir:
                            import shutil

                            target_path = os.path.join(
                                target_dir, os.path.basename(file_path)
                            )
                            os.makedirs(target_dir, exist_ok=True)
                            shutil.copy2(file_path, target_path)
                            processed_files.append(
                                f"Copied: {file_path} -> {target_path}"
                            )
                        elif operation == "move" and target_dir:
                            import shutil

                            target_path = os.path.join(
                                target_dir, os.path.basename(file_path)
                            )
                            os.makedirs(target_dir, exist_ok=True)
                            shutil.move(file_path, target_path)
                            processed_files.append(
                                f"Moved: {file_path} -> {target_path}"
                            )
                        elif operation == "delete":
                            os.remove(file_path)
                            processed_files.append(f"Deleted: {file_path}")
                        elif operation == "compress":
                            import zipfile

                            zip_path = file_path + ".zip"
                            with zipfile.ZipFile(zip_path, "w") as zipf:
                                zipf.write(file_path, os.path.basename(file_path))
                            processed_files.append(
                                f"Compressed: {file_path} -> {zip_path}"
                            )

                    except Exception as e:
                        processed_files.append(
                            f"Error processing {file_path}: {str(e)}"
                        )

            return ToolResult(
                output=f"Batch processed {len(processed_files)} files:\n"
                + "\n".join(processed_files[:20])
            )

        except Exception as e:
            return ToolResult(error=f"Failed to batch process files: {str(e)}")

    # System maintenance methods
    async def _system_cleanup(self, parameters: Dict[str, Any]) -> ToolResult:
        """Perform system cleanup tasks"""
        try:
            cleanup_tasks = []

            # Clean temp files
            if parameters.get("clean_temp", True):
                import tempfile

                temp_dir = tempfile.gettempdir()
                temp_files = 0
                for root, dirs, files in os.walk(temp_dir):
                    for file in files:
                        try:
                            file_path = os.path.join(root, file)
                            if (
                                os.path.getmtime(file_path) < time.time() - 86400
                            ):  # 1 day old
                                os.remove(file_path)
                                temp_files += 1
                        except:
                            pass
                cleanup_tasks.append(f"Cleaned {temp_files} temporary files")

            # Empty recycle bin (Windows)
            if parameters.get("empty_recycle_bin", True) and os.name == "nt":
                try:
                    import winshell

                    winshell.recycle_bin().empty(
                        confirm=False, show_progress=False, sound=False
                    )
                    cleanup_tasks.append("Emptied recycle bin")
                except:
                    cleanup_tasks.append("Could not empty recycle bin")

            # Clear browser cache (basic)
            if parameters.get("clear_browser_cache", False):
                browser_dirs = [
                    os.path.expanduser(
                        "~\\AppData\\Local\\Google\\Chrome\\User Data\\Default\\Cache"
                    ),
                    os.path.expanduser(
                        "~\\AppData\\Local\\Microsoft\\Edge\\User Data\\Default\\Cache"
                    ),
                ]

                for cache_dir in browser_dirs:
                    if os.path.exists(cache_dir):
                        try:
                            import shutil

                            shutil.rmtree(cache_dir)
                            os.makedirs(cache_dir)
                            cleanup_tasks.append(
                                f"Cleared cache: {os.path.basename(os.path.dirname(cache_dir))}"
                            )
                        except:
                            cleanup_tasks.append(
                                f"Could not clear cache: {os.path.basename(os.path.dirname(cache_dir))}"
                            )

            return ToolResult(
                output="🧹 System Cleanup Completed:\n" + "\n".join(cleanup_tasks)
            )

        except Exception as e:
            return ToolResult(error=f"Failed to perform system cleanup: {str(e)}")

    async def _monitor_system(self, parameters: Dict[str, Any]) -> ToolResult:
        """Monitor system performance and resources"""
        try:
            import psutil

            # Get system information
            cpu_percent = psutil.cpu_percent(interval=1)
            memory = psutil.virtual_memory()
            disk = psutil.disk_usage("/")

            # Get network info
            network = psutil.net_io_counters()

            # Get top processes
            processes = []
            for proc in psutil.process_iter(
                ["pid", "name", "cpu_percent", "memory_percent"]
            ):
                try:
                    processes.append(proc.info)
                except (psutil.NoSuchProcess, psutil.AccessDenied):
                    pass

            processes.sort(key=lambda x: x["cpu_percent"] or 0, reverse=True)
            top_processes = processes[:5]

            monitor_info = f"""📊 System Monitor Report:

💻 CPU Usage: {cpu_percent}%
🧠 Memory: {memory.percent}% used ({memory.used // (1024**3)} GB / {memory.total // (1024**3)} GB)
💾 Disk: {(disk.used / disk.total) * 100:.1f}% used ({disk.used // (1024**3)} GB / {disk.total // (1024**3)} GB)
🌐 Network: ↑{network.bytes_sent // (1024**2)} MB sent, ↓{network.bytes_recv // (1024**2)} MB received

🔥 Top CPU Processes:"""

            for proc in top_processes:
                monitor_info += f"\n  {proc['name']} (PID: {proc['pid']}): {proc['cpu_percent'] or 0:.1f}% CPU"

            return ToolResult(output=monitor_info)

        except Exception as e:
            return ToolResult(error=f"Failed to monitor system: {str(e)}")

    # Advanced automation methods
    async def _backup_automation(
        self, source_dir: str, target_dir: str, parameters: Dict[str, Any]
    ) -> ToolResult:
        """Automated backup system"""
        try:
            if not source_dir or not target_dir:
                return ToolResult(error="Source and target directories are required")

            import shutil
            import zipfile
            from datetime import datetime

            # Create backup with timestamp
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            backup_name = f"backup_{timestamp}"

            compression = parameters.get("compression", "zip")

            if compression == "zip":
                backup_file = os.path.join(target_dir, f"{backup_name}.zip")
                os.makedirs(target_dir, exist_ok=True)

                with zipfile.ZipFile(backup_file, "w", zipfile.ZIP_DEFLATED) as zipf:
                    for root, dirs, files in os.walk(source_dir):
                        for file in files:
                            file_path = os.path.join(root, file)
                            arcname = os.path.relpath(file_path, source_dir)
                            zipf.write(file_path, arcname)

                return ToolResult(output=f"✅ Backup created: {backup_file}")

            else:
                # Simple copy backup
                backup_dir = os.path.join(target_dir, backup_name)
                shutil.copytree(source_dir, backup_dir)
                return ToolResult(output=f"✅ Backup created: {backup_dir}")

        except Exception as e:
            return ToolResult(error=f"Failed to create backup: {str(e)}")

    async def _performance_optimization(self, parameters: Dict[str, Any]) -> ToolResult:
        """Perform system performance optimizations"""
        try:
            optimizations = []

            # Kill high CPU processes if requested
            if parameters.get("kill_high_cpu", False):
                import psutil

                cpu_threshold = parameters.get("cpu_threshold", 80)

                for proc in psutil.process_iter(["pid", "name", "cpu_percent"]):
                    try:
                        if (
                            proc.info["cpu_percent"]
                            and proc.info["cpu_percent"] > cpu_threshold
                        ):
                            proc.terminate()
                            optimizations.append(
                                f"Terminated high CPU process: {proc.info['name']}"
                            )
                    except (psutil.NoSuchProcess, psutil.AccessDenied):
                        pass

            # Clear memory cache
            if parameters.get("clear_memory_cache", True):
                try:
                    # Force garbage collection
                    import gc

                    gc.collect()
                    optimizations.append("Cleared memory cache")
                except:
                    pass

            # Optimize startup programs (Windows)
            if parameters.get("optimize_startup", False) and os.name == "nt":
                try:
                    # This would require more complex registry operations
                    optimizations.append(
                        "Startup optimization requires administrator privileges"
                    )
                except:
                    pass

            return ToolResult(
                output="⚡ Performance Optimization Completed:\n"
                + "\n".join(optimizations)
            )

        except Exception as e:
            return ToolResult(error=f"Failed to optimize performance: {str(e)}")
