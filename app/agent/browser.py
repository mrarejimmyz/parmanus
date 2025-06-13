import json
import time
from typing import TYPE_CHECKING, Dict, List, Optional

from pydantic import Field, model_validator

from app.agent.toolcall import ToolCallAgent
from app.logger import logger
from app.prompt.browser import NEXT_STEP_PROMPT, SYSTEM_PROMPT
from app.schema import Message, ToolChoice
from app.tool import BrowserUseTool, Terminate, ToolCollection

# Avoid circular import if BrowserAgent needs BrowserContextHelper
if TYPE_CHECKING:
    from app.agent.base import BaseAgent


class BrowserContextHelper:
    """Helper class for managing browser context and state."""

    def __init__(self, agent: "BaseAgent"):
        self.agent = agent
        self._current_base64_image: Optional[str] = None
        self._last_successful_state: Optional[dict] = None

    async def get_browser_state(self) -> Optional[dict]:
        """Get current browser state with error handling and caching."""
        browser_tool = self.agent.available_tools.get_tool(BrowserUseTool().name)
        if not browser_tool or not hasattr(browser_tool, "get_current_state"):
            logger.warning("BrowserUseTool not found or doesn't have get_current_state")
            return self._last_successful_state

        try:
            result = await browser_tool.get_current_state()
            if result.error:
                logger.debug(f"Browser state error: {result.error}")
                return self._last_successful_state

            # Parse and validate state
            state = json.loads(result.output)
            if not isinstance(state, dict):
                logger.warning("Invalid browser state format")
                return self._last_successful_state

            # Cache successful state
            self._last_successful_state = state

            # Handle base64 image
            if hasattr(result, "base64_image") and result.base64_image:
                self._current_base64_image = result.base64_image
            else:
                self._current_base64_image = None

            return state

        except json.JSONDecodeError as e:
            logger.warning(f"Failed to parse browser state JSON: {e}")
            return self._last_successful_state
        except Exception as e:
            logger.debug(f"Failed to get browser state: {str(e)}")
            return self._last_successful_state

    async def format_next_step_prompt(self) -> str:
        """Format browser prompt with current state information."""
        browser_state = await self.get_browser_state()
        url_info, tabs_info, content_above_info, content_below_info = "", "", "", ""
        results_info = ""

        if browser_state and not browser_state.get("error"):
            # Format URL and title info
            url = browser_state.get("url", "N/A")
            title = browser_state.get("title", "N/A")
            url_info = f"\n   URL: {url}\n   Title: {title}"

            # Format tabs info
            tabs = browser_state.get("tabs", [])
            if tabs:
                tabs_info = f"\n   {len(tabs)} tab(s) available"

            # Format content info
            pixels_above = browser_state.get("pixels_above", 0)
            pixels_below = browser_state.get("pixels_below", 0)
            if pixels_above > 0:
                content_above_info = f" ({pixels_above} pixels)"
            if pixels_below > 0:
                content_below_info = f" ({pixels_below} pixels)"

            # Add screenshot to memory if available
            if self._current_base64_image:
                try:
                    image_message = Message.user_message(
                        content="Current browser screenshot:",
                        base64_image=self._current_base64_image,
                    )
                    self.agent.memory.add_message(image_message)
                    self._current_base64_image = None  # Consume the image
                except Exception as e:
                    logger.warning(f"Failed to add screenshot to memory: {e}")

        return NEXT_STEP_PROMPT.format(
            url_placeholder=url_info,
            tabs_placeholder=tabs_info,
            content_above_placeholder=content_above_info,
            content_below_placeholder=content_below_info,
            results_placeholder=results_info,
        )

    async def cleanup_browser(self):
        """Clean up browser resources safely."""
        try:
            browser_tool = self.agent.available_tools.get_tool(BrowserUseTool().name)
            if browser_tool and hasattr(browser_tool, "cleanup"):
                await browser_tool.cleanup()
                logger.debug("Browser cleanup completed")
        except Exception as e:
            logger.warning(f"Error during browser cleanup: {e}")


class BrowserAgent(ToolCallAgent):
    """
    A browser agent that uses the browser_use library to control a browser.asx
    This agent can navigate web pages, interact with elements, fill forms,
    extract content, and perform other browser-based actions to accomplish tasks.

    Features:
    - Robust error handling and recovery
    - Browser state caching and validation
    - Optimized screenshot handling
    - Graceful degradation on failures
    - Hallucination loop prevention
    """

    name: str = "browser"
    description: str = "A browser agent that can control a browser to accomplish tasks"
    system_prompt: str = SYSTEM_PROMPT
    next_step_prompt: str = NEXT_STEP_PROMPT
    max_observe: int = 10000
    max_steps: int = 20

    # Configure the available tools
    available_tools: ToolCollection = Field(
        default_factory=lambda: ToolCollection(BrowserUseTool(), Terminate())
    )

    # Use Auto for tool choice to allow both tool usage and free-form responses
    tool_choices: ToolChoice = ToolChoice.AUTO
    special_tool_names: list[str] = Field(default_factory=lambda: [Terminate().name])

    browser_context_helper: Optional[BrowserContextHelper] = None

    # Loop prevention tracking
    repeated_actions: Dict[str, int] = Field(default_factory=dict)
    action_timestamps: Dict[str, float] = Field(default_factory=dict)
    max_repetitions: int = 3
    repetition_window: float = 60.0  # seconds
    recent_actions: List[str] = Field(default_factory=list)
    max_recent_actions: int = 10
    hallucination_detected: bool = False

    def _extract_url(self, prompt: str) -> str:
        """Extract and format URL from user prompt (private helper)."""
        for separator in ["go to", "open", "navigate to"]:
            if separator in prompt.lower():
                url = prompt.lower().split(separator)[-1].strip()
                break
        else:
            url = prompt.strip()

        if not url.startswith(("http://", "https://")):
            url = f"https://{url}"

        if "." not in url.split("//")[-1]:
            url += ".com"

        return url

    @classmethod
    async def create(cls, **kwargs) -> "BrowserAgent":
        """Factory method to create and properly initialize a BrowserAgent instance."""
        instance = cls(**kwargs)

        # Validate browser tool availability
        browser_tool = instance.available_tools.get_tool(BrowserUseTool().name)
        if not browser_tool:
            logger.warning(
                "BrowserUseTool not available, browser functionality may be limited"
            )

        return instance

    @model_validator(mode="after")
    def initialize_helper(self) -> "BrowserAgent":
        """Initialize the browser context helper."""
        self.browser_context_helper = BrowserContextHelper(self)
        return self

    def _track_action(self, action_str: str) -> bool:
        """
        Track an action to detect repetitive patterns and hallucination loops.
        Returns True if action should be allowed, False if it's part of a loop.
        """
        current_time = time.time()

        # Add to recent actions list
        self.recent_actions.append(action_str)
        if len(self.recent_actions) > self.max_recent_actions:
            self.recent_actions.pop(0)

        # Check for repetitive patterns in recent actions
        if len(self.recent_actions) >= 3:
            last_three = self.recent_actions[-3:]
            if len(set(last_three)) == 1:  # All three are the same
                logger.warning(f"Detected repetitive action pattern: {action_str}")
                self.hallucination_detected = True
                return False

        # Clean up old timestamps
        for action in list(self.action_timestamps.keys()):
            if current_time - self.action_timestamps[action] > self.repetition_window:
                del self.action_timestamps[action]
                if action in self.repeated_actions:
                    del self.repeated_actions[action]

        # Track this action
        self.action_timestamps[action_str] = current_time
        self.repeated_actions[action_str] = self.repeated_actions.get(action_str, 0) + 1

        # Check if action is repeated too many times
        if self.repeated_actions[action_str] > self.max_repetitions:
            logger.warning(
                f"Action '{action_str}' repeated too many times ({self.repeated_actions[action_str]})"
            )
            self.hallucination_detected = True
            return False

        return True

    async def think(self) -> bool:
        """Process current state and decide next actions using tools, with browser state info added."""
        try:
            # Check if hallucination loop was detected
            if self.hallucination_detected:
                logger.warning("Hallucination loop detected, breaking execution")
                self.memory.add_message(
                    Message.assistant_message(
                        "I detected a potential hallucination loop. Stopping execution to prevent infinite loops."
                    )
                )
                self.state = "FINISHED"
                return False

            # Check for navigation commands first and handle them directly
            user_messages = [msg for msg in self.memory.messages if msg.role == "user"]
            if user_messages and any(
                cmd in user_messages[-1].content.lower()
                for cmd in ["go to", "open", "navigate to"]
            ):
                prompt = user_messages[-1].content
                url = self._extract_url(prompt)
                logger.info(f"Direct navigation requested to: {url}")

                # Force the browser tool to be used for navigation
                from app.schema import Function, ToolCall

                # Create a tool call for navigation
                nav_tool_call = ToolCall(
                    id="nav_001",
                    function=Function(
                        name="browser_use",
                        arguments=json.dumps({"action": "go_to_url", "url": url}),
                    ),
                )

                # Set this as our tool call
                self.tool_calls = [nav_tool_call]
                return True  # Indicate that a tool call has been set
            # Use simplified prompt for first few steps to avoid complexity
            if self.current_step <= 3:
                try:
                    from app.prompt.browser import SIMPLE_NEXT_STEP_PROMPT

                    # Get the original user request from memory
                    user_messages = [
                        msg for msg in self.memory.messages if msg.role == "user"
                    ]
                    task = (
                        user_messages[0].content
                        if user_messages
                        else "Navigate and analyze the website"
                    )
                    self.next_step_prompt = SIMPLE_NEXT_STEP_PROMPT.format(task=task)
                    logger.info(f"Using simplified prompt for step {self.current_step}")
                except ImportError:
                    logger.warning("SIMPLE_NEXT_STEP_PROMPT not found, using default")
            else:
                # Update next step prompt with current browser state for later steps
                if self.browser_context_helper:
                    self.next_step_prompt = (
                        await self.browser_context_helper.format_next_step_prompt()
                    )

            # Call parent think method
            result = await super().think()

            # Track actions to detect loops
            if self.tool_calls:
                for call in self.tool_calls:
                    if call.function and call.function.name == "browser_use":
                        try:
                            args = json.loads(call.function.arguments)
                            action = args.get("action", "")

                            # Create a unique action signature
                            action_signature = f"{action}"
                            if action == "extract_content" and "goal" in args:
                                action_signature += f":{args['goal']}"
                            if "selector" in args:
                                action_signature += f":{args['selector']}"

                            # Track and check if this action is part of a loop
                            if not self._track_action(action_signature):
                                logger.warning(
                                    f"Blocking repetitive action: {action_signature}"
                                )
                                self.memory.add_message(
                                    Message.assistant_message(
                                        "I detected a potential hallucination loop. Changing approach to avoid infinite loops."
                                    )
                                )
                                # Don't set hallucination_detected to True here to give it one more chance
                        except Exception as e:
                            logger.error(f"Error tracking browser action: {e}")

            return result

        except Exception as e:
            logger.error(f"Error in browser agent think method: {e}")
            # Fallback to basic thinking without browser state
            return await super().think()

    async def cleanup(self):
        """Clean up browser agent resources."""
        try:
            if self.browser_context_helper:
                await self.browser_context_helper.cleanup_browser()

            # Call parent cleanup if available
            if hasattr(super(), "cleanup"):
                await super().cleanup()

        except Exception as e:
            logger.error(f"Error during browser agent cleanup: {e}")

    async def handle_browser_error(self, error: Exception) -> bool:
        """
        Handle browser-specific errors with recovery strategies.

        Args:
            error: The error that occurred

        Returns:
            True if recovery was successful, False otherwise
        """
        logger.warning(f"Browser error encountered: {error}")

        # Try to recover browser state
        if self.browser_context_helper:
            try:
                state = await self.browser_context_helper.get_browser_state()
                if state:
                    logger.info("Browser state recovered successfully")
                    return True
            except Exception as recovery_error:
                logger.error(f"Failed to recover browser state: {recovery_error}")

        return False

    def is_browser_available(self) -> bool:
        """Check if browser functionality is available."""
        browser_tool = self.available_tools.get_tool(BrowserUseTool().name)
        return browser_tool is not None
