import asyncio
import base64
import json
import re
import time
from typing import Dict, Generic, List, Optional, TypeVar

from browser_use import Browser as BrowserUseBrowser
from browser_use import BrowserConfig
from browser_use.browser.context import BrowserContext, BrowserContextConfig
from browser_use.dom.service import DomService
from pydantic import Field, field_validator
from pydantic_core.core_schema import ValidationInfo

from app.config import config
from app.llm import LLM
from app.logger import logger
from app.tool.base import BaseTool, ToolResult
from app.tool.web_search import WebSearch
from app.tool.enhanced_browser import EnhancedContentExtractor


_BROWSER_DESCRIPTION = """\
A powerful browser automation tool that allows interaction with web pages through various actions.
* This tool provides commands for controlling a browser session, navigating web pages, and extracting information
* It maintains state across calls, keeping the browser session alive until explicitly closed
* Use this when you need to browse websites, fill forms, click buttons, extract content, or perform web searches
* Each action requires specific parameters as defined in the tool's dependencies

Key capabilities include:
* Navigation: Go to specific URLs, go back, search the web, or refresh pages
* Interaction: Click elements, input text, select from dropdowns, send keyboard commands
* Scrolling: Scroll up/down by pixel amount or scroll to specific text
* Content extraction: Extract and analyze content from web pages based on specific goals
* Tab management: Switch between tabs, open new tabs, or close tabs

Note: When using element indices, refer to the numbered elements shown in the current browser state.
"""

Context = TypeVar("Context")


# Track selector usage to prevent hallucination loops
class SelectorTracker:
    def __init__(self, max_retries: int = 3):
        self.selector_counts: Dict[str, int] = {}
        self.max_retries = max_retries
        self.recent_selectors: List[str] = []
        self.max_recent = 10

    def track_selector(self, selector: str) -> bool:
        """
        Track a selector usage and determine if it's being used too many times
        Returns True if the selector should be allowed, False if it's being used too much
        """
        # Add to recent selectors list
        self.recent_selectors.append(selector)
        if len(self.recent_selectors) > self.max_recent:
            self.recent_selectors.pop(0)

        # Check for repetitive pattern
        if len(self.recent_selectors) >= 3:
            last_three = self.recent_selectors[-3:]
            if len(set(last_three)) == 1:  # All three are the same
                return False

        # Track individual selector usage
        if selector in self.selector_counts:
            self.selector_counts[selector] += 1
        else:
            self.selector_counts[selector] = 1

        return self.selector_counts[selector] <= self.max_retries

    def is_valid_selector(self, selector: str) -> bool:
        """
        Validate if a selector is properly formatted
        """
        # Basic validation for CSS selectors
        if not selector or not isinstance(selector, str):
            return False

        # Check for common hallucinated patterns (random strings of letters/numbers)
        if re.match(r"^\.?[a-z0-9]{10,}$", selector):
            return False

        # Check for obviously invalid selectors
        invalid_patterns = [
            r"\.rhp\d+[a-z]+",  # Matches patterns like .rhp90mdlnikmevrp
            r"\.random",
            r"\.undefined",
            r"\.null",
        ]

        for pattern in invalid_patterns:
            if re.match(pattern, selector):
                return False

        return True


class BrowserUseTool(BaseTool, Generic[Context]):
    name: str = "browser_use"
    description: str = _BROWSER_DESCRIPTION
    parameters: dict = {
        "type": "object",
        "properties": {
            "action": {
                "type": "string",
                "enum": [
                    "go_to_url",
                    "click_element",
                    "input_text",
                    "scroll_down",
                    "scroll_up",
                    "scroll_to_text",
                    "send_keys",
                    "get_dropdown_options",
                    "select_dropdown_option",
                    "go_back",
                    "web_search",
                    "wait",
                    "extract_content",
                    "switch_tab",
                    "open_tab",
                    "close_tab",
                ],
                "description": "The browser action to perform",
            },
            "url": {
                "type": "string",
                "description": "URL for 'go_to_url' or 'open_tab' actions",
            },
            "index": {
                "type": "integer",
                "description": "Element index for 'click_element', 'input_text', 'get_dropdown_options', or 'select_dropdown_option' actions",
            },
            "text": {
                "type": "string",
                "description": "Text for 'input_text', 'scroll_to_text', or 'select_dropdown_option' actions",
            },
            "scroll_amount": {
                "type": "integer",
                "description": "Pixels to scroll (positive for down, negative for up) for 'scroll_down' or 'scroll_up' actions",
            },
            "tab_id": {
                "type": "integer",
                "description": "Tab ID for 'switch_tab' action",
            },
            "query": {
                "type": "string",
                "description": "Search query for 'web_search' action",
            },
            "goal": {
                "type": "string",
                "description": "Extraction goal for 'extract_content' action",
            },
            "keys": {
                "type": "string",
                "description": "Keys to send for 'send_keys' action",
            },
            "seconds": {
                "type": "integer",
                "description": "Seconds to wait for 'wait' action",
            },
            "selector": {
                "type": "string",
                "description": "CSS selector for element targeting (optional, use with caution)",
            },
        },
        "required": ["action"],
        "dependencies": {
            "go_to_url": ["url"],
            "click_element": ["index"],
            "input_text": ["index", "text"],
            "switch_tab": ["tab_id"],
            "open_tab": ["url"],
            "scroll_down": ["scroll_amount"],
            "scroll_up": ["scroll_amount"],
            "scroll_to_text": ["text"],
            "send_keys": ["keys"],
            "get_dropdown_options": ["index"],
            "select_dropdown_option": ["index", "text"],
            "go_back": [],
            "web_search": ["query"],
            "wait": ["seconds"],
            "extract_content": ["goal"],
        },
    }

    lock: asyncio.Lock = Field(default_factory=asyncio.Lock)
    browser: Optional[BrowserUseBrowser] = Field(default=None, exclude=True)
    context: Optional[BrowserContext] = Field(default=None, exclude=True)
    dom_service: Optional[DomService] = Field(default=None, exclude=True)
    web_search_tool: WebSearch = Field(default_factory=WebSearch, exclude=True)

    # Add selector tracker to prevent hallucination loops
    selector_tracker: SelectorTracker = Field(
        default_factory=SelectorTracker, exclude=True
    )
    
    # Add enhanced content extractor for JavaScript support
    enhanced_content_extractor: Optional[EnhancedContentExtractor] = Field(default=None, exclude=True)

    # Action deduplication system to prevent loops
    last_action: Optional[str] = Field(default=None, exclude=True)
    last_action_time: float = Field(default=0.0, exclude=True)
    action_cooldown: float = Field(default=2.0, exclude=True)  # seconds
    
    # EMERGENCY FIX: Add action_history field properly
    action_history: List[str] = Field(default_factory=list, exclude=True)

    # Context for generic functionality
    tool_context: Optional[Context] = Field(default=None, exclude=True)

    llm: Optional[LLM] = Field(default_factory=LLM)

    @field_validator("parameters", mode="before")
    def validate_parameters(cls, v: dict, info: ValidationInfo) -> dict:
        if not v:
            raise ValueError("Parameters cannot be empty")
        return v

    async def _ensure_browser_initialized(self) -> BrowserContext:
        """Ensure browser and context are initialized."""
        if self.browser is None:
            browser_config_kwargs = {"headless": False, "disable_security": True}

            if config.browser_config:
                from browser_use.browser.browser import ProxySettings

                # handle proxy settings.
                if config.browser_config.proxy and config.browser_config.proxy.server:
                    browser_config_kwargs["proxy"] = ProxySettings(
                        server=config.browser_config.proxy.server,
                        username=config.browser_config.proxy.username,
                        password=config.browser_config.proxy.password,
                    )

                browser_attrs = [
                    "headless",
                    "disable_security",
                    "extra_chromium_args",
                    "chrome_instance_path",
                    "wss_url",
                    "cdp_url",
                ]

                for attr in browser_attrs:
                    value = getattr(config.browser_config, attr, None)
                    if value is not None:
                        if not isinstance(value, list) or value:
                            browser_config_kwargs[attr] = value

            self.browser = BrowserUseBrowser(BrowserConfig(**browser_config_kwargs))

        if self.context is None:
            context_config = BrowserContextConfig()

            # if there is context config in the config, use it.
            if (
                config.browser_config
                and hasattr(config.browser_config, "new_context_config")
                and config.browser_config.new_context_config
            ):
                context_config = config.browser_config.new_context_config

            self.context = await self.browser.new_context(context_config)
            self.dom_service = DomService(await self.context.get_current_page())

        # Initialize enhanced content extractor if not already done
        if self.enhanced_content_extractor is None:
            self.enhanced_content_extractor = EnhancedContentExtractor(self.llm)

        return self.context

    async def execute(
        self,
        action: str,
        url: Optional[str] = None,
        index: Optional[int] = None,
        text: Optional[str] = None,
        scroll_amount: Optional[int] = None,
        tab_id: Optional[int] = None,
        query: Optional[str] = None,
        goal: Optional[str] = None,
        keys: Optional[str] = None,
        seconds: Optional[int] = None,
        selector: Optional[str] = None,
        **kwargs,
    ) -> ToolResult:
        """
        Execute a specified browser action.

        Args:
            action: The browser action to perform
            url: URL for navigation or new tab
            index: Element index for click or input actions
            text: Text for input action or search query
            scroll_amount: Pixels to scroll for scroll action
            tab_id: Tab ID for switch_tab action
            query: Search query for Google search
            goal: Extraction goal for content extraction
            keys: Keys to send for keyboard actions
            seconds: Seconds to wait
            selector: CSS selector (optional)
            **kwargs: Additional arguments

        Returns:
            ToolResult with the action's output or error
        """
        async with self.lock:
            try:
                # Action deduplication check to prevent loops - ENHANCED VERSION
                current_time = time.time()
                current_action_key = f"{action}:{url or ''}"
                
                # EMERGENCY FIX: Enhanced rate limiting for browser actions
                if (self.last_action == current_action_key and 
                    current_time - self.last_action_time < self.action_cooldown):
                    # Add exponential backoff for repeated actions
                    wait_time = min(self.action_cooldown * 2, 10)  # Max 10 seconds
                    return ToolResult(
                        error=f"Action '{action}' was just performed. Please wait {wait_time:.1f} seconds before retrying or try a different approach."
                    )
                
                # EMERGENCY FIX: Detect infinite loops of same action - SIMPLIFIED VERSION
                self.action_history.append(current_action_key)
                # Keep only last 5 actions
                if len(self.action_history) > 5:
                    self.action_history.pop(0)
                
                # Check for repeated patterns
                if len(self.action_history) >= 3:
                    if all(action_key == current_action_key for action_key in self.action_history[-3:]):
                        return ToolResult(
                            error=f"Detected infinite loop: action '{action}' repeated 3+ times. Please try a completely different approach."
                        )
                
                # Update action tracking
                self.last_action = current_action_key
                self.last_action_time = current_time

                # Validate selector if provided to prevent hallucination loops
                if selector:
                    if not self.selector_tracker.is_valid_selector(selector):
                        return ToolResult(
                            error=f"Invalid selector format: '{selector}'. Please use valid CSS selectors."
                        )

                    if not self.selector_tracker.track_selector(selector):
                        return ToolResult(
                            error=f"Selector '{selector}' has been used too many times. Please try a different approach."
                        )

                context = await self._ensure_browser_initialized()

                # Get max content length from config
                max_content_length = getattr(
                    config.browser_config, "max_content_length", 2000
                )

                # Navigation actions
                if action == "go_to_url":
                    if not url:
                        return ToolResult(
                            error="URL is required for 'go_to_url' action"
                        )
                    page = await context.get_current_page()
                    await page.goto(url)
                    await page.wait_for_load_state()
                    return ToolResult(output=f"Navigated to {url}")

                elif action == "go_back":
                    await context.go_back()
                    return ToolResult(output="Navigated back")

                elif action == "refresh":
                    await context.refresh_page()
                    return ToolResult(output="Refreshed current page")

                elif action == "web_search":
                    if not query:
                        return ToolResult(
                            error="Query is required for 'web_search' action"
                        )
                    # Execute the web search and return results directly without browser navigation
                    search_response = await self.web_search_tool.execute(
                        query=query, fetch_content=True, num_results=1
                    )
                    # Navigate to the first search result
                    first_search_result = search_response.results[0]
                    url_to_navigate = first_search_result.url

                    page = await context.get_current_page()
                    await page.goto(url_to_navigate)
                    await page.wait_for_load_state()

                    return search_response

                # Element interaction actions
                elif action == "click_element":
                    if index is None:
                        return ToolResult(
                            error="Index is required for 'click_element' action"
                        )
                    element = await context.get_dom_element_by_index(index)
                    if not element:
                        return ToolResult(error=f"Element with index {index} not found")
                    download_path = await context._click_element_node(element)
                    output = f"Clicked element at index {index}"
                    if download_path:
                        output += f" - Downloaded file to {download_path}"
                    return ToolResult(output=output)

                elif action == "input_text":
                    if index is None or not text:
                        return ToolResult(
                            error="Index and text are required for 'input_text' action"
                        )
                    element = await context.get_dom_element_by_index(index)
                    if not element:
                        return ToolResult(error=f"Element with index {index} not found")
                    await context._input_text_element_node(element, text)
                    return ToolResult(
                        output=f"Input '{text}' into element at index {index}"
                    )

                elif action == "scroll_down" or action == "scroll_up":
                    direction = 1 if action == "scroll_down" else -1
                    amount = (
                        scroll_amount
                        if scroll_amount is not None
                        else context.config.browser_window_size["height"]
                    )
                    await context.execute_javascript(
                        f"window.scrollBy(0, {direction * amount});"
                    )
                    return ToolResult(
                        output=f"Scrolled {'down' if direction > 0 else 'up'} by {amount} pixels"
                    )

                elif action == "scroll_to_text":
                    if not text:
                        return ToolResult(
                            error="Text is required for 'scroll_to_text' action"
                        )
                    page = await context.get_current_page()
                    try:
                        locator = page.get_by_text(text, exact=False)
                        await locator.scroll_into_view_if_needed()
                        return ToolResult(output=f"Scrolled to text: '{text}'")
                    except Exception as e:
                        return ToolResult(error=f"Failed to scroll to text: {str(e)}")

                elif action == "send_keys":
                    if not keys:
                        return ToolResult(
                            error="Keys are required for 'send_keys' action"
                        )
                    page = await context.get_current_page()
                    await page.keyboard.press(keys)
                    return ToolResult(output=f"Sent keys: {keys}")

                elif action == "get_dropdown_options":
                    if index is None:
                        return ToolResult(
                            error="Index is required for 'get_dropdown_options' action"
                        )
                    element = await context.get_dom_element_by_index(index)
                    if not element:
                        return ToolResult(error=f"Element with index {index} not found")
                    page = await context.get_current_page()
                    options = await page.evaluate(
                        """
                        (xpath) => {
                            const select = document.evaluate(xpath, document, null,
                                XPathResult.FIRST_ORDERED_NODE_TYPE, null).singleNodeValue;
                            if (!select) return null;
                            return Array.from(select.options).map(opt => ({
                                text: opt.text,
                                value: opt.value,
                                index: opt.index
                            }));
                        }
                    """,
                        element.xpath,
                    )
                    return ToolResult(output=f"Dropdown options: {options}")

                elif action == "select_dropdown_option":
                    if index is None or not text:
                        return ToolResult(
                            error="Index and text are required for 'select_dropdown_option' action"
                        )
                    element = await context.get_dom_element_by_index(index)
                    if not element:
                        return ToolResult(error=f"Element with index {index} not found")
                    page = await context.get_current_page()
                    await page.select_option(element.xpath, label=text)
                    return ToolResult(
                        output=f"Selected option '{text}' from dropdown at index {index}"
                    )

                # Content extraction actions
                elif action == "extract_content":
                    if not goal:
                        return ToolResult(
                            error="Goal is required for 'extract_content' action"
                        )

                    # Use enhanced content extraction with JavaScript support
                    try:
                        page = await context.get_current_page()
                        
                        # Get current page URL for logging
                        page_url = page.url
                        logger.info(f"Using enhanced content extraction for: {page_url}")
                        
                        # Use enhanced content extraction
                        extraction_results = await self.enhanced_content_extractor.extract_content_with_js_support(page, goal)
                        
                        # Check if extraction was successful
                        text_content = extraction_results.get('text_content', '')
                        if not text_content or len(text_content.strip()) < 50:
                            # Try fallback strategies
                            logger.warning("Primary extraction failed, trying fallback strategies...")
                            
                            # Fallback 1: Wait longer and retry
                            await asyncio.sleep(5)
                            extraction_results = await self.enhanced_content_extractor.extract_content_with_js_support(page, goal)
                            text_content = extraction_results.get('text_content', '')
                            
                            # Fallback 2: Take screenshot for visual analysis
                            if not text_content or len(text_content.strip()) < 50:
                                try:
                                    screenshot = await page.screenshot()
                                    extraction_results['screenshot_taken'] = True
                                    extraction_results['fallback_reason'] = "Text extraction failed, screenshot captured for visual analysis"
                                except Exception as e:
                                    logger.warning(f"Screenshot fallback failed: {e}")
                        
                        # Analyze results with LLM
                        analysis_results = await self.enhanced_content_extractor.analyze_extraction_results(extraction_results, goal)
                        
                        # Format final output
                        if analysis_results.get('extraction_success'):
                            output = f"""✅ ENHANCED CONTENT EXTRACTION SUCCESSFUL

🎯 **Analysis Goal:** {goal}

📊 **Website Analysis:**
{analysis_results['analysis']}

📈 **Technical Summary:**
- Content Length: {analysis_results['raw_data']['text_length']} characters
- Headings Found: {analysis_results['raw_data']['structure_summary']['headings_count']}
- Links Found: {analysis_results['raw_data']['structure_summary']['links_count']}
- Images Found: {analysis_results['raw_data']['structure_summary']['images_count']}
- Framework: {analysis_results['raw_data']['technical_info'].get('framework', 'Unknown')}

📝 **Content Preview:**
{analysis_results['content_preview']}

🔗 **Source:** {page_url}
"""
                        else:
                            # Provide helpful error information
                            dynamic_info = extraction_results.get('dynamic_info', {})
                            error_details = f"""❌ CONTENT EXTRACTION CHALLENGES

🎯 **Attempted Goal:** {goal}
🔗 **URL:** {page_url}

🔍 **Technical Analysis:**
- Framework Detected: {self.enhanced_content_extractor._detect_framework(dynamic_info)}
- SPA Indicators: {dynamic_info.get('has_spa_indicators', False)}
- Scripts Found: {dynamic_info.get('script_count', 0)}

⚠️ **Possible Issues:**
- Website may require user interaction (login, CAPTCHA)
- Content may be behind authentication
- Heavy JavaScript rendering may need more time
- Anti-bot protection may be active

💡 **Recommendations:**
- Try accessing the website manually to verify availability
- Check if the site requires login or has geographic restrictions
- Consider alternative sources for this information

📊 **Extraction Attempt Results:**
{json.dumps(extraction_results, indent=2)}
"""
                            output = error_details
                        
                        return ToolResult(output=output)
                        
                    except Exception as e:
                        logger.error(f"Enhanced content extraction failed: {e}")
                        return ToolResult(
                            error=f"Enhanced content extraction failed: {str(e)}. "
                            f"This may indicate a technical issue with the website or network connectivity."
                        )

                # Tab management actions
                elif action == "switch_tab":
                    if tab_id is None:
                        return ToolResult(
                            error="Tab ID is required for 'switch_tab' action"
                        )
                    await context.switch_to_tab(tab_id)
                    page = await context.get_current_page()
                    await page.wait_for_load_state()
                    return ToolResult(output=f"Switched to tab {tab_id}")

                elif action == "open_tab":
                    if not url:
                        return ToolResult(error="URL is required for 'open_tab' action")
                    await context.create_new_tab(url)
                    return ToolResult(output=f"Opened new tab with {url}")

                elif action == "close_tab":
                    await context.close_current_tab()
                    return ToolResult(output="Closed current tab")

                # Utility actions
                elif action == "wait":
                    seconds_to_wait = seconds if seconds is not None else 3
                    await asyncio.sleep(seconds_to_wait)
                    return ToolResult(output=f"Waited for {seconds_to_wait} seconds")

                else:
                    return ToolResult(error=f"Unknown action: {action}")

            except Exception as e:
                error_msg = f"Browser action '{action}' failed: {str(e)}"
                
                # Add specific guidance for common errors
                error_str = str(e).lower()
                if "tool_calls" in error_str:
                    error_msg += "\nSuggestion: Try using a different approach to extract content, such as basic page content extraction."
                elif "navigation" in error_str or "timeout" in error_str:
                    error_msg += "\nSuggestion: Check if the URL is accessible and try again, or try a different URL."
                elif "element" in error_str and "not found" in error_str:
                    error_msg += "\nSuggestion: The page may have changed. Try refreshing the page or using a different element."
                elif "selector" in error_str:
                    error_msg += "\nSuggestion: Try using element indices instead of CSS selectors, or use a simpler selector."
                elif "connection" in error_str or "network" in error_str:
                    error_msg += "\nSuggestion: Check your internet connection and try again."
                else:
                    error_msg += "\nSuggestion: Try a different approach or check if the page has loaded completely."
                
                return ToolResult(error=error_msg)

    async def get_current_state(
        self, context: Optional[BrowserContext] = None
    ) -> ToolResult:
        """
        Get the current browser state as a ToolResult.
        If context is not provided, uses self.context.
        """
        try:
            # Use provided context or fall back to self.context
            ctx = context or self.context
            if not ctx:
                return ToolResult(error="Browser context not initialized")

            state = await ctx.get_state()

            # Create a viewport_info dictionary if it doesn't exist
            if hasattr(state, "viewport_info") and state.viewport_info:
                state.viewport_info.height
            elif hasattr(ctx, "config") and hasattr(ctx.config, "browser_window_size"):
                ctx.config.browser_window_size.get("height", 0)

            # Get the current page
            page = await ctx.get_current_page()

            # Get screenshot as base64
            screenshot = None
            try:
                screenshot_bytes = await page.screenshot(type="jpeg", quality=50)
                screenshot = base64.b64encode(screenshot_bytes).decode("utf-8")
            except Exception as e:
                logger.warning(f"Failed to take screenshot: {e}")

            # Return the state
            result = ToolResult(
                output=json.dumps(state),
                base64_image=screenshot,
            )

            return result
        except Exception as e:
            return ToolResult(error=f"Failed to get browser state: {str(e)}")

    async def cleanup(self):
        """Clean up browser resources."""
        try:
            if self.context:
                await self.context.close()
                self.context = None

            if self.browser:
                await self.browser.close()
                self.browser = None

            logger.info("Browser resources cleaned up")
        except Exception as e:
            logger.error(f"Error during browser cleanup: {e}")
