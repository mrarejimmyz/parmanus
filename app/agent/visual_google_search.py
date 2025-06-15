"""
Visual Google Search Implementation
Enhanced with vision-enabled browser interaction
"""

import asyncio
import json
import re
import time
from typing import Dict, List, Optional
from uuid import uuid4

from app.agent.visual_element_annotator import VisualElementAnnotator
from app.logger import logger
from app.schema import Function, ToolCall
from app.tool.browser_use_tool import BrowserUseTool


class VisualGoogleSearch:
    """Handles visual Google search with vision-enabled browser interaction"""

    def __init__(self, agent):
        self.agent = agent
        self.annotator = VisualElementAnnotator(agent)

    async def perform_visual_google_search(self, query: str) -> Dict:
        """Perform Google search using vision-enabled browser interaction"""
        logger.info(f"🔍 Starting visual Google search with annotation for: {query}")

        try:
            # Step 1: Navigate to Google
            navigation_result = await self._navigate_to_google()
            if not navigation_result["success"]:
                return navigation_result

            # Step 2: Find and use search box with visual annotation
            search_result = await self._find_and_use_search_box_with_annotation(query)
            if not search_result["success"]:
                return search_result

            # Step 3: Extract search results
            results = await self._extract_search_results()

            return {"success": True, "query": query, "results": results}

        except Exception as e:
            logger.error(f"❌ Visual Google search failed: {str(e)}")
            return {"success": False, "error": str(e)}

    def _create_tool_call(self, name: str, args: Dict) -> ToolCall:
        """Create a properly formatted tool call"""
        return ToolCall(
            id=f"call_{name}_{uuid4().hex[:8]}",
            type="function",
            function=Function(name=name, arguments=json.dumps(args)),
        )

    async def _navigate_to_google(self) -> Dict:
        """Navigate to Google.com with visual verification"""
        logger.info("🌐 Navigating to Google.com")
        max_retries = 3
        retry_count = 0

        while retry_count < max_retries:
            try:
                # Create navigation tool call
                nav_call = self._create_tool_call(
                    "browser_navigate", {"url": "https://www.google.com"}
                )

                # Execute navigation
                nav_result = await self.agent.execute_tool_call(nav_call)

                # Verify page load with vision
                verify_call = self._create_tool_call(
                    "browser_vision_analyze",
                    {
                        "prompt": "Verify this is the Google search homepage with search box visible",
                        "element_types": ["search input", "Google logo"],
                    },
                )
                verify_result = await self.agent.execute_tool_call(verify_call)

                if verify_result.get("is_verified"):
                    logger.info("✅ Successfully navigated to Google.com")
                    return {"success": True}

                retry_count += 1
                logger.warning(
                    f"⚠️ Page verification failed, attempt {retry_count} of {max_retries}"
                )
                await asyncio.sleep(1)  # Brief delay before retry

            except Exception as e:
                logger.error(f"❌ Navigation failed: {str(e)}")
                retry_count += 1
                if retry_count >= max_retries:
                    return {
                        "success": False,
                        "error": f"Failed to navigate after {max_retries} attempts",
                    }
                await asyncio.sleep(1)

        return {"success": False, "error": "Failed to verify Google homepage"}

    async def _find_and_use_search_box_with_annotation(self, query: str) -> Dict:
        """Use vision to find and interact with search box"""
        logger.info("🔍 Finding and using search box")
        max_retries = 3
        retry_count = 0

        while retry_count < max_retries:
            try:
                # Use vision to locate search box
                locate_call = self._create_tool_call(
                    "browser_vision_analyze",
                    {
                        "prompt": "Locate the main Google search input box",
                        "element_types": ["search input"],
                    },
                )
                locate_result = await self.agent.execute_tool_call(locate_call)

                if locate_result.get("elements"):
                    # Input the search query
                    input_call = self._create_tool_call(
                        "browser_element_action",
                        {
                            "action": "input",
                            "value": query,
                            "selector": locate_result["elements"][0].get(
                                "selector", "input[name='q']"
                            ),
                        },
                    )
                    await self.agent.execute_tool_call(input_call)

                    # Submit the search
                    submit_call = self._create_tool_call(
                        "browser_element_action",
                        {
                            "action": "press",
                            "key": "Enter",
                            "selector": locate_result["elements"][0].get(
                                "selector", "input[name='q']"
                            ),
                        },
                    )
                    await self.agent.execute_tool_call(submit_call)

                    # Verify search results page loaded
                    await asyncio.sleep(2)  # Brief delay for page load
                    verify_call = self._create_tool_call(
                        "browser_vision_analyze",
                        {
                            "prompt": "Verify this is a Google search results page",
                            "element_types": ["search results", "result links"],
                        },
                    )
                    verify_result = await self.agent.execute_tool_call(verify_call)

                    if verify_result.get("is_verified"):
                        logger.info("✅ Successfully executed search")
                        return {"success": True}

                retry_count += 1
                logger.warning(
                    f"⚠️ Search execution failed, attempt {retry_count} of {max_retries}"
                )
                await asyncio.sleep(1)

            except Exception as e:
                logger.error(f"❌ Search box interaction failed: {str(e)}")
                retry_count += 1
                if retry_count >= max_retries:
                    return {
                        "success": False,
                        "error": f"Failed to execute search after {max_retries} attempts",
                    }
                await asyncio.sleep(1)

        return {"success": False, "error": "Failed to execute search"}

    async def _extract_search_results(self) -> List[Dict]:
        """Extract search results using vision analysis"""
        logger.info("📑 Extracting search results with visual analysis")

        try:
            # Take screenshot and analyze search results
            vision_call = self._create_tool_call(
                "browser_vision_analyze",
                {
                    "prompt": "Analyze Google search results. Find main result titles, URLs, and descriptions.",
                    "element_types": ["search result", "link", "text snippet"],
                    "screenshot_area": "main",
                },
            )

            vision_result = await self.agent.execute_tool_call(vision_call)

            if not vision_result.get("elements"):
                logger.warning("⚠️ No search results found in visual analysis")
                return []

            results = []
            for element in vision_result.get("elements", []):
                if element.get("type") == "search result":
                    result = {
                        "title": element.get("text", "").strip(),
                        "url": element.get("url", ""),
                        "snippet": element.get("description", "").strip(),
                    }
                    if result["title"] and (result["url"] or result["snippet"]):
                        results.append(result)

            logger.info(f"✅ Successfully extracted {len(results)} search results")
            return results

        except Exception as e:
            logger.error(f"❌ Failed to extract search results: {str(e)}")
            return []

    async def close(self) -> None:
        """Cleanup browser resources"""
        logger.info("🧹 Cleaning up browser resources")
        try:
            close_call = self._create_tool_call("browser_close", {})
            await self.agent.execute_tool_call(close_call)
        except Exception as e:
            logger.error(f"❌ Failed to close browser: {str(e)}")

    def _create_tool_call(self, function_name: str, arguments: Dict) -> ToolCall:
        """Create a properly formatted ToolCall object"""
        return ToolCall(
            id=f"call_{function_name}_{id(arguments)}",
            type="function",
            function=Function(name=function_name, arguments=json.dumps(arguments)),
        )

    async def _structure_search_results(self, raw_content: str) -> List[Dict]:
        """Structure the raw search results into a clean format"""
        logger.info("🔧 Structuring search results")

        results = []

        # Extract URLs from the content
        url_pattern = r'https?://[^\s<>"\']+[^\s<>"\'.,)]'
        urls = re.findall(url_pattern, raw_content)

        # Remove duplicates while preserving order
        unique_urls = []
        seen = set()
        for url in urls:
            if url not in seen and not any(
                skip in url
                for skip in ["google.com", "gstatic.com", "googleusercontent.com"]
            ):
                unique_urls.append(url)
                seen.add(url)

        # Take top 10 results
        for i, url in enumerate(unique_urls[:10]):
            results.append(
                {
                    "rank": i + 1,
                    "url": url,
                    "title": f"Search Result {i + 1}",
                    "description": "Found via visual Google search",
                }
            )

        logger.info(f"📈 Structured {len(results)} search results")
        return results

    async def visit_search_result(self, url: str) -> Dict:
        """Visit a specific search result URL and extract content"""
        logger.info(f"🔗 Visiting search result: {url}")

        try:
            # Navigate to the URL
            tool_call = self._create_tool_call(
                "browser_use", {"action": "go_to_url", "url": url}
            )

            nav_result = await self.agent.execute_tool(tool_call)

            if "error" in nav_result.output.lower():
                return {"success": False, "error": nav_result.output}

            # Wait for page to load
            await asyncio.sleep(3)

            # Extract content from the page
            tool_call = self._create_tool_call(
                "browser_use",
                {
                    "action": "extract_content",
                    "goal": f"Extract relevant information from this webpage about the search topic",
                },
            )

            content_result = await self.agent.execute_tool(tool_call)

            return {"success": True, "url": url, "content": content_result.output}

        except Exception as e:
            return {"success": False, "error": str(e)}

    def _create_tool_call(self, function_name: str, arguments: Dict) -> ToolCall:
        """Create a properly formatted ToolCall object"""
        return ToolCall(
            id=f"call_{function_name}_{id(arguments)}",
            type="function",
            function=Function(name=function_name, arguments=json.dumps(arguments)),
        )
