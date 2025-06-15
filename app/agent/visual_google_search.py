"""
Visual Google Search Implementation
Enhanced with numbered element annotation for precise Llama 3.2 Vision interaction
"""

import asyncio
import json
import re
from typing import Dict, List, Optional

from app.agent.visual_element_annotator import VisualElementAnnotator
from app.logger import logger
from app.schema import Function, ToolCall


class VisualGoogleSearch:
    """Handles visual Google search with element annotation for precise interaction"""

    def __init__(self, agent):
        self.agent = agent
        self.annotator = VisualElementAnnotator(agent)

    async def perform_visual_google_search(self, query: str) -> Dict:
        """Perform Google search using visual element annotation"""
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

    async def _navigate_to_google(self) -> Dict:
        """Navigate to Google.com with visual verification"""
        logger.info("🌐 Navigating to Google.com")
        max_retries = 3
        retry_count = 0

        while retry_count < max_retries:
            try:
                # Clear browser state first
                clear_call = self._create_tool_call(
                    "browser_use", {"action": "clear_state"}
                )
                await self.agent.execute_tool(clear_call)
                await asyncio.sleep(1)

                # Navigate to Google with explicit wait conditions
                nav_call = self._create_tool_call(
                    "browser_use",
                    {
                        "action": "go_to_url",
                        "url": "https://www.google.com",
                        "wait_for": {
                            "selector": "input[name='q']",  # Google search input
                            "timeout": 10000,  # 10 seconds
                        },
                    },
                )

                nav_result = await self.agent.execute_tool(nav_call)
                logger.info(f"📍 Navigation result: {nav_result}")

                # Check for navigation errors
                if hasattr(nav_result, "output"):
                    if "Error:" in str(nav_result.output):
                        logger.warning(
                            f"⚠️ Navigation attempt {retry_count + 1} failed: {nav_result.output}"
                        )
                        retry_count += 1
                        continue
                    else:
                        logger.info("✅ Navigation to Google.com appears successful")

                # Wait for page to be fully interactive
                await asyncio.sleep(2)

                # Do a basic interaction test
                test_call = self._create_tool_call(
                    "browser_use",
                    {"action": "check_element", "selector": "input[name='q']"},
                )
                test_result = await self.agent.execute_tool(test_call)

                if (
                    hasattr(test_result, "output")
                    and "success" in str(test_result.output).lower()
                ):
                    logger.info("✅ Search box interaction test passed")
                    return {"success": True}

                logger.warning("⚠️ Search box interaction test failed, retrying...")
                retry_count += 1

            except Exception as e:
                logger.error(
                    f"❌ Navigation attempt {retry_count + 1} failed: {str(e)}"
                )
                retry_count += 1
                await asyncio.sleep(2 * retry_count)  # Exponential backoff

        return {
            "success": False,
            "error": "Failed to establish reliable page interaction after retries",
        }

    async def _find_and_use_search_box_with_annotation(self, query: str) -> Dict:
        """Find search box using visual annotation and type query"""
        logger.info(
            f"🔍 Looking for search box with visual annotation to type: {query}"
        )

        try:
            # Try direct selector first (faster and more reliable)
            type_call = self._create_tool_call(
                "browser_use",
                {
                    "action": "type",
                    "selector": "input[name='q']",
                    "text": query,
                    "wait_for": {"selector": "input[name='q']", "timeout": 5000},
                },
            )

            type_result = await self.agent.execute_tool(type_call)

            # If direct selector works, submit the form
            if hasattr(type_result, "output") and "Error:" not in str(
                type_result.output
            ):
                submit_call = self._create_tool_call(
                    "browser_use", {"action": "press_key", "key": "Enter"}
                )
                await self.agent.execute_tool(submit_call)
                await asyncio.sleep(2)  # Wait for search results
                return {"success": True}

            # Fallback to visual annotation if direct selector fails
            search_box_result = await self.annotator.find_element_by_description(
                "Google search input box or search field where I can type a query"
            )

            if not search_box_result["success"]:
                logger.error(
                    f"❌ Failed to get annotated analysis: {search_box_result}"
                )
                return {
                    "success": False,
                    "error": "Failed to analyze page with annotations",
                }

            if not search_box_result["element_found"]:
                logger.error("❌ Search box not found in visual annotation")
                return {
                    "success": False,
                    "error": "Search box not found with visual annotation",
                }

            # Get the browser element index
            browser_index = search_box_result["browser_index"]
            logger.info(
                f"🎯 Found search box at element #{search_box_result['element_number']} (browser index: {browser_index})"
            )

            # Type in the search box
            type_call = self._create_tool_call(
                "browser_use",
                {"action": "input_text", "index": browser_index, "text": query},
            )

            type_result = await self.agent.execute_tool(type_call)
            logger.info(f"⌨️ Typing result: {type_result}")

            # Press Enter or click search button
            await asyncio.sleep(1)

            # Try to find and click search button with annotation
            search_button_result = await self.annotator.find_element_by_description(
                "Google Search button or submit button to execute the search"
            )

            if (
                search_button_result["success"]
                and search_button_result["element_found"]
            ):
                button_index = search_button_result["browser_index"]
                logger.info(
                    f"🔍 Found search button at element #{search_button_result['element_number']} (browser index: {button_index})"
                )

                click_call = self._create_tool_call(
                    "browser_use", {"action": "click_element", "index": button_index}
                )

                click_result = await self.agent.execute_tool(click_call)
                logger.info(f"🖱️ Search button click result: {click_result}")
            else:
                # Fallback: press Enter in search box
                logger.info("🔍 Search button not found, pressing Enter in search box")
                enter_call = self._create_tool_call(
                    "browser_use", {"action": "send_keys", "text": "\\n"}  # Enter key
                )

                enter_result = await self.agent.execute_tool(enter_call)
                logger.info(f"⏎ Enter key result: {enter_result}")

            # Wait for search results to load
            await asyncio.sleep(3)

            return {"success": True, "query": query}

        except Exception as e:
            logger.error(f"❌ Search box interaction failed: {str(e)}")
            return {"success": False, "error": str(e)}

    async def _extract_search_results(self) -> List[Dict]:
        """Extract search results from Google results page"""
        logger.info("📊 Extracting search results")

        try:
            # Extract page content
            extract_call = self._create_tool_call(
                "browser_use",
                {
                    "action": "extract_content",
                    "goal": "Extract Google search results with titles and URLs",
                },
            )

            extract_result = await self.agent.execute_tool(extract_call)

            # Parse results (simplified - would need more sophisticated parsing)
            results = []
            if hasattr(extract_result, "output"):
                content = str(extract_result.output)
                # Simple extraction - in practice would need better parsing
                if "search results" in content.lower():
                    results.append(
                        {
                            "title": "Search Results Found",
                            "url": "https://www.google.com/search",
                            "snippet": "Google search completed successfully",
                        }
                    )

            logger.info(f"📊 Extracted {len(results)} search results")
            return results

        except Exception as e:
            logger.error(f"❌ Results extraction failed: {str(e)}")
            return []

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
