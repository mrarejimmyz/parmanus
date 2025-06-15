"""
Visual Google Search Implementation
Provides true visual browser interaction with Google search
"""

import asyncio
import json
import re
from typing import Dict, List, Optional, Tuple
from app.logger import logger
from app.schema import ToolCall, Function


class VisualGoogleSearch:
    """Handles visual Google search interactions like a human would"""
    
    def __init__(self, agent):
        self.agent = agent
        
    async def perform_visual_google_search(self, query: str) -> Dict:
        """
        Perform a complete visual Google search like a human would:
        1. Navigate to Google.com
        2. Visually find the search box
        3. Type the query
        4. Press Enter
        5. Analyze search results
        6. Return structured results
        """
        logger.info(f"🔍 Starting visual Google search for: {query}")
        
        try:
            # Step 1: Navigate to Google
            navigation_result = await self._navigate_to_google()
            if not navigation_result["success"]:
                return {"success": False, "error": "Failed to navigate to Google"}
            
            # Step 2: Analyze the Google page and find search box
            search_box_result = await self._find_and_use_search_box(query)
            if not search_box_result["success"]:
                return {"success": False, "error": "Failed to find or use search box"}
            
            # Step 3: Submit search and get results
            search_results = await self._get_search_results()
            if not search_results["success"]:
                return {"success": False, "error": "Failed to get search results"}
            
            # Step 4: Extract and structure the results
            structured_results = await self._structure_search_results(search_results["content"])
            
            return {
                "success": True,
                "query": query,
                "results": structured_results,
                "raw_content": search_results["content"]
            }
            
        except Exception as e:
            logger.error(f"❌ Visual Google search failed: {str(e)}")
            return {"success": False, "error": str(e)}
    
    async def _navigate_to_google(self) -> Dict:
        """Navigate to Google.com and verify we're there"""
        logger.info("🌐 Navigating to Google.com")
        
        try:
            # Navigate to Google
            tool_call = self._create_tool_call("browser_use", {
                "action": "go_to_url",
                "url": "https://www.google.com"
            })
            
            result = await self.agent.execute_tool(tool_call)
            
            if "error" in result.output.lower():
                return {"success": False, "error": result.output}
            
            # Verify we're on Google by extracting page content
            verification_result = await self._verify_google_page()
            
            return {
                "success": verification_result["is_google"],
                "content": verification_result["content"]
            }
            
        except Exception as e:
            return {"success": False, "error": str(e)}
    
    async def _verify_google_page(self) -> Dict:
        """Verify we're on Google by analyzing page content"""
        logger.info("👁️ Verifying we're on Google page")
        
        try:
            tool_call = self._create_tool_call("browser_use", {
                "action": "extract_content",
                "goal": "Check if this is Google search page and identify search input elements"
            })
            
            result = await self.agent.execute_tool(tool_call)
            content = result.output
            
            # Check for Google-specific elements
            is_google = any(indicator in content.lower() for indicator in [
                "google search", "google.com", "i'm feeling lucky", 
                "search the web", "google logo"
            ])
            
            return {
                "is_google": is_google,
                "content": content
            }
            
        except Exception as e:
            return {"is_google": False, "content": "", "error": str(e)}
    
    async def _find_and_use_search_box(self, query: str) -> Dict:
        """Find the search box and type the query"""
        logger.info(f"🔍 Looking for search box to type: {query}")
        
        # Try different element indices to find the search box
        # Google's search box is usually at index 0, 1, or 2
        for index in range(0, 10):  # Try first 10 elements
            logger.info(f"🎯 Trying to type in element #{index}")
            
            try:
                # Try to input text at this index
                tool_call = self._create_tool_call("browser_use", {
                    "action": "input_text",
                    "index": index,
                    "text": query
                })
                
                result = await self.agent.execute_tool(tool_call)
                
                # Check if input was successful
                if "error" not in result.output.lower() and "failed" not in result.output.lower():
                    logger.info(f"✅ Successfully typed query in element #{index}")
                    
                    # Now press Enter to submit the search
                    enter_result = await self._submit_search()
                    
                    if enter_result["success"]:
                        return {"success": True, "search_box_index": index}
                    else:
                        logger.warning(f"⚠️ Typed successfully but Enter failed for element #{index}")
                        continue
                else:
                    logger.debug(f"❌ Failed to type in element #{index}: {result.output}")
                    continue
                    
            except Exception as e:
                logger.debug(f"❌ Exception trying element #{index}: {str(e)}")
                continue
        
        return {"success": False, "error": "Could not find working search box"}
    
    async def _submit_search(self) -> Dict:
        """Submit the search by pressing Enter"""
        logger.info("⏎ Pressing Enter to submit search")
        
        try:
            tool_call = self._create_tool_call("browser_use", {
                "action": "send_keys",
                "keys": "Return"
            })
            
            result = await self.agent.execute_tool(tool_call)
            
            # Wait a moment for the search to process
            await asyncio.sleep(2)
            
            success = "error" not in result.output.lower()
            return {"success": success, "result": result.output}
            
        except Exception as e:
            return {"success": False, "error": str(e)}
    
    async def _get_search_results(self) -> Dict:
        """Extract search results from the current page"""
        logger.info("📊 Extracting search results from page")
        
        try:
            tool_call = self._create_tool_call("browser_use", {
                "action": "extract_content",
                "goal": "Extract all search results including titles, URLs, and descriptions"
            })
            
            result = await self.agent.execute_tool(tool_call)
            
            return {
                "success": True,
                "content": result.output
            }
            
        except Exception as e:
            return {"success": False, "error": str(e)}
    
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
            if url not in seen and not any(skip in url for skip in ['google.com', 'gstatic.com', 'googleusercontent.com']):
                unique_urls.append(url)
                seen.add(url)
        
        # Take top 10 results
        for i, url in enumerate(unique_urls[:10]):
            results.append({
                "rank": i + 1,
                "url": url,
                "title": f"Search Result {i + 1}",
                "description": "Found via visual Google search"
            })
        
        logger.info(f"📈 Structured {len(results)} search results")
        return results
    
    async def visit_search_result(self, url: str) -> Dict:
        """Visit a specific search result URL and extract content"""
        logger.info(f"🔗 Visiting search result: {url}")
        
        try:
            # Navigate to the URL
            tool_call = self._create_tool_call("browser_use", {
                "action": "go_to_url",
                "url": url
            })
            
            nav_result = await self.agent.execute_tool(tool_call)
            
            if "error" in nav_result.output.lower():
                return {"success": False, "error": nav_result.output}
            
            # Wait for page to load
            await asyncio.sleep(3)
            
            # Extract content from the page
            tool_call = self._create_tool_call("browser_use", {
                "action": "extract_content",
                "goal": f"Extract relevant information from this webpage about the search topic"
            })
            
            content_result = await self.agent.execute_tool(tool_call)
            
            return {
                "success": True,
                "url": url,
                "content": content_result.output
            }
            
        except Exception as e:
            return {"success": False, "error": str(e)}
    
    def _create_tool_call(self, function_name: str, arguments: Dict) -> ToolCall:
        """Create a properly formatted ToolCall object"""
        return ToolCall(
            id=f"call_{function_name}_{id(arguments)}",
            type="function",
            function=Function(name=function_name, arguments=json.dumps(arguments))
        )

