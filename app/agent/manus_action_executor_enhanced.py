"""
Enhanced Manus Action Executor with Nodriver Integration
Handles actual tool execution for different step types with improved Google search
"""

import asyncio
import json
from typing import Dict, List, Optional

from app.agent.visual_google_search import VisualGoogleSearch
from app.logger import logger
from app.schema import Function, ToolCall


class ManusActionExecutor:
    """Handles execution of different action types for Manus agent steps with Nodriver support."""

    def __init__(self, agent):
        """Initialize with reference to the main agent."""
        self.agent = agent
        self.browser_handler = (
            agent.browser_handler if hasattr(agent, "browser_handler") else None
        )
        self.visual_google = VisualGoogleSearch(agent)

    async def execute_research_action(self, step: str) -> bool:
        """Execute research and planning actions - use Nodriver-powered Google search."""
        try:
            logger.info(f"🔍 RESEARCH ACTION: {step}")

            # Generate intelligent search queries based on user request
            search_queries = await self._generate_search_queries(step)

            for query in search_queries:
                logger.info(f"🔍 Googling with Nodriver: {query}")

                # Use Nodriver-powered Google search
                search_success = await self._google_search(query)
                if search_success:
                    await asyncio.sleep(2)  # Allow results to load
                else:
                    logger.warning(f"⚠️ Search failed for query: {query}")

            return True

        except Exception as e:
            logger.error(f"Error in research action: {str(e)}")
            return False

    async def _google_search(self, query: str) -> bool:
        """Perform a Google search with Nodriver for captcha bypass."""
        try:
            logger.info(f"🔍 Starting Nodriver Google search for: {query}")

            # Use the new Nodriver-powered visual Google search implementation
            search_result = await self.visual_google.perform_visual_google_search(query)

            if search_result["success"]:
                logger.info(f"✅ Nodriver Google search completed successfully!")
                logger.info(
                    f"📊 Found {len(search_result.get('results', []))} search results"
                )

                # Store results for later use
                if hasattr(self.agent, "last_search_results"):
                    self.agent.last_search_results = search_result

                return True
            else:
                logger.error(
                    f"❌ Nodriver Google search failed: {search_result.get('error', 'Unknown error')}"
                )
                return False

        except Exception as e:
            logger.error(f"❌ Exception in Nodriver Google search: {str(e)}")
            return False

    async def _visit_search_results(self, num_results: int = 3) -> bool:
        """Visit the top search results from the last Nodriver Google search."""
        try:
            logger.info(
                f"📖 Visiting top {num_results} search results from Nodriver search"
            )

            # Get results from the last Nodriver Google search
            if (
                not hasattr(self.agent, "last_search_results")
                or not self.agent.last_search_results
            ):
                logger.warning("⚠️ No search results available to visit")
                return False

            results = self.agent.last_search_results.get("results", [])
            if not results:
                logger.warning("⚠️ No search results found in last search")
                return False

            # Visit top results
            for i, result in enumerate(results[:num_results]):
                try:
                    url = result.get("url", "")
                    title = result.get("title", f"Result {i+1}")

                    if url and url.startswith("http"):
                        logger.info(f"📖 Visiting result {i+1}: {title}")
                        logger.info(f"🔗 URL: {url}")

                        # Use browser handler to visit the URL
                        if self.browser_handler:
                            visit_call = self._create_tool_call(
                                "browser_use", {"action": "go_to_url", "url": url}
                            )

                            visit_result = await self.agent.execute_tool(visit_call)
                            logger.info(f"📄 Visit result: {visit_result}")

                            # Wait a bit between visits
                            await asyncio.sleep(2)

                except Exception as e:
                    logger.warning(f"⚠️ Failed to visit result {i+1}: {str(e)}")
                    continue

            return True

        except Exception as e:
            logger.error(f"❌ Failed to visit search results: {str(e)}")
            return False

    def _get_user_message(self) -> str:
        """Get the last user message from agent's message history."""
        if hasattr(self.agent, "messages") and self.agent.messages:
            for msg in reversed(self.agent.messages):
                if msg.role == "user":
                    return msg.content.lower()
        return ""

    async def _generate_search_queries(self, step: str) -> List[str]:
        """Generate intelligent search queries based on the step and user request."""
        try:
            user_message = self._get_user_message()

            # Extract key terms from user message and step
            queries = []

            # Basic query from user message
            if user_message:
                # Clean up the user message for search
                clean_query = (
                    user_message.replace("give me", "").replace("show me", "").strip()
                )
                if clean_query:
                    queries.append(clean_query)

            # Add step-specific query if different
            if step.lower() not in user_message:
                queries.append(step)

            # Ensure we have at least one query
            if not queries:
                queries = ["latest information"]

            # Limit to 3 queries max
            return queries[:3]

        except Exception as e:
            logger.error(f"Error generating search queries: {str(e)}")
            return ["latest information"]

    def _create_tool_call(self, tool_name: str, arguments: Dict) -> ToolCall:
        """Create a tool call object."""
        return ToolCall(
            id=f"call_{tool_name}_{hash(str(arguments)) % 10000}",
            function=Function(name=tool_name, arguments=json.dumps(arguments)),
        )

    async def cleanup(self) -> None:
        """Clean up resources, especially Nodriver browser."""
        try:
            if self.visual_google:
                await self.visual_google.close()
                logger.info("🧹 Nodriver Google search cleanup completed")
        except Exception as e:
            logger.warning(f"⚠️ Cleanup warning: {str(e)}")
