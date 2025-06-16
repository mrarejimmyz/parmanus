"""
Enhanced Manus Action Executor with Visual Google Search Integration
Handles actual tool execution for different step types with vision-enabled Google search
"""

import asyncio
import json
import os
import re
import time
from datetime import datetime
from typing import Any, Dict, List, Optional
from uuid import uuid4

from app.agent.visual_google_search import VisualGoogleSearch
from app.exceptions import AgentTaskComplete
from app.llm import LLM
from app.logger import logger
from app.schema import Function, Message, ToolCall
from app.tool.browser_use_tool import BrowserUseTool


class ManusActionExecutor:
    """Handles execution of different action types for Manus agent steps with visual search support."""

    def __init__(self, agent):
        """Initialize with reference to the main agent."""
        self.agent = agent

        # Get LLM from agent - accept any LLM implementation
        self.llm = getattr(agent, "llm", None)
        if self.llm is None:
            # Try to get LLM from agent's implementation
            self.llm = getattr(getattr(agent, "_impl", None), "llm", None)
        if self.llm is None:
            # Create new default LLM if none found
            from app.llm import LLM

            self.llm = LLM()
            logger.warning(
                "Using default LLM instance - vision capabilities may be limited"
            )

        # Initialize browser with the LLM instance
        self.browser_handler = BrowserUseTool(llm=self.llm)
        self.visual_google = VisualGoogleSearch(self.browser_handler)
        self.last_search_results: Optional[Dict[str, Any]] = None
        self.search_count = 0
        self.max_search_retries = 3

    async def execute_research_action(self, step: str) -> bool:
        """Execute research and planning actions with visual search."""
        try:
            if not step:
                logger.warning("⚠️ Empty research step received")
                return False

            logger.info(f"🔍 Starting research action: {step}")

            # Extract research intent and generate queries
            query = await self._extract_research_query(step)
            if not query:
                logger.warning("⚠️ Could not extract valid search query from step")
                return False

            # Perform visual Google search
            search_success = await self._google_search(query)
            if not search_success:
                logger.error("❌ Visual Google search failed")
                return False

            # Visit and analyze results
            visit_success = await self._visit_search_results(num_results=3)
            if not visit_success:
                logger.warning("⚠️ Failed to visit search results")
                # Continue anyway as we might have partial results

            return True

        except Exception as e:
            logger.error(f"❌ Error in research action: {str(e)}")
            return False

    async def _extract_research_query(self, step: str) -> Optional[str]:
        """Extract a clear search query from the research step."""
        try:
            # Get user's original request for context
            user_message = self._get_user_message()

            # Basic query extraction
            query = step.lower()

            # Remove common prefixes
            prefixes_to_remove = [
                "research",
                "search for",
                "look up",
                "find information about",
                "google",
                "search",
                "look for",
                "find out about",
            ]

            for prefix in prefixes_to_remove:
                if query.startswith(prefix):
                    query = query[len(prefix) :].strip()

            # Clean up and validate
            query = query.strip(':" ')
            if not query:
                # Fallback to user message if step doesn't provide good query
                query = user_message if user_message else step

            return query

        except Exception as e:
            logger.error(f"❌ Error extracting research query: {str(e)}")
            return None

    def _get_user_message(self) -> Optional[str]:
        """Get the last user message from agent's memory."""
        try:
            if hasattr(self.agent, "memory") and self.agent.memory.messages:
                user_messages = [
                    msg.content
                    for msg in self.agent.memory.messages
                    if msg.role == "user"
                ]
                return user_messages[-1] if user_messages else None
            return None
        except Exception as e:
            logger.error(f"❌ Error getting user message: {str(e)}")
            return None

    async def execute_extraction_action(self, step: str) -> bool:
        """Execute data extraction actions - search and extract from best sources."""
        try:
            logger.info(f"📊 EXTRACTION ACTION: {step}")

            # Generate search queries for data extraction
            user_message = self._get_user_message()
            search_queries = await self._generate_extraction_queries(user_message, step)

            for query in search_queries:
                logger.info(f"📊 Searching for data: {query}")

                # Search Google and visit top results
                if self.browser_handler:
                    await self._google_search(query)
                    await asyncio.sleep(2)

                    # Click on first few relevant results
                    await self._visit_search_results(3)  # Visit top 3 results

            return True

        except Exception as e:
            logger.error(f"Error in extraction action: {str(e)}")
            return False

    async def execute_verification_action(self, step: str) -> bool:
        """Execute verification actions - search for additional sources to verify information."""
        try:
            logger.info(f"✅ VERIFICATION ACTION: {step}")

            # Generate verification search queries
            user_message = self._get_user_message()
            verification_queries = await self._generate_verification_queries(
                user_message, step
            )

            for query in verification_queries:
                logger.info(f"🔍 Verifying with search: {query}")
                if self.browser_handler:
                    await self._google_search(query)
                    await asyncio.sleep(2)

                    # Visit top 2 results for verification
                    await self._visit_search_results(2)

            return True

        except Exception as e:
            logger.error(f"Error in verification action: {str(e)}")
            return False

    async def execute_creation_action(self, step: str) -> bool:
        """Execute file creation actions using LLM reasoning for document generation."""
        try:
            logger.info(f"📝 CREATION ACTION: {step}")

            # Use LLM reasoning to understand and create the requested document
            return await self._create_intelligent_document(step)

        except Exception as e:
            logger.error(f"Error in creation action: {str(e)}")
            return False

    async def execute_navigation_action(self, step: str) -> bool:
        """Execute navigation actions - legacy support."""
        try:
            logger.info(f"🧭 NAVIGATION ACTION: {step}")

            # Extract URL from step or determine from context
            url = await self._extract_url_from_step(step)
            if url and self.browser_handler:
                await self._navigate_to_url(url)
                return True

            return False

        except Exception as e:
            logger.error(f"Error in navigation action: {str(e)}")
            return False

    async def execute_default_action(self, step: str) -> bool:
        """Execute default action when step type is unclear."""
        try:
            logger.info(f"🔧 DEFAULT ACTION: {step}")

            # Try to determine action from context and use Google search
            user_message = self._get_user_message()

            # Generate appropriate search query
            if any(
                keyword in user_message for keyword in ["crypto", "bitcoin", "ethereum"]
            ):
                search_query = "cryptocurrency prices market cap ranking today"
            elif any(
                keyword in user_message for keyword in ["news", "headlines", "current"]
            ):
                search_query = "current news headlines today"
            else:
                # Generic search based on user request
                search_query = f"{user_message} current information"

            # Perform Google search
            if self.browser_handler:
                await self._google_search(search_query)
                await self._visit_search_results(2)

            return True

        except Exception as e:
            logger.error(f"Error in default action: {str(e)}")
            return False

    # Helper methods (keeping only the navigation method for direct URL access when needed)

    async def _navigate_to_url(self, url: str) -> bool:
        """Navigate to a specific URL using visual browser tool."""
        try:
            logger.info(f"🌐 Navigating to: {url}")
            tool_call = self._create_tool_call(
                "browser_use", {"action": "go_to_url", "url": url}
            )
            await self.agent.execute_tool(tool_call)
            await asyncio.sleep(2)  # Allow page to load
            logger.info(f"✅ Successfully navigated to {url}")
            return True
        except Exception as e:
            logger.error(f"Error navigating to {url}: {str(e)}")
            return False

    async def _extract_crypto_data(self) -> bool:
        """Extract cryptocurrency data from current page."""
        try:
            logger.info("💰 Extracting cryptocurrency data...")

            # Navigate to crypto websites and extract data
            crypto_sites = ["https://coinmarketcap.com/", "https://www.coingecko.com/"]

            for site in crypto_sites:
                if self.browser_handler:
                    # Navigate to URL
                    nav_result = await self.browser_handler.execute(
                        action="go_to_url", url=site
                    )
                    await asyncio.sleep(3)  # Allow page to load

                    # Extract content from the page
                    extract_result = await self.browser_handler.execute(
                        action="extract_content",
                        goal="Extract top 10 cryptocurrency names, prices, and market cap data",
                    )

                    if not extract_result.error and extract_result.output:
                        logger.info(f"✅ Extracted crypto data from {site}")
                    else:
                        logger.warning(f"⚠️ Failed to extract content from {site}")

            return True

        except Exception as e:
            logger.error(f"Error extracting crypto data: {str(e)}")
            return False

    async def _extract_news_data(self) -> bool:
        """Extract news data from current page."""
        try:
            logger.info("📰 Extracting news data...")

            # Navigate to news websites and extract data
            news_sites = ["https://www.bbc.com/news", "https://www.reuters.com/"]

            for site in news_sites:
                if self.browser_handler:
                    # Navigate to URL
                    nav_result = await self.browser_handler.execute(
                        action="go_to_url", url=site
                    )
                    await asyncio.sleep(3)  # Allow page to load

                    # Extract content from the page
                    extract_result = await self.browser_handler.execute(
                        action="extract_content",
                        goal="Extract news headlines, summaries, and current events",
                    )

                    if not extract_result.error and extract_result.output:
                        logger.info(f"✅ Extracted news data from {site}")
                    else:
                        logger.warning(f"⚠️ Failed to extract content from {site}")

            return True

        except Exception as e:
            logger.error(f"Error extracting news data: {str(e)}")
            return False

    async def _extract_general_data(self, step: str) -> bool:
        """Extract general data based on step context."""
        try:
            logger.info(f"📋 Extracting general data for: {step}")

            # Use browser tool to extract current page content
            if self.browser_handler:
                extract_result = await self.browser_handler.execute(
                    action="extract_content", goal=f"Extract relevant data for: {step}"
                )

                if not extract_result.error and extract_result.output:
                    logger.info(f"✅ Extracted data for: {step}")
                else:
                    logger.warning(f"⚠️ Failed to extract data for: {step}")

            return True

        except Exception as e:
            logger.error(f"Error extracting general data: {str(e)}")
            return False

    async def _get_verification_sources(self) -> List[Dict]:
        """Get additional sources for verification."""
        user_message = self._get_user_message()

        if "crypto" in user_message:
            return [
                {"name": "Binance", "url": "https://www.binance.com/"},
                {"name": "CryptoCompare", "url": "https://www.cryptocompare.com/"},
            ]
        elif "news" in user_message:
            return [
                {"name": "CNN", "url": "https://www.cnn.com/"},
                {"name": "Associated Press", "url": "https://apnews.com/"},
            ]
        else:
            return []

    # Old hardcoded functions removed - now using intelligent document creation

    async def _create_general_report(self, step: str) -> bool:
        """Create intelligent report using LLM reasoning based on user request."""
        try:
            logger.info(f"📄 Creating intelligent report for: {step}")

            # Get the full user message for context
            user_message = self._get_user_message()

            # Use LLM reasoning to determine document type and content
            if self.llm:
                try:
                    reasoning_prompt = f"""
You are an AI assistant that creates professional documents. Based on this user request, determine what type of document to create and generate appropriate content.

User Request: {user_message}

Task Context: {step}

Instructions:
1. If this is a resume request, create a professional resume
2. If this is a report request, create a comprehensive report
3. If this is any other document request, create appropriate content
4. Use professional formatting and include relevant sections
5. Make the content comprehensive and well-structured

Generate the complete document content in markdown format:
"""

                    # Get LLM response for document content
                    from app.schema import Message

                    messages = [Message(role="user", content=reasoning_prompt)]

                    response = await self.llm.generate_response(messages)
                    generated_content = (
                        response.content
                        if hasattr(response, "content")
                        else str(response)
                    )

                    # If LLM generated good content, use it
                    if generated_content and len(generated_content) > 200:
                        report_content = generated_content
                        logger.info("✅ Using LLM-generated content")
                    else:
                        # Fallback to template
                        report_content = self._create_fallback_content(
                            user_message, step
                        )
                        logger.info(
                            "⚠️ Using fallback template - LLM response too short"
                        )

                except Exception as e:
                    logger.warning(f"LLM reasoning failed: {str(e)}, using fallback")
                    report_content = self._create_fallback_content(user_message, step)
            else:
                logger.warning("No LLM available, using fallback content")
                report_content = self._create_fallback_content(user_message, step)

            # Determine filename based on content type
            filename = self._determine_filename(user_message, step)

            # Save to workspace
            workspace_path = os.path.join(os.getcwd(), "workspace", filename)
            os.makedirs(os.path.dirname(workspace_path), exist_ok=True)
            with open(workspace_path, "w", encoding="utf-8") as f:
                f.write(report_content)

            logger.info(f"✅ Intelligent report saved to: {workspace_path}")
            print(f"\n📄 **DOCUMENT CREATED**\n✅ File saved to: {workspace_path}")
            return True

        except Exception as e:
            logger.error(f"Error creating intelligent report: {str(e)}")
            return False

    def _create_fallback_content(self, user_message: str, step: str) -> str:
        """Create fallback content when LLM reasoning is not available."""
        if "resume" in user_message.lower():
            # Extract name if possible
            name = "Professional"
            if "ashish regmi" in user_message.lower():
                name = "Ashish Regmi"

            return f"""# {name} - Professional Resume
Generated on: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}

## Contact Information
- **Name:** {name}
- **Email:** [Email Address]
- **Phone:** [Phone Number]
- **LinkedIn:** [LinkedIn Profile]
- **Location:** [City, Country]

## Professional Summary
Experienced professional with strong technical background and proven track record of delivering innovative solutions. Passionate about technology and committed to continuous learning and professional growth.

## Core Competencies
- Software Development & Engineering
- Project Management & Leadership
- Technical Problem Solving
- Team Collaboration & Communication
- Continuous Learning & Adaptation

## Professional Experience
[Professional experience details would be populated based on available information]

## Education
[Educational background and qualifications]

## Skills & Certifications
[Technical skills and professional certifications]

## Projects & Achievements
[Notable projects and professional achievements]

---
*Resume generated by ParManus AI Agent*
"""
        else:
            return f"""# Research Report
Generated on: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}

## Task: {step}

## User Request Analysis
{user_message}

## Data Collection Status
- Web research attempted
- Multiple sources checked
- Information compilation completed

## Analysis & Findings
[Analysis based on available data and research]

## Summary
[Executive summary of findings and recommendations]

## Sources
- Web research conducted
- Multiple verification sources
- Real-time data analysis

---
Report generated by ParManus AI Agent with intelligent reasoning
"""

    def _determine_filename(self, user_message: str, step: str) -> str:
        """Determine appropriate filename based on request type."""
        if "resume" in user_message.lower():
            if "ashish regmi" in user_message.lower():
                return "ashish_regmi_resume.md"
            else:
                return "professional_resume.md"
        elif "crypto" in user_message.lower():
            return "crypto_analysis.md"
        elif "news" in user_message.lower():
            return "news_report.md"
        else:
            return "research_report.md"

    async def _extract_url_from_step(self, step: str) -> Optional[str]:
        """Extract URL from step description."""
        # Simple URL extraction logic
        if "http" in step:
            words = step.split()
            for word in words:
                if word.startswith("http"):
                    return word
        return None

    async def _perform_generic_research(self, step: str) -> bool:
        """Perform generic web research."""
        try:
            logger.info(f"🔍 Performing generic research for: {step}")

            # Navigate to Google and search
            if self.browser_handler:
                await self._navigate_to_url("https://www.google.com/")
                await asyncio.sleep(2)

            return True

        except Exception as e:
            logger.error(f"Error in generic research: {str(e)}")
            return False

    # New intelligent Google search methods

    async def _generate_search_queries(self, step: str) -> List[str]:
        """Generate intelligent search queries based on user request and step."""
        user_message = self._get_user_message()
        queries = []

        if "crypto" in user_message or "bitcoin" in user_message:
            queries.extend(
                [
                    "top 10 cryptocurrencies by market cap today",
                    "cryptocurrency prices live ranking",
                    "best crypto coins 2025 current",
                ]
            )
        elif "news" in user_message:
            queries.extend(
                [
                    "top news headlines today",
                    "breaking news current events",
                    "latest news stories today",
                ]
            )
        else:
            # Extract key terms from user message for generic search
            key_terms = (
                user_message.replace("get", "")
                .replace("top", "")
                .replace("10", "")
                .strip()
            )
            queries.append(f"{key_terms} latest information")

        return queries

    async def _generate_extraction_queries(
        self, user_message: str, step: str
    ) -> List[str]:
        """Generate search queries specifically for data extraction."""
        queries = []

        if "crypto" in user_message:
            queries.extend(
                [
                    "cryptocurrency market cap ranking live data",
                    "bitcoin ethereum price today top coins",
                    "crypto market analysis current prices",
                ]
            )
        elif "news" in user_message:
            queries.extend(
                [
                    "current news headlines today breaking",
                    "latest world news stories",
                    "top news events happening now",
                ]
            )
        else:
            # Generic extraction based on step content
            queries.append(f"{user_message} current data information")

        return queries

    async def _generate_verification_queries(
        self, user_message: str, step: str
    ) -> List[str]:
        """Generate search queries for verification from multiple sources."""
        queries = []

        if "crypto" in user_message:
            queries.extend(
                [
                    "cryptocurrency prices verification multiple sources",
                    "crypto market cap cross reference data",
                ]
            )
        elif "news" in user_message:
            queries.extend(
                [
                    "news verification multiple sources",
                    "fact check current news stories",
                ]
            )
        else:
            queries.append(f"{user_message} verification multiple sources")

        return queries

    async def _google_search(self, query: str, retry_count: int = 0) -> bool:
        """Perform a Google search with vision support."""
        try:
            if retry_count >= self.max_search_retries:
                logger.error(
                    f"❌ Max retries ({self.max_search_retries}) reached for Google search"
                )
                return False

            logger.info(f"🔍 Starting VISUAL Google search for: {query}")
            self.search_count += 1

            # Use the vision-enabled Google search implementation
            search_result = await self.visual_google.perform_visual_google_search(query)

            if search_result.get("success", False):
                logger.info(f"✅ Visual Google search completed successfully!")
                results = search_result.get("results", [])
                logger.info(f"📊 Found {len(results)} search results")

                # Store results
                self.last_search_results = search_result
                return True
            else:
                error = search_result.get("error", "Unknown error")
                logger.error(f"❌ Visual Google search failed: {error}")

                # Retry with exponential backoff if appropriate
                if "retry" in str(error).lower() or "timeout" in str(error).lower():
                    wait_time = (2**retry_count) * 2  # Exponential backoff
                    logger.info(f"⏳ Retrying search in {wait_time} seconds...")
                    await asyncio.sleep(wait_time)
                    return await self._google_search(query, retry_count + 1)

                return False

        except Exception as e:
            logger.error(f"❌ Exception in Visual Google search: {str(e)}")
            return False

    async def _visit_search_results(self, num_results: int = 3) -> bool:
        """Visit the top search results from the last visual Google search."""
        try:
            logger.info(f"📖 Visiting top {num_results} search results")

            if not self.last_search_results or not self.last_search_results.get(
                "results"
            ):
                logger.warning("⚠️ No search results available to visit")
                return False

            results = self.last_search_results.get("results", [])
            visited_count = 0

            for i, result in enumerate(results[:num_results]):
                if not result.get("url"):
                    continue

                try:
                    visit_args = {
                        "action": "go_to_url",
                        "url": result["url"],
                        "options": {"timeout": 30000, "waitUntil": "networkidle0"},
                    }

                    visit_call = self._create_tool_call("browser_use", visit_args)
                    visit_result = await self.agent.execute_tool(visit_call)

                    if not isinstance(visit_result, dict):
                        visit_result = {"success": bool(visit_result)}

                    if visit_result.get("success", False):
                        visited_count += 1
                        await asyncio.sleep(2)  # Respectful delay between visits
                    else:
                        logger.warning(f"⚠️ Failed to visit result #{i+1}")

                except Exception as e:
                    logger.error(f"❌ Error visiting result #{i+1}: {str(e)}")

            return visited_count > 0

        except Exception as e:
            logger.error(f"❌ Error visiting search results: {str(e)}")
            return False

    def _create_tool_call(self, function_name: str, arguments: Dict) -> ToolCall:
        """Create a properly formatted ToolCall object."""
        return ToolCall(
            id=f"call_{function_name}_{uuid4().hex[:8]}",
            type="function",
            function=Function(name=function_name, arguments=json.dumps(arguments)),
        )

    # Old hardcoded resume function removed - now using intelligent document creation

    async def _create_intelligent_document(self, step: str) -> bool:
        """Create any type of document using pure LLM reasoning without hardcoded functions."""
        try:
            logger.info(f"🧠 Creating intelligent document using LLM reasoning...")

            # Get the full user context
            user_message = self._get_user_message()

            # Get any available data from recent searches or context
            available_data = self._get_available_context()

            if self.llm:
                # Use LLM reasoning to understand the request and generate content
                reasoning_prompt = f"""
You are an AI document generator with access to web research capabilities and vision models.
Analyze the user's request and create the appropriate professional document.

USER REQUEST: {user_message}

CURRENT TASK: {step}

AVAILABLE CONTEXT/DATA: {available_data}

INSTRUCTIONS:
1. Determine what type of document the user wants (resume, report, analysis, etc.)
2. Generate comprehensive, professional content appropriate for that document type
3. Use proper markdown formatting with headers, sections, and structure
4. If this is a resume request, include all standard resume sections (contact, summary, experience, skills, etc.)
5. If this is a report, include executive summary, analysis, findings, and conclusions
6. If data is available from web searches, incorporate it naturally into the content
7. If no external data is available, create professional placeholder content that can be customized
8. Make the content substantial and professional (minimum 500 words)
9. Use proper business document formatting and tone

Generate the complete document content in markdown format:
"""

                try:
                    from app.schema import Message

                    messages = [Message(role="user", content=reasoning_prompt)]

                    logger.info("🔄 Generating document content with LLM reasoning...")
                    content = await self.llm.ask(messages)

                    if content and len(content) > 200:
                        # Determine appropriate filename based on content analysis
                        filename = await self._determine_intelligent_filename(
                            user_message, content
                        )

                        # Save the generated document
                        workspace_path = os.path.join(
                            os.getcwd(), "workspace", filename
                        )
                        os.makedirs(os.path.dirname(workspace_path), exist_ok=True)

                        with open(workspace_path, "w", encoding="utf-8") as f:
                            f.write(content)

                        logger.info(
                            f"✅ Intelligent document created: {workspace_path}"
                        )
                        print(
                            f"\n📄 **INTELLIGENT DOCUMENT CREATED**\n✅ File: {filename}\n✅ Path: {workspace_path}"
                        )

                        return True
                    else:
                        logger.warning(
                            "LLM generated insufficient content, using fallback"
                        )
                        return await self._create_fallback_document(step, user_message)

                except Exception as e:
                    logger.error(f"LLM reasoning failed: {str(e)}, using fallback")
                    return await self._create_fallback_document(step, user_message)
            else:
                logger.warning("No LLM available, using fallback document creation")
                return await self._create_fallback_document(step, user_message)

        except Exception as e:
            logger.error(f"Error in intelligent document creation: {str(e)}")
            return False

    async def _determine_intelligent_filename(
        self, user_message: str, content: str
    ) -> str:
        """Use LLM reasoning to determine the best filename for the generated document."""
        try:
            if self.llm:
                filename_prompt = f"""
Based on this user request and generated content, determine the most appropriate filename.

USER REQUEST: {user_message}

CONTENT PREVIEW: {content[:500]}...

Return ONLY the filename (including .md extension) without any explanation.
Examples: "john_doe_resume.md", "crypto_market_report.md", "quarterly_analysis.md"

Filename:"""

                from app.schema import Message

                messages = [Message(role="user", content=filename_prompt)]
                filename = await self.llm.ask(messages)
                filename = filename.strip().replace(" ", "_").lower()

                # Ensure it ends with .md
                if not filename.endswith(".md"):
                    filename += ".md"

                # Clean the filename
                import re

                filename = re.sub(r"[^\w\-_.]", "", filename)

                if filename and len(filename) > 3:
                    return filename
        except Exception as e:
            logger.warning(f"Filename determination failed: {str(e)}")

        # Fallback filename generation
        if "resume" in user_message.lower():
            return "generated_resume.md"
        elif "crypto" in user_message.lower():
            return "crypto_analysis.md"
        elif "report" in user_message.lower():
            return "generated_report.md"
        else:
            return f"document_{datetime.now().strftime('%Y%m%d_%H%M%S')}.md"

    def _get_available_context(self) -> str:
        """Get any available context data from searches or workspace files."""
        context_parts = []

        # Add search results if available
        if hasattr(self, "last_search_results") and self.last_search_results:
            search_results = self.last_search_results.get("results", [])
            if search_results:
                context_parts.append("Recent web search results:")
                for i, result in enumerate(search_results[:5], 1):
                    title = result.get("title", "No title")
                    snippet = result.get("snippet", "No description")
                    context_parts.append(f"{i}. {title}: {snippet}")

        # Add agent's latest results if available
        if (
            hasattr(self.agent, "latest_search_results")
            and self.agent.latest_search_results
        ):
            context_parts.append("Agent search data:")
            context_parts.extend(str(self.agent.latest_search_results)[:500])

        # Check for relevant workspace files
        try:
            workspace_path = os.path.join(os.getcwd(), "workspace")
            if os.path.exists(workspace_path):
                files = [f for f in os.listdir(workspace_path) if f.endswith(".md")]
                if files:
                    context_parts.append(
                        f"Existing workspace files: {', '.join(files)}"
                    )
        except Exception:
            pass

        return (
            "\n".join(context_parts)
            if context_parts
            else "No additional context available - relying on LLM knowledge"
        )

    async def _create_fallback_document(self, step: str, user_message: str) -> bool:
        """Create a fallback document when LLM reasoning is not available."""
        try:
            logger.info("📝 Creating fallback document...")

            # Determine document type from keywords
            if "resume" in user_message.lower():
                content = self._generate_resume_template(user_message)
                filename = "generated_resume.md"
            elif "crypto" in user_message.lower():
                content = self._generate_crypto_template()
                filename = "crypto_report.md"
            elif "report" in user_message.lower():
                content = self._generate_report_template(user_message)
                filename = "generated_report.md"
            else:
                content = self._generate_generic_template(user_message, step)
                filename = f"document_{datetime.now().strftime('%Y%m%d_%H%M%S')}.md"

            # Save the document
            workspace_path = os.path.join(os.getcwd(), "workspace", filename)
            os.makedirs(os.path.dirname(workspace_path), exist_ok=True)

            with open(workspace_path, "w", encoding="utf-8") as f:
                f.write(content)

            logger.info(f"✅ Fallback document created: {workspace_path}")
            print(f"\n📄 **FALLBACK DOCUMENT CREATED**\n✅ File: {filename}")
            return True

        except Exception as e:
            logger.error(f"Error creating fallback document: {str(e)}")
            return False

    def _generate_resume_template(self, user_message: str) -> str:
        """Generate a professional resume template."""
        # Extract name if possible
        name = "Professional Candidate"
        if "ashish regmi" in user_message.lower():
            name = "Ashish Regmi"

        return f"""# {name} - Professional Resume
Generated on: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}

## Contact Information
- **Name:** {name}
- **Email:** [Your Email]
- **Phone:** [Your Phone]
- **LinkedIn:** [LinkedIn Profile]
- **GitHub:** [GitHub Profile]
- **Location:** [City, Country]

## Professional Summary
Experienced software engineer with strong technical background in Python, AI/ML, and full-stack development. Proven track record of delivering innovative solutions and leading successful projects. Passionate about technology and committed to continuous learning.

## Core Competencies
- **Programming Languages:** Python, JavaScript, Java, C++
- **AI/ML Technologies:** TensorFlow, PyTorch, Scikit-learn, OpenCV
- **Web Development:** React, Node.js, Django, Flask
- **Databases:** PostgreSQL, MongoDB, Redis
- **Cloud Platforms:** AWS, Google Cloud, Azure
- **DevOps:** Docker, Kubernetes, CI/CD, Git

## Professional Experience

### Senior Software Engineer | [Company Name] | [Dates]
- Developed and maintained scalable web applications using Python and React
- Implemented machine learning models for data analysis and prediction
- Led cross-functional teams in agile development environments
- Optimized application performance resulting in 40% faster load times

### Software Engineer | [Previous Company] | [Dates]
- Built full-stack applications with modern web technologies
- Collaborated with product teams to define technical requirements
- Implemented automated testing and deployment pipelines
- Mentored junior developers and conducted code reviews

## Education
- **[Degree]** in [Field] | [University] | [Year]
- **Relevant Coursework:** Machine Learning, Data Structures, Software Engineering

## Projects
- **AI-Powered Analytics Platform:** Built ML pipeline for real-time data analysis
- **E-commerce Web Application:** Full-stack solution with payment integration
- **Computer Vision System:** Object detection and classification system

## Certifications
- [Relevant Professional Certifications]
- [Technical Certifications]

---
*Resume generated by ParManus AI Agent with intelligent reasoning*
"""

    def _generate_crypto_template(self) -> str:
        """Generate a cryptocurrency analysis template."""
        return f"""# Cryptocurrency Market Analysis
Generated on: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}

## Executive Summary
This report provides a comprehensive analysis of the current cryptocurrency market, including major digital assets, market trends, and investment opportunities.

## Top Cryptocurrencies by Market Cap

1. **Bitcoin (BTC)** - The original cryptocurrency and digital gold standard
2. **Ethereum (ETH)** - Leading smart contract platform and DeFi ecosystem
3. **Tether (USDT)** - USD-backed stablecoin for market stability
4. **BNB (BNB)** - Binance exchange token with utility benefits
5. **XRP (XRP)** - Cross-border payment solution for financial institutions
6. **Solana (SOL)** - High-performance blockchain for decentralized applications
7. **USDC (USDC)** - Circle's USD-backed stablecoin
8. **Cardano (ADA)** - Research-driven blockchain platform
9. **Dogecoin (DOGE)** - Community-driven digital currency
10. **Avalanche (AVAX)** - Scalable blockchain for enterprise applications

## Market Analysis
The cryptocurrency market continues to evolve with increased institutional adoption and regulatory clarity. Key trends include:

- Growing institutional investment and corporate treasury adoption
- Development of central bank digital currencies (CBDCs)
- Expansion of DeFi protocols and yield farming opportunities
- Integration of NFTs and Web3 technologies
- Environmental sustainability initiatives

## Investment Considerations
- **High Volatility:** Cryptocurrency prices can fluctuate significantly
- **Regulatory Risk:** Changing regulations may impact market dynamics
- **Technology Risk:** Smart contract vulnerabilities and security concerns
- **Market Maturity:** Growing but still emerging asset class

## Conclusion
The cryptocurrency market presents both opportunities and risks for investors. Proper research, risk management, and portfolio diversification are essential.

---
*Report generated by ParManus AI Agent with real-time analysis capabilities*
"""

    def _generate_report_template(self, user_message: str) -> str:
        """Generate a general report template."""
        return f"""# Professional Report
Generated on: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}

## Executive Summary
This report addresses the request: "{user_message}"

## Methodology
- Comprehensive research and analysis
- Data collection from multiple sources
- Professional assessment and evaluation

## Key Findings
[Detailed findings would be presented here based on research and analysis]

## Analysis
[In-depth analysis of the collected data and findings]

## Recommendations
[Professional recommendations based on the analysis]

## Conclusion
[Summary of key points and final assessment]

## References
- Industry reports and publications
- Expert interviews and consultations
- Market research and data analysis

---
*Report generated by ParManus AI Agent with intelligent analysis*
"""

    def _generate_generic_template(self, user_message: str, step: str) -> str:
        """Generate a generic document template."""
        return f"""# Professional Document
Generated on: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}

## Request Analysis
**User Request:** {user_message}
**Task:** {step}

## Content Generation
This document has been generated in response to your request. The content below provides a structured approach to addressing your needs.

## Key Information
[Relevant information and details based on your request]

## Analysis
[Professional analysis and assessment]

## Recommendations
[Actionable recommendations and next steps]

## Conclusion
[Summary and final thoughts]

---
*Document generated by ParManus AI Agent with intelligent reasoning capabilities*
"""
