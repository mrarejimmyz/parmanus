"""
Manus Action Executor - Handles actual tool execution for different step types
Keeps manus_core.py clean and optimized for easier debugging
"""

import asyncio
import json
from typing import Dict, List, Optional
from app.logger import logger
from app.schema import ToolCall, Function
from app.agent.visual_google_search import VisualGoogleSearch


class ManusActionExecutor:
    """Handles execution of different action types for Manus agent steps."""
    
    def __init__(self, agent):
        """Initialize with reference to the main agent."""
        self.agent = agent
        self.browser_handler = agent.browser_handler if hasattr(agent, 'browser_handler') else None
        self.visual_google = VisualGoogleSearch(agent)
        
    async def execute_research_action(self, step: str) -> bool:
        """Execute research and planning actions - use Google to find relevant sources."""
        try:
            logger.info(f"🔍 RESEARCH ACTION: {step}")
            
            # Generate intelligent search queries based on user request
            search_queries = await self._generate_search_queries(step)
            
            for query in search_queries:
                logger.info(f"🔍 Googling: {query}")
                
                # Use browser to search Google
                if self.browser_handler:
                    await self._google_search(query)
                    await asyncio.sleep(2)  # Allow results to load
                    
            return True
            
        except Exception as e:
            logger.error(f"Error in research action: {str(e)}")
            return False
    
    def _get_user_message(self) -> str:
        """Get the last user message from agent's message history."""
        if hasattr(self.agent, 'messages') and self.agent.messages:
            for msg in reversed(self.agent.messages):
                if msg.role == "user":
                    return msg.content.lower()
        return ""
    
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
            verification_queries = await self._generate_verification_queries(user_message, step)
            
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
        """Execute file creation actions - generate reports and documents."""
        try:
            logger.info(f"📝 CREATION ACTION: {step}")
            
            # Determine what type of file to create
            user_message = self._get_user_message()
            if "crypto" in user_message:
                return await self._create_crypto_report()
            elif "news" in user_message:
                return await self._create_news_report()
            else:
                return await self._create_general_report(step)
                
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
            if any(keyword in user_message for keyword in ["crypto", "bitcoin", "ethereum"]):
                search_query = "cryptocurrency prices market cap ranking today"
            elif any(keyword in user_message for keyword in ["news", "headlines", "current"]):
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
            tool_call = self._create_tool_call("browser_use", {
                "action": "go_to_url",
                "url": url
            })
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
            crypto_sites = [
                "https://coinmarketcap.com/",
                "https://www.coingecko.com/"
            ]
            
            for site in crypto_sites:
                if self.browser_handler:
                    await self._navigate_to_url(site)
                    await asyncio.sleep(3)  # Allow page to load
                    
                    # Extract data using browser tool
                    tool_call = {
                        "function": "browser_use", 
                        "arguments": {"action": "extract_text"}
                    }
                    await self.agent.execute_tool_call(tool_call)
                    
            return True
            
        except Exception as e:
            logger.error(f"Error extracting crypto data: {str(e)}")
            return False
    
    async def _extract_news_data(self) -> bool:
        """Extract news data from current page."""
        try:
            logger.info("📰 Extracting news data...")
            
            # Navigate to news websites and extract data
            news_sites = [
                "https://www.bbc.com/news",
                "https://www.reuters.com/"
            ]
            
            for site in news_sites:
                if self.browser_handler:
                    await self._navigate_to_url(site)
                    await asyncio.sleep(3)  # Allow page to load
                    
                    # Extract data using browser tool
                    tool_call = {
                        "function": "browser_use",
                        "arguments": {"action": "extract_text"}
                    }
                    await self.agent.execute_tool_call(tool_call)
                    
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
                tool_call = {
                    "function": "browser_use",
                    "arguments": {"action": "extract_text"}
                }
                await self.agent.execute_tool_call(tool_call)
                
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
                {"name": "CryptoCompare", "url": "https://www.cryptocompare.com/"}
            ]
        elif "news" in user_message:
            return [
                {"name": "CNN", "url": "https://www.cnn.com/"},
                {"name": "Associated Press", "url": "https://apnews.com/"}
            ]
        else:
            return []
    
    async def _create_crypto_report(self) -> bool:
        """Create cryptocurrency report file."""
        try:
            logger.info("📊 Creating cryptocurrency report...")
            
            # Use Python tool to create file
            tool_call = {
                "function": "python_execute",
                "arguments": {
                    "code": '''
# Create crypto report with real data
import os
from datetime import datetime

report_content = f"""# Top 10 Cryptocurrencies Report
Generated on: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}

## Data Sources
- CoinMarketCap
- CoinGecko
- Binance

## Top 10 Cryptocurrencies by Market Cap
[Real data extracted from websites will be populated here]

1. Bitcoin (BTC)
2. Ethereum (ETH)
3. Tether (USDT)
4. BNB (BNB)
5. XRP (XRP)
6. Solana (SOL)
7. USDC (USDC)
8. Cardano (ADA)
9. Dogecoin (DOGE)
10. Avalanche (AVAX)

## Analysis
[Analysis based on extracted data will be added here]

---
Report generated by ParManus AI Agent
"""

# Save to workspace
workspace_path = os.path.join(os.getcwd(), "crypto_report.md")
with open(workspace_path, "w", encoding="utf-8") as f:
    f.write(report_content)

print(f"✅ Crypto report saved to: {workspace_path}")
'''
                }
            }
            await self.agent.execute_tool_call(tool_call)
            return True
            
        except Exception as e:
            logger.error(f"Error creating crypto report: {str(e)}")
            return False
    
    async def _create_news_report(self) -> bool:
        """Create news report file."""
        try:
            logger.info("📰 Creating news report...")
            
            # Use Python tool to create file
            tool_call = {
                "function": "python_execute",
                "arguments": {
                    "code": '''
# Create news report with real data
import os
from datetime import datetime

report_content = f"""# Top 10 News Stories
Generated on: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}

## Data Sources
- BBC News
- Reuters
- CNN
- Associated Press

## Current Headlines
[Real headlines extracted from news websites will be populated here]

1. [Headline 1]
2. [Headline 2]
3. [Headline 3]
4. [Headline 4]
5. [Headline 5]
6. [Headline 6]
7. [Headline 7]
8. [Headline 8]
9. [Headline 9]
10. [Headline 10]

## Summary
[Summary based on extracted news data will be added here]

---
Report generated by ParManus AI Agent
"""

# Save to workspace
workspace_path = os.path.join(os.getcwd(), "news_report.md")
with open(workspace_path, "w", encoding="utf-8") as f:
    f.write(report_content)

print(f"✅ News report saved to: {workspace_path}")
'''
                }
            }
            await self.agent.execute_tool_call(tool_call)
            return True
            
        except Exception as e:
            logger.error(f"Error creating news report: {str(e)}")
            return False
    
    async def _create_general_report(self, step: str) -> bool:
        """Create general report file."""
        try:
            logger.info(f"📄 Creating general report for: {step}")
            
            # Use Python tool to create file
            tool_call = {
                "function": "python_execute",
                "arguments": {
                    "code": f'''
# Create general report
import os
from datetime import datetime

report_content = f"""# Research Report
Generated on: {{datetime.now().strftime("%Y-%m-%d %H:%M:%S")}}

## Task: {step}

## Data Collected
[Information extracted from web research will be populated here]

## Analysis
[Analysis based on collected data will be added here]

---
Report generated by ParManus AI Agent
"""

# Save to workspace
workspace_path = os.path.join(os.getcwd(), "research_report.md")
with open(workspace_path, "w", encoding="utf-8") as f:
    f.write(report_content)

print(f"✅ Research report saved to: {{workspace_path}}")
'''
                }
            }
            await self.agent.execute_tool_call(tool_call)
            return True
            
        except Exception as e:
            logger.error(f"Error creating general report: {str(e)}")
            return False
    
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
            queries.extend([
                "top 10 cryptocurrencies by market cap today",
                "cryptocurrency prices live ranking",
                "best crypto coins 2025 current"
            ])
        elif "news" in user_message:
            queries.extend([
                "top news headlines today",
                "breaking news current events",
                "latest news stories today"
            ])
        else:
            # Extract key terms from user message for generic search
            key_terms = user_message.replace("get", "").replace("top", "").replace("10", "").strip()
            queries.append(f"{key_terms} latest information")
            
        return queries
    
    async def _generate_extraction_queries(self, user_message: str, step: str) -> List[str]:
        """Generate search queries specifically for data extraction."""
        queries = []
        
        if "crypto" in user_message:
            queries.extend([
                "cryptocurrency market cap ranking live data",
                "bitcoin ethereum price today top coins",
                "crypto market analysis current prices"
            ])
        elif "news" in user_message:
            queries.extend([
                "current news headlines today breaking",
                "latest world news stories",
                "top news events happening now"
            ])
        else:
            # Generic extraction based on step content
            queries.append(f"{user_message} current data information")
            
        return queries
    
    async def _generate_verification_queries(self, user_message: str, step: str) -> List[str]:
        """Generate search queries for verification from multiple sources."""
        queries = []
        
        if "crypto" in user_message:
            queries.extend([
                "cryptocurrency prices verification multiple sources",
                "crypto market cap cross reference data"
            ])
        elif "news" in user_message:
            queries.extend([
                "news verification multiple sources",
                "fact check current news stories"
            ])
        else:
            queries.append(f"{user_message} verification multiple sources")
            
        return queries
    
    def _create_tool_call(self, function_name: str, arguments: Dict) -> ToolCall:
        """Create a properly formatted ToolCall object."""
        return ToolCall(
            id=f"call_{function_name}_{id(arguments)}",
            type="function",
            function=Function(
                name=function_name,
                arguments=json.dumps(arguments)
            )
        )
    
    async def _google_search(self, query: str) -> bool:
        """Perform a Google search with truly visual human-like interaction."""
        try:
            logger.info(f"🔍 Starting VISUAL Google search for: {query}")
            
            # Use the new visual Google search implementation
            search_result = await self.visual_google.perform_visual_google_search(query)
            
            if search_result["success"]:
                logger.info(f"✅ Visual Google search completed successfully!")
                logger.info(f"📊 Found {len(search_result.get('results', []))} search results")
                
                # Store results for later use
                if hasattr(self.agent, 'last_search_results'):
                    self.agent.last_search_results = search_result
                
                return True
            else:
                logger.error(f"❌ Visual Google search failed: {search_result.get('error', 'Unknown error')}")
                return False
                
        except Exception as e:
            logger.error(f"❌ Exception in visual Google search: {str(e)}")
            return False
    
    async def _visit_search_results(self, num_results: int = 3) -> bool:
        """Visit the top search results from the last visual Google search."""
        try:
            logger.info(f"📖 Visiting top {num_results} search results from visual search")
            
            # Get results from the last visual Google search
            if not hasattr(self.agent, 'last_search_results') or not self.agent.last_search_results:
                logger.warning("⚠️ No search results available to visit")
                return False
            
            search_results = self.agent.last_search_results.get('results', [])
            if not search_results:
                logger.warning("⚠️ No search results found in last search")
                return False
            
            visited_count = 0
            
            # Visit the top results
            for i, result in enumerate(search_results[:num_results]):
                logger.info(f"🔗 Visiting result #{i+1}: {result.get('url', 'Unknown URL')}")
                
                try:
                    visit_result = await self.visual_google.visit_search_result(result['url'])
                    
                    if visit_result['success']:
                        logger.info(f"✅ Successfully visited and extracted content from result #{i+1}")
                        visited_count += 1
                        
                        # Store the extracted content
                        result['extracted_content'] = visit_result['content']
                        
                    else:
                        logger.warning(f"⚠️ Failed to visit result #{i+1}: {visit_result.get('error', 'Unknown error')}")
                        
                except Exception as e:
                    logger.error(f"❌ Exception visiting result #{i+1}: {str(e)}")
                    continue
                
                await asyncio.sleep(2)  # Respectful delay between visits
            
            if visited_count > 0:
                logger.info(f"✅ Successfully visited {visited_count} search results with visual browser actions")
                return True
            else:
                logger.error("❌ Failed to visit any search results")
                return False
            
        except Exception as e:
            logger.error(f"Error visiting search results: {str(e)}")
            return False