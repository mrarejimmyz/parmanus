"""
Nodriver-based Visual Google Search Implementation
Enhanced to bypass captcha and anti-bot detection
"""

import asyncio
import json
import re
from typing import Dict, List, Optional
from urllib.parse import quote_plus

import nodriver as uc

from app.logger import logger


class NodriverGoogleSearch:
    """Handles Google search using Nodriver for stealth browsing"""

    def __init__(self, agent):
        self.agent = agent
        self.browser = None
        self.page = None

    async def perform_visual_google_search(self, query: str) -> Dict:
        """Perform Google search using Nodriver for stealth"""
        logger.info(f"🔍 Starting Nodriver Google search for: {query}")

        try:
            # Initialize browser if not already done
            if not self.browser:
                await self._initialize_browser()

            # Perform the search
            search_result = await self._execute_search(query)
            if not search_result["success"]:
                return search_result

            # Extract search results
            results = await self._extract_search_results()

            return {"success": True, "query": query, "results": results}

        except Exception as e:
            logger.error(f"❌ Nodriver Google search failed: {str(e)}")
            await self._cleanup_browser()
            return {"success": False, "error": str(e)}

    async def _initialize_browser(self) -> None:
        """Initialize Nodriver browser with stealth settings"""
        logger.info("🚀 Initializing Nodriver browser")

        try:
            # Configure browser options for maximum stealth
            # Configure browser options for maximum stealth
            config = uc.Config(
                headless=True,  # Run headless in sandbox
                user_data_dir=None,  # Use temporary profile
                no_sandbox=True,  # Required for root user
                browser_args=[
                    "--no-first-run",
                    "--no-default-browser-check",
                    "--disable-blink-features=AutomationControlled",
                    "--disable-web-security",
                    "--disable-features=VizDisplayCompositor",
                    "--disable-extensions",
                    "--no-sandbox",
                    "--disable-dev-shm-usage",
                    "--disable-gpu",
                    "--remote-debugging-port=0",
                    "--disable-background-timer-throttling",
                    "--disable-backgrounding-occluded-windows",
                    "--disable-renderer-backgrounding",
                    "--disable-field-trial-config",
                    "--disable-back-forward-cache",
                    "--disable-ipc-flooding-protection",
                    "--enable-features=NetworkService,NetworkServiceInProcess",
                    "--force-color-profile=srgb",
                    "--metrics-recording-only",
                    "--use-mock-keychain",
                ],
            )

            self.browser = await uc.start(config=config)
            self.page = await self.browser.get("https://www.google.com")

            # Wait for page to load
            await asyncio.sleep(2)

            # Execute stealth scripts
            await self._execute_stealth_scripts()

            logger.info("✅ Nodriver browser initialized successfully")

        except Exception as e:
            logger.error(f"❌ Failed to initialize Nodriver browser: {str(e)}")
            raise

    async def _execute_stealth_scripts(self) -> None:
        """Execute JavaScript to make browser more human-like"""
        stealth_scripts = [
            # Remove webdriver property
            "Object.defineProperty(navigator, 'webdriver', {get: () => undefined})",
            # Spoof plugins
            """
            Object.defineProperty(navigator, 'plugins', {
                get: () => [1, 2, 3, 4, 5]
            });
            """,
            # Spoof languages
            """
            Object.defineProperty(navigator, 'languages', {
                get: () => ['en-US', 'en']
            });
            """,
            # Override permissions
            """
            const originalQuery = window.navigator.permissions.query;
            window.navigator.permissions.query = (parameters) => (
                parameters.name === 'notifications' ?
                    Promise.resolve({ state: Notification.permission }) :
                    originalQuery(parameters)
            );
            """,
            # Mock chrome runtime
            """
            if (!window.chrome) {
                window.chrome = {};
            }
            if (!window.chrome.runtime) {
                window.chrome.runtime = {};
            }
            """,
        ]

        for script in stealth_scripts:
            try:
                await self.page.evaluate(script)
            except Exception as e:
                logger.warning(f"⚠️ Stealth script failed: {str(e)}")

    async def _execute_search(self, query: str) -> Dict:
        """Execute the Google search with human-like behavior"""
        logger.info(f"🔍 Executing search for: {query}")

        try:
            # Wait for page to be ready
            await self.page.wait_for('input[name="q"]', timeout=10)

            # Find search input
            search_input = await self.page.select('input[name="q"]')
            if not search_input:
                # Try alternative selectors
                search_input = await self.page.select('textarea[name="q"]')
                if not search_input:
                    return {"success": False, "error": "Search input not found"}

            # Clear any existing text and type query with human-like delays
            await search_input.click()
            await asyncio.sleep(0.1)

            # Clear existing content
            await search_input.send_keys("\x01")  # Ctrl+A equivalent
            await asyncio.sleep(0.1)

            # Type query with random delays between characters
            for char in query:
                await search_input.send_keys(char)
                await asyncio.sleep(
                    0.05 + (0.1 * asyncio.get_event_loop().time() % 0.1)
                )

            # Wait a bit before submitting
            await asyncio.sleep(0.5)

            # Submit search
            await search_input.send_keys("\r")  # Enter key

            # Wait for results to load
            await asyncio.sleep(3)

            # Check if we got search results or captcha
            page_content = await self.page.get_content()
            if (
                "captcha" in page_content.lower()
                or "unusual traffic" in page_content.lower()
            ):
                logger.warning("⚠️ Captcha detected, attempting to handle")
                return await self._handle_captcha()

            logger.info("✅ Search executed successfully")
            return {"success": True}

        except Exception as e:
            logger.error(f"❌ Search execution failed: {str(e)}")
            return {"success": False, "error": str(e)}

    async def _handle_captcha(self) -> Dict:
        """Handle captcha if encountered"""
        logger.info("🤖 Attempting to handle captcha")

        try:
            # Wait a bit and try to refresh
            await asyncio.sleep(5)
            await self.page.reload()
            await asyncio.sleep(3)

            # Check if captcha is gone
            page_content = await self.page.get_content()
            if (
                "captcha" not in page_content.lower()
                and "unusual traffic" not in page_content.lower()
            ):
                logger.info("✅ Captcha resolved after refresh")
                return {"success": True}

            # If still captcha, try a different approach
            logger.warning("⚠️ Captcha still present, trying new session")
            await self._cleanup_browser()
            await asyncio.sleep(10)  # Wait before retry
            await self._initialize_browser()

            return {"success": False, "error": "Captcha encountered, session reset"}

        except Exception as e:
            logger.error(f"❌ Captcha handling failed: {str(e)}")
            return {"success": False, "error": f"Captcha handling failed: {str(e)}"}

    async def _extract_search_results(self) -> List[Dict]:
        """Extract search results from Google results page"""
        logger.info("📊 Extracting search results")

        try:
            results = []

            # Wait for search results to be present
            await self.page.wait_for("div[data-ved]", timeout=10)

            # Extract search result elements
            search_results = await self.page.select_all("div.g")

            for i, result in enumerate(search_results[:10]):  # Limit to top 10
                try:
                    # Extract title
                    title_element = await result.select("h3")
                    title = (
                        await title_element.get_text()
                        if title_element
                        else f"Result {i+1}"
                    )

                    # Extract URL
                    link_element = await result.select("a")
                    url = (
                        await link_element.get_attribute("href") if link_element else ""
                    )

                    # Extract snippet
                    snippet_element = await result.select("span[data-ved]")
                    snippet = (
                        await snippet_element.get_text() if snippet_element else ""
                    )

                    if title and url:
                        results.append(
                            {
                                "title": title.strip(),
                                "url": url.strip(),
                                "snippet": (
                                    snippet.strip()[:200] + "..."
                                    if len(snippet) > 200
                                    else snippet.strip()
                                ),
                            }
                        )

                except Exception as e:
                    logger.warning(f"⚠️ Failed to extract result {i}: {str(e)}")
                    continue

            logger.info(f"✅ Extracted {len(results)} search results")
            return results

        except Exception as e:
            logger.error(f"❌ Failed to extract search results: {str(e)}")
            return []

    async def _cleanup_browser(self) -> None:
        """Clean up browser resources"""
        try:
            if self.page:
                await self.page.close()
                self.page = None
            if self.browser:
                await self.browser.stop()
                self.browser = None
            logger.info("🧹 Browser cleanup completed")
        except Exception as e:
            logger.warning(f"⚠️ Browser cleanup warning: {str(e)}")

    async def close(self) -> None:
        """Close the browser and clean up resources"""
        await self._cleanup_browser()


# Compatibility wrapper for existing code
class VisualGoogleSearch:
    """Compatibility wrapper that uses Nodriver implementation"""

    def __init__(self, agent):
        self.agent = agent
        self.nodriver_search = NodriverGoogleSearch(agent)

    async def perform_visual_google_search(self, query: str) -> Dict:
        """Perform Google search using Nodriver backend"""
        return await self.nodriver_search.perform_visual_google_search(query)

    async def close(self) -> None:
        """Close the search engine"""
        await self.nodriver_search.close()
