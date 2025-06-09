from app.agent.base import BaseAgent
from app.llm_optimized import LLMOptimized
from app.logger import logger


class BrowserAgent(BaseAgent):
    def __init__(self, llm: LLMOptimized):
        super().__init__(llm)
        self.llm = llm

    async def handle_url(self, url: str) -> str:
        """Handle URL navigation requests."""
        try:
            success = await self._browse_url(url)
            if success:
                return f"Successfully navigated to {url}"
            return f"Failed to navigate to {url}"
        except Exception as e:
            logger.error(f"Error handling URL: {e}")
            return f"Error: {str(e)}"

    async def _browse_url(self, url: str) -> bool:
        """Internal method to handle browser navigation."""
        try:
            await self.llm.initialize_browser()
            success = await self.llm.navigate(url)
            if not success:
                logger.error(f"Failed to browse {url}")
            return success
        except Exception as e:
            logger.error(f"Browser navigation failed: {e}")
            return False

    async def think(self, request: str) -> str:
        """Main entry point for handling browser requests."""
        # Check if request is a URL or navigation command
        if "go to" in request.lower() or "navigate to" in request.lower():
            url = (
                request.lower().replace("go to", "").replace("navigate to", "").strip()
            )
            return await self.handle_url(url)

        return await super().think(request)
