"""Agent routing system based on Parmanus's Interaction class."""

from typing import Any, Dict, List, Optional

from app.agent.base import BaseAgent
from app.agent.browser import BrowserAgent
from app.agent.manus import Manus
from app.config import config
from app.gpu_manager import GPUManager, get_gpu_manager
from app.llm_optimized import LLMOptimized
from app.logger import logger


class AgentRouter:
    """Routes user queries to the appropriate specialized agent."""

    def __init__(self, llm: Optional[LLMOptimized] = None):
        """Initialize the agent router.

        Args:
            llm: LLMOptimized instance to use for agents
        """
        self.agents: Dict[str, BaseAgent] = {}
        self.current_agent: Optional[BaseAgent] = None
        self.default_agent_name = "manus"
        self.llm = (
            llm if llm and isinstance(llm, LLMOptimized) else LLMOptimized(config.llm)
        )

        # Initialize default agents
        self._initialize_default_agents()

    def _initialize_default_agents(self):
        """Initialize default agents for the router."""
        # Add Manus agent as default
        self.agents["manus"] = Manus(llm=self.llm)

        try:
            # Add browser agent with error handling
            browser_agent = BrowserAgent(llm=self.llm)
            self.agents["browser"] = browser_agent
            logger.info("Browser agent initialized successfully")
        except Exception as e:
            logger.error(f"Failed to initialize browser agent: {e}")
            # Continue without browser agent

    async def route(self, user_input: str) -> str:
        """Route the user input to appropriate agent and get response.

        Args:
            user_input: The user's input text

        Returns:
            The agent's response text
        """
        try:
            # Determine appropriate agent
            if "browse" in user_input.lower() or "go to" in user_input.lower():
                if "browser" not in self.agents:
                    return "Browser functionality is not available."
                self.current_agent = self.agents["browser"]
            else:
                self.current_agent = self.agents[self.default_agent_name]

            # Process request through selected agent
            response = await self.current_agent.run(user_input)
            return response

        except Exception as e:
            logger.error(f"Error routing request: {e}")
            return f"Error processing request: {str(e)}"
