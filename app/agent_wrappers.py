from typing import Any

from app.agent.browser import BrowserAgent
from app.agent.code import CodeAgent
from app.agent.manus_core import Manus
from app.config import Config
from app.logger import logger


class SimpleAgent:
    """Simplified agent for basic functionality when ParManus is not available."""

    def __init__(self, name: str, llm, config: Config):
        self.name = name
        self.llm = llm
        self.config = config
        self.messages = []

    async def run(self, prompt: str) -> str:
        """Run the agent with a simple prompt."""
        try:
            system_prompt = self._get_system_prompt()
            messages = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": prompt},
            ]

            response = await self.llm.ask(messages)
            return response

        except Exception as e:
            logger.error(f"Error in simple agent {self.name}: {e}")
            return f"Error: {e}"

    def _get_system_prompt(self) -> str:
        """Get system prompt based on agent type."""
        prompts = {
            "manus": "You are Manus, a helpful AI assistant. Provide clear, accurate, and helpful responses.",
            "code": "You are a coding assistant. Help with programming tasks, debugging, and code explanations.",
            "browser": "You are a browser automation assistant. Help with web scraping and browser tasks.",
            "file": "You are a file management assistant. Help with file operations and data processing.",
            "planner": "You are a planning assistant. Help break down tasks and create actionable plans.",
        }
        return prompts.get(self.name, prompts["manus"])


class ParManusAgentWrapper:
    """Wrapper for full ParManus agents."""

    def __init__(self, agent_class, llm, config: Config):
        self.agent_class = agent_class
        self.llm = llm
        self.config = config
        self.agent = None

    async def run(self, prompt: str) -> str:
        """Run the ParManus agent."""
        try:
            if not self.agent:
                # Initialize agent
                if self.agent_class == Manus:
                    logger.debug("Attempting to create Manus agent...")
                    self.agent = await Manus.create()
                    logger.debug("Manus agent created successfully.")
                else:
                    self.agent = self.agent_class()

                # Set LLM
                self.agent.llm = self.llm

            # Run agent
            result = await self.agent.run(prompt)
            return result

        except Exception as e:
            logger.error(f"Error in ParManus agent: {e}", exc_info=True)
            logger.error(f"Type of error: {type(e)}")
            logger.error(f"Representation of error: {repr(e)}")
            return f"Error: {e}"
        finally:
            # Cleanup
            if self.agent and hasattr(self.agent, "cleanup"):
                try:
                    await self.agent.cleanup()
                except Exception as e:
                    logger.warning(f"Error during agent cleanup: {e}")

    # Removed _create_parmanus_config method


def create_agent(agent_name: str, llm, config: Config):
    """Create appropriate agent based on name and availability."""
    # Assuming PARMANUS_AVAILABLE is handled in main.py or passed in
    # For now, we'll assume full agents are always available if imported

    agent_map = {
        "manus": Manus,
        "code": CodeAgent,
        "browser": BrowserAgent,
    }

    agent_class = agent_map.get(agent_name)
    if agent_class:
        return ParManusAgentWrapper(agent_class, llm, config)

    # Fallback to simple agent
    return SimpleAgent(agent_name, llm, config)
