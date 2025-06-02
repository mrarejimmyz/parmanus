"""Agent routing system based on Parmanus's Interaction class."""

from typing import Any, Dict, List, Optional

from app.agent.base import BaseAgent
from app.agent.manus import Manus
from app.logger import logger


class AgentRouter:
    """Routes user queries to the appropriate specialized agent."""

    def __init__(self, agents: Optional[List[BaseAgent]] = None):
        """Initialize the agent router.

        Args:
            agents: List of available agents. If None, default agents will be created.
        """
        self.agents: Dict[str, BaseAgent] = {}
        self.current_agent: Optional[BaseAgent] = None
        self.default_agent_name = "manus"

        # Initialize default agents if none provided
        if agents is None:
            self._initialize_default_agents()
        else:
            for agent in agents:
                self.agents[agent.name.lower()] = agent

    def _initialize_default_agents(self):
        """Initialize default agents for the router."""
        # For now, we'll start with the existing Manus agent
        # Additional agents will be added as we implement them
        self.agents["manus"] = None  # Will be created when needed

    async def route(self, query: str) -> BaseAgent:
        """Route the query to the most appropriate agent.

        Args:
            query: The user query to analyze and route.

        Returns:
            The selected agent for handling the query.
        """
        # Analyze query to determine best agent
        agent_name = await self._analyze_query(query)

        # Get or create the selected agent
        if agent_name not in self.agents or self.agents[agent_name] is None:
            self.agents[agent_name] = await self._create_agent(agent_name)

        self.current_agent = self.agents[agent_name]
        logger.info(f"Routed query to {agent_name} agent")
        return self.current_agent

    async def _analyze_query(self, query: str) -> str:
        """Analyze the query to determine the best agent.

        Args:
            query: The user query to analyze.

        Returns:
            The name of the best agent for this query.
        """
        query_lower = query.lower()
        
        # Log the query for debugging
        logger.info(f"Analyzing query for routing: {query}")

        # Browser-related queries (check first for web content)
        if any(
            keyword in query_lower
            for keyword in [
                "browse",
                "website",
                "web",
                "www",
                "http",
                "url",
                "search",
                "click",
                "navigate",
                "download",
                "scrape",
                "form",
                "button",
                "google.com",
                "rate",
                "feedback",
                "visit",
                "page"
            ]
        ):
            logger.info("Routing to browser agent based on web keywords")
            return "browser"

        # Code-related queries
        if any(
            keyword in query_lower
            for keyword in [
                "code",
                "program",
                "script",
                "function",
                "debug",
                "compile",
                "execute",
                "python",
                "javascript",
                "java",
                "c++",
                "go",
                "rust",
            ]
        ):
            logger.info("Routing to code agent based on programming keywords")
            return "code"

        # File-related queries
        if any(
            keyword in query_lower
            for keyword in [
                "file",
                "folder",
                "directory",
                "save",
                "read",
                "write",
                "delete",
                "copy",
                "move",
                "create",
                "edit",
            ]
        ):
            logger.info("Routing to file agent based on file keywords")
            return "file"

        # Planning-related queries
        if any(
            keyword in query_lower
            for keyword in [
                "plan",
                "task",
                "step",
                "organize",
                "schedule",
                "workflow",
                "project",
                "break down",
                "strategy",
            ]
        ):
            logger.info("Routing to planner agent based on planning keywords")
            return "planner"

        # Default to manus agent for general queries
        logger.info("Routing to default manus agent")
        return self.default_agent_name

    async def _create_agent(self, agent_name: str) -> BaseAgent:
        """Create an agent instance by name.

        Args:
            agent_name: The name of the agent to create.

        Returns:
            The created agent instance.
        """
        if agent_name == "manus":
            return await Manus.create()
        elif agent_name == "code":
            # Will be implemented when CodeAgent is created
            from app.agent.code import CodeAgent

            return await CodeAgent.create()
        elif agent_name == "browser":
            # Will be implemented when BrowserAgent is created
            from app.agent.browser import BrowserAgent

            return await BrowserAgent.create()
        elif agent_name == "file":
            # Will be implemented when FileAgent is created
            from app.agent.file import FileAgent

            return await FileAgent.create()
        elif agent_name == "planner":
            # Will be implemented when PlannerAgent is created
            from app.agent.planner import PlannerAgent

            return await PlannerAgent.create()
        else:
            # Fallback to manus agent
            logger.warning(f"Unknown agent type: {agent_name}, falling back to manus")
            return await Manus.create()

    def get_available_agents(self) -> List[str]:
        """Get list of available agent names.

        Returns:
            List of agent names that can be routed to.
        """
        return ["manus", "code", "browser", "file", "planner"]

    def get_current_agent(self) -> Optional[BaseAgent]:
        """Get the currently active agent.

        Returns:
            The current agent or None if no agent is active.
        """
        return self.current_agent
