from app.agent.base import BaseAgent


class PlannerAgent(BaseAgent):
    """
    Stub implementation of PlannerAgent.
    This is a placeholder until the full implementation is developed.
    """

    name = "planner"

    @classmethod
    async def create(cls, **kwargs):
        """Factory method to create and properly initialize a PlannerAgent instance."""
        instance = cls(**kwargs)
        return instance
