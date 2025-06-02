from app.agent.base import BaseAgent


class FileAgent(BaseAgent):
    """
    Stub implementation of FileAgent.
    This is a placeholder until the full implementation is developed.
    """

    name = "file"

    @classmethod
    async def create(cls, **kwargs):
        """Factory method to create and properly initialize a FileAgent instance."""
        instance = cls(**kwargs)
        return instance
