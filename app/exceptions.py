class ToolError(Exception):
    """Raised when a tool encounters an error."""

    def __init__(self, message):
        self.message = message


class ParManusError(Exception):
    """Base exception for all ParManus errors"""


class TokenLimitExceeded(ParManusError):
    """Exception raised when the token limit is exceeded"""


class AgentTaskComplete(ParManusError):
    """Exception raised when an agent has completed its task and should terminate."""

    def __init__(self, message="Task completed successfully"):
        self.message = message
        super().__init__(self.message)
