class ToolError(Exception):
    """Raised when a tool encounters an error."""

    def __init__(self, message):
        self.message = message


class ParManusError(Exception):
    """Base exception for all ParManus errors"""

    pass


class TokenLimitExceeded(ParManusError):
    """Raised when a token limit is exceeded."""

    def __init__(self, message="Token limit exceeded"):
        self.message = message
        super().__init__(self.message)


class ModelTimeoutError(ParManusError):
    """Raised when model execution times out."""

    def __init__(self, message="Model execution timed out"):
        self.message = message
        super().__init__(self.message)
