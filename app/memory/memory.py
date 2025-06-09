from enum import Enum
from typing import Any, Dict, List, Optional, Union

from app.logger import logger
from app.schema import Message, Role


class Memory:
    """Manages conversation history."""

    def __init__(self):
        """Initialize an empty message list."""
        self._messages: List[Message] = []

    @property
    def messages(self) -> List[Message]:
        """Get all messages."""
        return self._messages

    @messages.setter
    def messages(self, value: List[Union[Dict[str, Any], Message, str]]) -> None:
        """Set messages with type conversion."""
        if not isinstance(value, list):
            value = [value]

        self._messages = []
        for msg in value:
            try:
                if isinstance(msg, Message):
                    # Ensure Message object is valid
                    if not hasattr(msg, "role") or not hasattr(msg, "content"):
                        raise ValueError(
                            "Invalid Message object - missing role or content"
                        )
                    self._messages.append(msg)

                elif isinstance(msg, dict):
                    # Handle dictionary format
                    if "role" not in msg or "content" not in msg:
                        raise ValueError(
                            "Dict messages must have 'role' and 'content' keys"
                        )

                    role = str(msg["role"]).strip().lower()
                    if not role:
                        role = Role.USER.value

                    message = Message(role=role, content=str(msg["content"]).strip())
                    self._messages.append(message)

                elif isinstance(msg, str):
                    # Create Message from string with default user role
                    message = Message(role=Role.USER.value, content=str(msg).strip())
                    self._messages.append(message)
                else:
                    raise ValueError(f"Invalid message type: {type(msg)}")

                # Log successful message addition
                logger.debug(
                    f"Added message: {self._messages[-1].role}:{self._messages[-1].content[:100]}"
                )

            except Exception as e:
                logger.error(f"Error adding message: {e}")
                raise

    def add_message(
        self,
        role_or_message: Optional[Union[str, Message]] = None,
        content: Optional[str] = None,
        **metadata,
    ) -> None:
        """Add a new message to memory."""
        if role_or_message is None:
            raise ValueError("Role or message is required")

        try:
            if isinstance(role_or_message, Message):
                message = role_or_message
            elif isinstance(role_or_message, str):
                if content is None:
                    # Create a new Message with the string as content
                    message = Message(
                        role=Role.USER.value, content=role_or_message.strip()
                    )
                else:
                    # Use string as role with provided content
                    message = Message(
                        role=role_or_message.strip().lower(),
                        content=content.strip(),
                        **metadata,
                    )
            else:
                raise ValueError(f"Invalid message type: {type(role_or_message)}")

            # Validate message before adding
            if not isinstance(message, Message):
                raise ValueError("Failed to create valid Message object")

            self._messages.append(message)
            logger.debug(f"Added message: {message.role}:{message.content[:100]}")

        except Exception as e:
            logger.error(f"Error adding message: {e}")
            raise

    def get_last_message(self) -> Optional[Message]:
        """Get the most recent message."""
        return self._messages[-1] if self._messages else None

    def clear(self) -> None:
        """Clear all messages."""
        self._messages = []

    def to_dict_list(self) -> List[Dict[str, Any]]:
        """Convert all messages to dictionary format."""
        return [msg.to_dict() for msg in self._messages]

    def get_messages(self, limit: Optional[int] = None) -> List[Message]:
        """Get messages from memory.

        Args:
            limit: Maximum number of recent messages to return. If None, returns all messages.

        Returns:
            List of messages from memory.
        """
        if limit is None:
            return self._messages.copy()
        return self._messages[-limit:] if limit > 0 else []

    def __len__(self) -> int:
        """Return the number of messages in memory."""
        return len(self._messages)

    def __bool__(self) -> bool:
        """Return True if memory contains messages."""
        return bool(self._messages)
