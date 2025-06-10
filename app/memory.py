"""Memory system based on Parmanus's Memory class."""

import json
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

from app.logger import logger
from app.schema import Message


class Memory:
    """Manages conversation history and context with session persistence."""

    def __init__(
        self,
        initial_prompt: Optional[str] = None,
        recover_last_session: bool = False,
        memory_compression: bool = False,
        model_provider: Optional[str] = None,
        session_dir: str = "sessions",
    ):
        """Initialize the memory system.

        Args:
            initial_prompt: Initial system prompt to set context.
            recover_last_session: Whether to recover the last saved session.
            memory_compression: Whether to compress old memories.
            model_provider: The LLM provider for memory compression.
            session_dir: Directory to store session files.
        """
        self.messages: List[Message] = []
        self.initial_prompt = initial_prompt
        self.recover_last_session = recover_last_session
        self.memory_compression = memory_compression
        self.model_provider = model_provider
        self.session_dir = Path(session_dir)
        self.session_file = self.session_dir / "last_session.json"

        # Create session directory if it doesn't exist
        self.session_dir.mkdir(exist_ok=True)

        # Add initial prompt if provided
        if initial_prompt:
            self.add_message(Message.system_message(initial_prompt))

        # Recover last session if requested
        if recover_last_session:
            self.load_session()

    def add_message(self, message: Message) -> None:
        """Add a message to memory.

        Args:
            message: The message to add to memory.
        """
        self.messages.append(message)
        logger.debug(f"Added {message.role} message to memory")

        # Trigger compression if enabled and memory is getting large
        if self.memory_compression and len(self.messages) > 50:
            self._compress_memory()

    def push(self, role: str, content: str, **kwargs) -> None:
        """Add a message to memory (Parmanus compatibility method).

        Args:
            role: The role of the message sender.
            content: The message content.
            **kwargs: Additional message parameters.
        """
        if role == "user":
            message = Message.user_message(content, **kwargs)
        elif role == "assistant":
            message = Message.assistant_message(content, **kwargs)
        elif role == "system":
            message = Message.system_message(content)
        elif role == "tool":
            message = Message.tool_message(content, **kwargs)
        else:
            logger.warning(f"Unknown role: {role}, treating as user message")
            message = Message.user_message(content, **kwargs)

        self.add_message(message)

    def get_messages(self, limit: Optional[int] = None) -> List[Message]:
        """Get messages from memory.

        Args:
            limit: Maximum number of recent messages to return.

        Returns:
            List of messages from memory.
        """
        if limit is None:
            return self.messages.copy()
        return self.messages[-limit:] if limit > 0 else []

    def get_context(self, max_tokens: int = 4000) -> List[Message]:
        """Get context messages within token limit.

        Args:
            max_tokens: Maximum tokens to include in context.

        Returns:
            List of messages that fit within the token limit.
        """
        # Simple approximation: 4 characters per token
        max_chars = max_tokens * 4
        total_chars = 0
        context_messages = []

        # Always include system messages
        system_messages = [msg for msg in self.messages if msg.role == "system"]
        for msg in system_messages:
            context_messages.append(msg)
            total_chars += len(msg.content or "")

        # Add recent messages in reverse order until we hit the limit
        for message in reversed(self.messages):
            if message.role == "system":
                continue  # Already included

            message_chars = len(message.content or "")
            if total_chars + message_chars > max_chars:
                break

            context_messages.insert(
                -len(system_messages) if system_messages else 0, message
            )
            total_chars += message_chars

        return context_messages

    def clear(self) -> None:
        """Clear all messages from memory."""
        self.messages.clear()
        logger.info("Memory cleared")

    def save_session(self, filename: Optional[str] = None) -> None:
        """Save the current session to disk.

        Args:
            filename: Optional filename to save to. Uses default if None.
        """
        try:
            save_file = self.session_dir / filename if filename else self.session_file

            session_data = {
                "timestamp": datetime.now().isoformat(),
                "initial_prompt": self.initial_prompt,
                "messages": [self._message_to_dict(msg) for msg in self.messages],
            }

            with open(save_file, "w", encoding="utf-8") as f:
                json.dump(session_data, f, indent=2, ensure_ascii=False)

            logger.info(f"Session saved to {save_file}")
        except Exception as e:
            logger.error(f"Failed to save session: {e}")

    def load_session(self, filename: Optional[str] = None) -> bool:
        """Load a previous session from disk.

        Args:
            filename: Optional filename to load from. Uses default if None.

        Returns:
            True if session was loaded successfully, False otherwise.
        """
        try:
            load_file = self.session_dir / filename if filename else self.session_file

            if not load_file.exists():
                logger.info("No previous session found")
                return False

            with open(load_file, "r", encoding="utf-8") as f:
                session_data = json.load(f)

            self.initial_prompt = session_data.get("initial_prompt")
            self.messages = [
                self._dict_to_message(msg_dict)
                for msg_dict in session_data.get("messages", [])
            ]

            logger.info(f"Session loaded from {load_file}")
            return True
        except Exception as e:
            logger.error(f"Failed to load session: {e}")
            return False

    def _compress_memory(self) -> None:
        """Compress old memories to save space."""
        if not self.memory_compression or len(self.messages) < 30:
            return

        # Keep system messages and recent messages
        system_messages = [msg for msg in self.messages if msg.role == "system"]
        recent_messages = self.messages[-20:]  # Keep last 20 messages

        # Compress middle messages into a summary
        middle_messages = self.messages[len(system_messages) : -20]
        if middle_messages:
            summary = self._create_summary(middle_messages)
            summary_message = Message.system_message(
                f"Previous conversation summary: {summary}"
            )

            # Rebuild messages list
            self.messages = system_messages + [summary_message] + recent_messages
            logger.info("Memory compressed")

    def _create_summary(self, messages: List[Message]) -> str:
        """Create a summary of messages.

        Args:
            messages: Messages to summarize.

        Returns:
            Summary text.
        """
        # Simple summary for now - could be enhanced with LLM-based summarization
        user_messages = [
            msg.content for msg in messages if msg.role == "user" and msg.content
        ]
        assistant_messages = [
            msg.content for msg in messages if msg.role == "assistant" and msg.content
        ]

        summary_parts = []
        if user_messages:
            summary_parts.append(f"User discussed: {', '.join(user_messages[:3])}")
        if assistant_messages:
            summary_parts.append(
                f"Assistant helped with: {', '.join(assistant_messages[:3])}"
            )

        return (
            "; ".join(summary_parts)
            if summary_parts
            else "Previous conversation occurred"
        )

    def _message_to_dict(self, message: Message) -> Dict[str, Any]:
        """Convert a message to dictionary for serialization.

        Args:
            message: Message to convert.

        Returns:
            Dictionary representation of the message.
        """
        return {
            "role": message.role,
            "content": message.content,
            "base64_image": getattr(message, "base64_image", None),
            "tool_calls": getattr(message, "tool_calls", None),
            "tool_call_id": getattr(message, "tool_call_id", None),
            "timestamp": getattr(message, "timestamp", datetime.now().isoformat()),
        }

    def _dict_to_message(self, msg_dict: Dict[str, Any]) -> Message:
        """Convert a dictionary to a message object.

        Args:
            msg_dict: Dictionary representation of a message.

        Returns:
            Message object.
        """
        role = msg_dict["role"]
        content = msg_dict["content"]

        if role == "user":
            return Message.user_message(
                content, base64_image=msg_dict.get("base64_image")
            )
        elif role == "assistant":
            message = Message.assistant_message(content)
            if "tool_calls" in msg_dict:
                message.tool_calls = msg_dict["tool_calls"]
            return message
        elif role == "system":
            return Message.system_message(content)
        elif role == "tool":
            return Message.tool_message(
                content, tool_call_id=msg_dict.get("tool_call_id")
            )
        else:
            # Fallback to user message
            return Message.user_message(content)

    def __len__(self) -> int:
        """Return the number of messages in memory."""
        return len(self.messages)

    def __bool__(self) -> bool:
        """Return True if memory contains messages."""
        return bool(self.messages)
