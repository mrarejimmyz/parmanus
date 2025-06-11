"""
LLM factory and routing for ParManusAI.
Routes to appropriate LLM implementation based on backend configuration.
"""

import asyncio
from typing import Any, Dict, List, Optional, Union

from app.config import LLMSettings, config
from app.llm_factory import create_llm, create_llm_async
from app.logger import logger
from app.schema import Message, ToolChoice


class LLM:
    """Main LLM class that routes to appropriate implementation based on backend."""

    def __init__(self, settings: Optional[LLMSettings] = None):
        """Initialize LLM with automatic routing based on backend configuration."""
        self.settings = settings or config.llm
        self._impl = create_llm(self.settings)

        # Expose common properties
        self.model = self._impl.model
        self.max_tokens = self._impl.max_tokens
        self.temperature = self._impl.temperature
        self.token_counter = self._impl.token_counter

        # Vision support
        self.vision_enabled = getattr(self._impl, 'vision_enabled', False)
        self.vision_settings = getattr(self._impl, 'vision_settings', None)

    async def ask(
        self,
        messages: List[Union[Message, Dict[str, Any]]],
        system_msgs: Optional[List[Union[Message, Dict[str, Any]]]] = None,
        temp: float = None,
        timeout: Optional[int] = None,
        **kwargs,
    ) -> str:
        """Ask the LLM a question and get a response."""
        return await self._impl.ask(messages, system_msgs, temp, timeout, **kwargs)

    async def ask_tool(
        self,
        messages: List[Union[Message, Dict[str, Any]]],
        system_msgs: Optional[List[Union[Message, Dict[str, Any]]]] = None,
        tools: Optional[List[Dict[str, Any]]] = None,
        tool_choice: Union[str, ToolChoice] = ToolChoice.AUTO,
        temp: float = None,
        timeout: Optional[int] = None,
        max_retries: int = 2,
        **kwargs,
    ) -> Dict[str, Any]:
        """Ask the LLM with tool support."""
        return await self._impl.ask_tool(
            messages, system_msgs, tools, tool_choice, temp, timeout, max_retries, **kwargs
        )

    def count_tokens(self, text: str) -> int:
        """Count tokens in text."""
        return self._impl.count_tokens(text)

    def cleanup_models(self):
        """Clean up models and free resources."""
        if hasattr(self._impl, 'cleanup_models'):
            self._impl.cleanup_models()
        elif hasattr(self._impl, 'cleanup'):
            if asyncio.iscoroutinefunction(self._impl.cleanup):
                # Handle async cleanup
                loop = asyncio.get_event_loop()
                if loop.is_running():
                    # Create a task if loop is running
                    loop.create_task(self._impl.cleanup())
                else:
                    # Run directly if no loop is running
                    asyncio.run(self._impl.cleanup())
            else:
                self._impl.cleanup()

    # Legacy compatibility methods
    def _format_prompt_for_llama(self, messages: List[Dict[str, Any]]) -> str:
        """Legacy method for backward compatibility."""
        if hasattr(self._impl, '_format_prompt_for_llama'):
            return self._impl._format_prompt_for_llama(messages)
        else:
            # Simple fallback formatting
            prompt = ""
            for msg in messages:
                role = msg.get('role', 'user')
                content = msg.get('content', '')
                prompt += f"<|{role}|>\n{content}\n"
            return prompt

    @property
    def text_model(self):
        """Legacy property for backward compatibility."""
        return getattr(self._impl, 'text_model', None)

    @property
    def vision_model(self):
        """Legacy property for backward compatibility."""
        return getattr(self._impl, 'vision_model', None)


# Factory function for creating LLM instances
def create_llm_instance(settings: Optional[LLMSettings] = None):
    """Factory function to create appropriate LLM instance based on backend."""
    return LLM(settings)


# For backward compatibility - provide async creation function
async def create_llm_instance_async(settings: Optional[LLMSettings] = None):
    """Async factory function to create appropriate LLM instance."""
    settings = settings or config.llm
    impl = await create_llm_async(settings)

    # Create a LLM wrapper but with the async-created implementation
    llm = LLM.__new__(LLM)  # Create instance without calling __init__
    llm.settings = settings
    llm._impl = impl
    llm.model = impl.model
    llm.max_tokens = impl.max_tokens
    llm.temperature = impl.temperature
    llm.token_counter = impl.token_counter
    llm.vision_enabled = getattr(impl, 'vision_enabled', False)
    llm.vision_settings = getattr(impl, 'vision_settings', None)

    return llm
