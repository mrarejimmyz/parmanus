"""
LLM factory and routing for ParManusAI.
Routes to appropriate LLM implementation based on api_type.
"""

import asyncio
from typing import Any, Dict, List, Optional, Union

from app.config import LLMSettings, config
from app.logger import logger
from app.schema import Message, ToolChoice


def create_llm(settings: Optional[LLMSettings] = None):
    """Factory function to create appropriate LLM instance based on api_type."""
    if settings is None:
        settings = config.llm
    
    api_type = getattr(settings, 'api_type', 'ollama').lower()
    
    if api_type == 'ollama':
        from app.ollama_llm import OllamaLLM
        return OllamaLLM(settings)
    elif api_type in ['openai', 'azure']:
        # Future: implement OpenAI LLM
        raise NotImplementedError(f"API type '{api_type}' not yet implemented")
    elif api_type == 'anthropic':
        # Future: implement Anthropic LLM
        raise NotImplementedError(f"API type '{api_type}' not yet implemented")
    elif api_type == 'local' or api_type == 'llama-cpp':
        # Legacy llama-cpp-python implementation
        from app.llm_legacy import LegacyLLM
        logger.warning("Using legacy llama-cpp-python implementation. Consider migrating to Ollama.")
        return LegacyLLM(settings)
    else:
        raise ValueError(f"Unsupported API type: {api_type}")


class LLM:
    """Main LLM class that routes to appropriate implementation."""
    
    def __init__(self, settings: Optional[LLMSettings] = None):
        """Initialize LLM with automatic routing based on api_type."""
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
    
    async def ask_vision(
        self,
        messages: List[Union[Message, Dict[str, Any]]],
        images: Optional[List[str]] = None,
        system_msgs: Optional[List[Union[Message, Dict[str, Any]]]] = None,
        temp: float = None,
        timeout: Optional[int] = None,
        **kwargs,
    ) -> str:
        """Ask the vision model with image support."""
        if hasattr(self._impl, 'ask_vision'):
            return await self._impl.ask_vision(messages, images, system_msgs, temp, timeout, **kwargs)
        else:
            raise NotImplementedError("Vision support not available for this LLM implementation")
    
    def get_token_count(self) -> Dict[str, int]:
        """Get current token usage."""
        return self._impl.get_token_count()
    
    def reset_token_count(self):
        """Reset token counter."""
        self._impl.reset_token_count()
    
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

