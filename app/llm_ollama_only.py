"""
Simplified LLM implementation using Ollama exclusively.
No more llama-cpp-python dependencies.
"""

import asyncio
import json
import time
from typing import Any, Dict, List, Optional, Union

from pydantic import BaseModel, Field

# Only need OpenAI for Ollama
try:
    from openai import AsyncOpenAI
    OPENAI_AVAILABLE = True
except ImportError:
    OPENAI_AVAILABLE = False

try:
    from loguru import logger
except ImportError:
    import logging
    logger = logging.getLogger(__name__)
    logging.basicConfig(level=logging.INFO)

from app.schema import Message, ToolChoice


class TokenCounter:
    """Token counter for tracking usage."""
    
    def __init__(self):
        self.prompt_tokens = 0
        self.completion_tokens = 0
        self.total_tokens = 0

    def update(self, prompt_tokens: int, completion_tokens: int):
        """Update token counts."""
        self.prompt_tokens += prompt_tokens
        self.completion_tokens += completion_tokens
        self.total_tokens = self.prompt_tokens + self.completion_tokens


class OllamaLLM:
    """Ollama-only LLM implementation with full tool calling support."""
    
    def __init__(self, config):
        if not OPENAI_AVAILABLE:
            raise ImportError("openai package not available. Install with: pip install openai")
        
        self.config = config
        self.token_counter = TokenCounter()
        
        # Model settings
        self.model = config.model
        self.max_tokens = config.max_tokens
        self.temperature = config.temperature
        self.base_url = config.base_url
        self.api_key = config.api_key
        
        # Vision settings (same model)
        self.vision_enabled = getattr(config, 'vision_enabled', True)
        
        # Initialize Ollama client
        self.client = AsyncOpenAI(
            base_url=self.base_url,
            api_key=self.api_key,
        )
        
        logger.info(f"🚀 Initialized Ollama LLM: {self.base_url}")
        logger.info(f"🤖 Model: {self.model}")
        logger.info("✅ Unified model handles both text and vision tasks")
    
    def _format_messages(self, messages: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Format messages for OpenAI API."""
        formatted = []
        for msg in messages:
            if isinstance(msg, str):
                formatted.append({"role": "user", "content": msg})
            elif isinstance(msg, dict):
                formatted.append(msg)
            else:
                formatted.append({"role": "user", "content": str(msg)})
        return formatted
    
    async def ask(self, messages: Union[str, List[Dict[str, Any]]], **kwargs) -> str:
        """Basic ask method."""
        try:
            if isinstance(messages, str):
                messages = [{"role": "user", "content": messages}]
            
            formatted_messages = self._format_messages(messages)
            
            response = await self.client.chat.completions.create(
                model=self.model,
                messages=formatted_messages,
                max_tokens=self.max_tokens,
                temperature=self.temperature,
                **kwargs
            )
            
            content = response.choices[0].message.content
            
            if hasattr(response, 'usage') and response.usage:
                self.token_counter.update(
                    response.usage.prompt_tokens,
                    response.usage.completion_tokens
                )
            
            return content
            
        except Exception as e:
            logger.error(f"Error in Ollama ask: {e}")
            raise
    
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
        """Ask with tool calling support."""
        try:
            # Format messages
            formatted_messages = []
            
            if system_msgs:
                for msg in system_msgs:
                    if isinstance(msg, Message):
                        formatted_messages.append(msg.to_dict())
                    else:
                        formatted_messages.append(msg)
            
            for msg in messages:
                if isinstance(msg, Message):
                    formatted_messages.append(msg.to_dict())
                else:
                    formatted_messages.append(msg)
            
            # Prepare tool choice
            tool_choice_param = None
            if tool_choice == ToolChoice.AUTO:
                tool_choice_param = "auto"
            elif tool_choice == ToolChoice.REQUIRED:
                tool_choice_param = "required"
            elif tool_choice == ToolChoice.NONE:
                tool_choice_param = "none"
            
            # Make API call
            call_params = {
                "model": self.model,
                "messages": formatted_messages,
                "max_tokens": self.max_tokens,
                "temperature": temp if temp is not None else self.temperature,
                **kwargs
            }
            
            if tools:
                call_params["tools"] = tools
                if tool_choice_param:
                    call_params["tool_choice"] = tool_choice_param
            
            response = await self.client.chat.completions.create(**call_params)
            
            # Extract response
            message = response.choices[0].message
            content = message.content or ""
            tool_calls = message.tool_calls or []
            
            # Update token counter
            if hasattr(response, 'usage') and response.usage:
                self.token_counter.update(
                    response.usage.prompt_tokens,
                    response.usage.completion_tokens
                )
            
            # Format tool calls
            formatted_tool_calls = []
            for tool_call in tool_calls:
                formatted_tool_calls.append({
                    "id": tool_call.id,
                    "type": "function",
                    "function": {
                        "name": tool_call.function.name,
                        "arguments": tool_call.function.arguments
                    }
                })
            
            return {
                "content": content,
                "tool_calls": formatted_tool_calls,
                "usage": {
                    "prompt_tokens": getattr(response.usage, 'prompt_tokens', 0) if hasattr(response, 'usage') else 0,
                    "completion_tokens": getattr(response.usage, 'completion_tokens', 0) if hasattr(response, 'usage') else 0,
                    "total_tokens": getattr(response.usage, 'total_tokens', 0) if hasattr(response, 'usage') else 0,
                }
            }
            
        except Exception as e:
            logger.error(f"Error in Ollama ask_tool: {e}")
            raise
    
    async def ask_vision(self, messages: Union[str, List[Dict[str, Any]]], images: List[str] = None, **kwargs) -> str:
        """Ask vision model - uses the same unified model."""
        # For Llama 3.2 Vision, vision is handled by the same model
        return await self.ask(messages, **kwargs)
    
    def get_token_count(self) -> Dict[str, int]:
        """Get current token usage."""
        return {
            "prompt_tokens": self.token_counter.prompt_tokens,
            "completion_tokens": self.token_counter.completion_tokens,
            "total_tokens": self.token_counter.total_tokens,
        }
    
    def reset_token_count(self):
        """Reset token counter."""
        self.token_counter = TokenCounter()


def create_llm_with_tools(config):
    """Factory function to create Ollama LLM (only option now)."""
    api_type = getattr(config, 'api_type', 'ollama').lower()
    
    if api_type != 'ollama':
        logger.warning(f"API type '{api_type}' not supported. Using Ollama.")
    
    if not OPENAI_AVAILABLE:
        logger.error("openai package not available for Ollama")
        raise ImportError("openai package required for Ollama")
    
    return OllamaLLM(config)


# Alias for backward compatibility
LLM = create_llm_with_tools

