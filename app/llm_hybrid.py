"""
Hybrid LLM implementation using both llama3.2 (tools) and llama3.2-vision (vision).
This gives us the best of both worlds - full tool capabilities AND vision support.
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


class HybridOllamaLLM:
    """Hybrid Ollama LLM using llama3.2 for tools and llama3.2-vision for vision."""
    
    def __init__(self, config):
        if not OPENAI_AVAILABLE:
            raise ImportError("openai package not available. Install with: pip install openai")
        
        self.config = config
        self.token_counter = TokenCounter()
        
        # Model settings
        self.tools_model = "llama3.2"  # For tool calling
        self.vision_model = "llama3.2-vision"  # For vision tasks
        self.max_tokens = config.max_tokens
        self.temperature = config.temperature
        self.base_url = config.base_url
        self.api_key = config.api_key
        
        # Vision settings
        self.vision_enabled = getattr(config, 'vision_enabled', True)
        
        # Initialize Ollama client
        self.client = AsyncOpenAI(
            base_url=self.base_url,
            api_key=self.api_key,
        )
        
        logger.info(f"🚀 Initialized Hybrid Ollama LLM: {self.base_url}")
        logger.info(f"🛠️ Tools Model: {self.tools_model}")
        logger.info(f"👁️ Vision Model: {self.vision_model}")
        logger.info("✅ Hybrid system: tools + vision capabilities")
    
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
    
    def _has_vision_content(self, messages: List[Dict[str, Any]]) -> bool:
        """Check if messages contain vision content (images)."""
        for msg in messages:
            if isinstance(msg, dict):
                content = msg.get('content', '')
                if isinstance(content, list):
                    # Check for image content in message
                    for item in content:
                        if isinstance(item, dict) and item.get('type') == 'image_url':
                            return True
                elif 'image' in str(content).lower() or 'screenshot' in str(content).lower():
                    return True
        return False
    
    def _should_use_vision_model(self, messages: List[Dict[str, Any]], tools: Optional[List[Dict[str, Any]]] = None) -> bool:
        """Determine if we should use the vision model."""
        # If tools are requested, always use tools model
        if tools:
            return False
        
        # Check for vision-related content
        return self._has_vision_content(messages)
    
    async def ask(self, messages: Union[str, List[Dict[str, Any]]], **kwargs) -> str:
        """Basic ask method - routes to appropriate model."""
        try:
            if isinstance(messages, str):
                messages = [{"role": "user", "content": messages}]
            
            formatted_messages = self._format_messages(messages)
            
            # Determine which model to use
            use_vision = self._should_use_vision_model(formatted_messages)
            model = self.vision_model if use_vision else self.tools_model
            
            logger.info(f"🤖 Using model: {model} ({'vision' if use_vision else 'tools'})")
            
            response = await self.client.chat.completions.create(
                model=model,
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
            logger.error(f"Error in Hybrid Ollama ask: {e}")
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
        """Ask with tool calling support - always uses tools model."""
        try:
            # Always use tools model for tool calling
            model = self.tools_model
            logger.info(f"🛠️ Using tools model: {model}")
            
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
                "model": model,
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
                # Handle both object and dict formats with proper error handling
                try:
                    if hasattr(tool_call, 'id'):
                        # OpenAI object format
                        formatted_tool_calls.append({
                            "id": tool_call.id,
                            "type": "function",
                            "function": {
                                "name": tool_call.function.name,
                                "arguments": tool_call.function.arguments
                            }
                        })
                    elif isinstance(tool_call, dict):
                        # Dict format - ensure proper structure
                        if 'function' in tool_call:
                            formatted_tool_calls.append({
                                "id": tool_call.get('id', f"call_{len(formatted_tool_calls)}"),
                                "type": "function",
                                "function": {
                                    "name": tool_call['function'].get('name', ''),
                                    "arguments": tool_call['function'].get('arguments', '{}')
                                }
                            })
                        else:
                            # Handle malformed tool call
                            logger.warning(f"Malformed tool call: {tool_call}")
                            continue
                    else:
                        # Handle unexpected format
                        logger.warning(f"Unexpected tool call format: {tool_call}")
                        continue
                except AttributeError as e:
                    logger.error(f"Error processing tool call {tool_call}: {e}")
                    # Create a fallback tool call with generated ID
                    formatted_tool_calls.append({
                        "id": f"call_{len(formatted_tool_calls)}",
                        "type": "function",
                        "function": {
                            "name": str(tool_call),
                            "arguments": "{}"
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
            logger.error(f"Error in Hybrid Ollama ask_tool: {e}")
            logger.error(f"Tool calls format: {tool_calls if 'tool_calls' in locals() else 'Not available'}")
            logger.error(f"Response format: {type(response) if 'response' in locals() else 'Not available'}")
            raise
    
    async def ask_vision(self, messages: Union[str, List[Dict[str, Any]]], images: List[str] = None, **kwargs) -> str:
        """Ask vision model - always uses vision model."""
        try:
            if isinstance(messages, str):
                messages = [{"role": "user", "content": messages}]
            
            formatted_messages = self._format_messages(messages)
            
            # Always use vision model for vision tasks
            model = self.vision_model
            logger.info(f"👁️ Using vision model: {model}")
            
            response = await self.client.chat.completions.create(
                model=model,
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
            logger.error(f"Error in Hybrid vision model: {e}")
            raise
    
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
    """Factory function to create Hybrid Ollama LLM."""
    api_type = getattr(config, 'api_type', 'ollama').lower()
    
    if api_type != 'ollama':
        logger.warning(f"API type '{api_type}' not supported. Using Ollama.")
    
    if not OPENAI_AVAILABLE:
        logger.error("openai package not available for Ollama")
        raise ImportError("openai package required for Ollama")
    
    return HybridOllamaLLM(config)


# Alias for backward compatibility
LLM = create_llm_with_tools

