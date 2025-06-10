"""
Ollama-based LLM implementation for ParManusAI.
Replaces the llama-cpp-python implementation with Ollama API calls.
"""

import asyncio
import json
import time
from typing import Any, Dict, List, Optional, Union

import httpx
from openai import AsyncOpenAI

from app.config import LLMSettings, config
from app.logger import logger
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
    """Ollama-based LLM implementation using OpenAI-compatible API."""

    def __init__(self, settings: Optional[LLMSettings] = None):
        """Initialize the Ollama LLM with settings."""
        if settings is None:
            settings = config.llm
        if not isinstance(settings, LLMSettings):
            raise TypeError(f"Expected LLMSettings instance, got {type(settings)}")

        self.settings = settings
        self.model = settings.model
        self.base_url = settings.base_url
        self.api_key = settings.api_key
        self.max_tokens = settings.max_tokens
        self.temperature = settings.temperature
        self.token_counter = TokenCounter()

        # Vision model settings
        self.vision_settings = settings.vision
        self.vision_enabled = self.vision_settings and self.vision_settings.enabled

        # Initialize OpenAI client for Ollama
        self.client = AsyncOpenAI(
            base_url=self.base_url,
            api_key=self.api_key,
        )

        # Vision client (if different from main client)
        if self.vision_enabled and self.vision_settings:
            if (self.vision_settings.base_url != self.base_url or 
                self.vision_settings.api_key != self.api_key):
                self.vision_client = AsyncOpenAI(
                    base_url=self.vision_settings.base_url,
                    api_key=self.vision_settings.api_key,
                )
            else:
                self.vision_client = self.client

        logger.info(f"Initialized Ollama LLM with model: {self.model}")
        if self.vision_enabled:
            logger.info(f"Vision enabled with model: {self.vision_settings.model}")

    def _format_messages(self, messages: List[Union[Message, Dict[str, Any]]]) -> List[Dict[str, Any]]:
        """Format messages for OpenAI API."""
        formatted_messages = []
        
        for msg in messages:
            if isinstance(msg, Message):
                msg_dict = msg.to_dict()
            else:
                msg_dict = msg
            
            # Ensure required fields
            if "role" not in msg_dict:
                msg_dict["role"] = "user"
            if "content" not in msg_dict:
                msg_dict["content"] = ""
            
            formatted_messages.append(msg_dict)
        
        return formatted_messages

    async def ask(
        self,
        messages: List[Union[Message, Dict[str, Any]]],
        system_msgs: Optional[List[Union[Message, Dict[str, Any]]]] = None,
        temp: float = None,
        timeout: Optional[int] = None,
        **kwargs,
    ) -> str:
        """Ask the LLM a question and get a response."""
        try:
            # Format messages
            formatted_messages = []
            
            # Add system messages first
            if system_msgs:
                for sys_msg in system_msgs:
                    if isinstance(sys_msg, Message):
                        sys_dict = sys_msg.to_dict()
                    else:
                        sys_dict = sys_msg
                    sys_dict["role"] = "system"
                    formatted_messages.append(sys_dict)
            
            # Add user messages
            formatted_messages.extend(self._format_messages(messages))
            
            # Use provided temperature or default
            temperature = temp if temp is not None else self.temperature
            
            # Make API call
            response = await self.client.chat.completions.create(
                model=self.model,
                messages=formatted_messages,
                max_tokens=self.max_tokens,
                temperature=temperature,
                **kwargs
            )
            
            # Extract response text
            content = response.choices[0].message.content
            
            # Update token counter if usage info is available
            if hasattr(response, 'usage') and response.usage:
                self.token_counter.update(
                    response.usage.prompt_tokens,
                    response.usage.completion_tokens
                )
            
            return content
            
        except Exception as e:
            logger.error(f"Error in Ollama LLM ask: {str(e)}")
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
        """Ask the LLM with tool support."""
        try:
            # Format messages
            formatted_messages = []
            
            # Add system messages first
            if system_msgs:
                for sys_msg in system_msgs:
                    if isinstance(sys_msg, Message):
                        sys_dict = sys_msg.to_dict()
                    else:
                        sys_dict = sys_msg
                    sys_dict["role"] = "system"
                    formatted_messages.append(sys_dict)
            
            # Add user messages
            formatted_messages.extend(self._format_messages(messages))
            
            # Use provided temperature or default
            temperature = temp if temp is not None else self.temperature
            
            # Prepare tool choice
            tool_choice_param = None
            if tool_choice == ToolChoice.AUTO:
                tool_choice_param = "auto"
            elif tool_choice == ToolChoice.REQUIRED:
                tool_choice_param = "required"
            elif tool_choice == ToolChoice.NONE:
                tool_choice_param = "none"
            
            # Make API call with tools
            call_params = {
                "model": self.model,
                "messages": formatted_messages,
                "max_tokens": self.max_tokens,
                "temperature": temperature,
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
            
            # Update token counter if usage info is available
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
            logger.error(f"Error in Ollama LLM ask_tool: {str(e)}")
            raise

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
        if not self.vision_enabled or not self.vision_settings:
            raise ValueError("Vision is not enabled or configured")
        
        try:
            # Format messages
            formatted_messages = []
            
            # Add system messages first
            if system_msgs:
                for sys_msg in system_msgs:
                    if isinstance(sys_msg, Message):
                        sys_dict = sys_msg.to_dict()
                    else:
                        sys_dict = sys_msg
                    sys_dict["role"] = "system"
                    formatted_messages.append(sys_dict)
            
            # Add user messages with images
            for msg in messages:
                if isinstance(msg, Message):
                    msg_dict = msg.to_dict()
                else:
                    msg_dict = msg
                
                # If images are provided, add them to the content
                if images and msg_dict.get("role") == "user":
                    content = []
                    if msg_dict.get("content"):
                        content.append({"type": "text", "text": msg_dict["content"]})
                    
                    for image in images:
                        content.append({
                            "type": "image_url",
                            "image_url": {"url": image}
                        })
                    
                    msg_dict["content"] = content
                
                formatted_messages.append(msg_dict)
            
            # Use provided temperature or vision model default
            temperature = temp if temp is not None else self.vision_settings.temperature
            
            # Make API call to vision model
            response = await self.vision_client.chat.completions.create(
                model=self.vision_settings.model,
                messages=formatted_messages,
                max_tokens=self.vision_settings.max_tokens,
                temperature=temperature,
                **kwargs
            )
            
            # Extract response text
            content = response.choices[0].message.content
            
            # Update token counter if usage info is available
            if hasattr(response, 'usage') and response.usage:
                self.token_counter.update(
                    response.usage.prompt_tokens,
                    response.usage.completion_tokens
                )
            
            return content
            
        except Exception as e:
            logger.error(f"Error in Ollama LLM ask_vision: {str(e)}")
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


# Alias for backward compatibility
LLM = OllamaLLM

