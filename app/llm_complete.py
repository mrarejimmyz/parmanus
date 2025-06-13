"""
Complete LLM implementation with tool calling support for local GGUF models.
Integrates with the existing ParManusAI tool ecosystem.
"""

import asyncio
import json
import os
import re
import time
from concurrent.futures import ThreadPoolExecutor
from typing import Any, Dict, List, Optional, Union

from pydantic import BaseModel, Field

# Conditional imports
try:
    from llama_cpp import Llama
    LLAMA_CPP_AVAILABLE = True
except ImportError:
    LLAMA_CPP_AVAILABLE = False

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


class LocalLLMWithTools:
    """Local GGUF model implementation with comprehensive tool calling support."""
    
    # Context window sizes
    TEXT_MODEL_CONTEXT_SIZE = 8192
    VISION_MODEL_CONTEXT_SIZE = 4096
    MAX_ALLOWED_OUTPUT_TOKENS = 2048
    
    # Thread pool for model operations
    _executor = ThreadPoolExecutor(max_workers=2)
    
    def __init__(self, config):
        if not LLAMA_CPP_AVAILABLE:
            raise ImportError("llama-cpp-python not available. Install with: pip install llama-cpp-python")
        
        self.config = config
        self.token_counter = TokenCounter()
        
        # Model settings
        self.model = config.model
        self.model_path = config.model_path
        self.max_tokens = min(config.max_tokens, self.MAX_ALLOWED_OUTPUT_TOKENS)
        self.temperature = config.temperature
        self.n_gpu_layers = config.n_gpu_layers
        
        # Vision settings (using same model)
        self.vision_enabled = config.vision_enabled
        self.vision_model_path = getattr(config, 'model_path', None)  # Same as main model
        self.vision_clip_path = None  # Not needed for unified model
        
        # Initialize models
        self._load_models()
        
        logger.info(f"Initialized Local LLM with unified model: {self.model}")
        logger.info("✅ Single model handles both text and vision tasks")
    
    def _load_models(self):
        """Load the unified Llama 3.2 Vision model."""
        # Load main text/vision model (unified)
        if os.path.exists(self.model_path):
            logger.info(f"Loading unified Llama 3.2 Vision model: {self.model_path}")
            self.text_model = Llama(
                model_path=self.model_path,
                n_gpu_layers=self.n_gpu_layers,
                n_ctx=self.TEXT_MODEL_CONTEXT_SIZE,
                verbose=False
            )
            # For Llama 3.2 Vision, the same model handles both text and vision
            self.vision_model = self.text_model
            logger.info("✅ Unified model loaded - handles both text and vision")
        else:
            raise FileNotFoundError(f"Model not found: {self.model_path}")
        
        # Note: For Llama 3.2 Vision, we don't need separate vision model loading
        # The unified model handles both text and vision tasks
    
    def _format_prompt_for_llama(self, messages: List[Dict[str, Any]]) -> str:
        """Format messages for Llama prompt format."""
        prompt = ""
        for msg in messages:
            role = msg.get('role', 'user')
            content = msg.get('content', '')
            
            if role == 'system':
                prompt += f"<|system|>\n{content}\n"
            elif role == 'user':
                prompt += f"<|user|>\n{content}\n"
            elif role == 'assistant':
                prompt += f"<|assistant|>\n{content}\n"
            elif role == 'tool':
                # Format tool results
                tool_name = msg.get('name', 'tool')
                prompt += f"<|tool|>\nTool '{tool_name}' result: {content}\n"
        
        prompt += "<|assistant|>\n"
        return prompt
    
    def _format_tools_for_prompt(self, tools: Optional[List[Dict[str, Any]]]) -> str:
        """Format tools for inclusion in prompt."""
        if not tools:
            return ""
        
        tools_text = "\n\nAvailable tools:\n"
        for tool in tools:
            if "function" in tool:
                func_data = tool["function"]
                name = func_data.get("name", "unknown")
                description = func_data.get("description", "")
                parameters = func_data.get("parameters", {})
            else:
                name = tool.get("name", "unknown")
                description = tool.get("description", "")
                parameters = tool.get("parameters", {})
            
            tools_text += f"\n- {name}: {description}"
            if parameters and "properties" in parameters:
                tools_text += "\n  Parameters:"
                for param_name, param_info in parameters.get("properties", {}).items():
                    required = param_name in parameters.get("required", [])
                    param_desc = param_info.get("description", "")
                    tools_text += f"\n    - {param_name} ({'required' if required else 'optional'}): {param_desc}"
        
        tools_text += "\n\nTo use a tool, respond with JSON in this format:"
        tools_text += '\n{"tool_calls": [{"name": "tool_name", "arguments": {"param": "value"}}]}'
        tools_text += "\n\nYou can also provide regular text response along with tool calls."
        
        return tools_text
    
    def _get_tool_instructions(self, tool_choice: Union[str, ToolChoice]) -> str:
        """Get tool usage instructions based on choice."""
        if tool_choice == ToolChoice.AUTO:
            return "\n\nYou can use tools if needed to complete the task, but only if necessary."
        elif tool_choice == ToolChoice.REQUIRED:
            return "\n\nYou MUST use one of the available tools to complete this task."
        elif tool_choice == ToolChoice.NONE:
            return "\n\nDo not use any tools for this task. Respond with text only."
        return ""
    
    def _parse_tool_calls_from_response(self, text: str) -> List[Dict[str, Any]]:
        """Parse tool calls from model response with multiple strategies."""
        tool_calls = []
        
        try:
            # Strategy 1: Look for JSON blocks with tool_calls
            json_pattern = r'```(?:json)?\s*(\{.*?"tool_calls".*?\})\s*```'
            json_matches = re.finditer(json_pattern, text, re.DOTALL | re.IGNORECASE)
            
            for match in json_matches:
                try:
                    json_content = match.group(1)
                    data = json.loads(json_content)
                    if "tool_calls" in data:
                        for call in data["tool_calls"]:
                            formatted_call = self._format_tool_call(call)
                            if formatted_call:
                                tool_calls.append(formatted_call)
                except json.JSONDecodeError:
                    continue
            
            if tool_calls:
                return tool_calls
            
            # Strategy 2: Look for direct JSON with tool_calls
            try:
                # Try to parse the entire response as JSON
                data = json.loads(text.strip())
                if "tool_calls" in data:
                    for call in data["tool_calls"]:
                        formatted_call = self._format_tool_call(call)
                        if formatted_call:
                            tool_calls.append(formatted_call)
                    return tool_calls
            except json.JSONDecodeError:
                pass
            
            # Strategy 3: Look for individual tool call patterns
            tool_call_pattern = r'\{"name":\s*"([^"]+)",\s*"arguments":\s*(\{[^}]*\})\}'
            tool_matches = re.finditer(tool_call_pattern, text)
            
            for match in tool_matches:
                try:
                    name = match.group(1)
                    args_str = match.group(2)
                    args = json.loads(args_str)
                    
                    tool_call = {
                        "id": f"call_{time.time_ns()}",
                        "type": "function",
                        "function": {
                            "name": name,
                            "arguments": json.dumps(args)
                        }
                    }
                    tool_calls.append(tool_call)
                except (json.JSONDecodeError, IndexError):
                    continue
            
            # Strategy 4: Look for natural language tool calls
            natural_pattern = r'(?:use|call|execute)\s+(\w+)\s+(?:with|using)?\s*\{([^}]+)\}'
            natural_matches = re.finditer(natural_pattern, text, re.IGNORECASE)
            
            for match in natural_matches:
                try:
                    name = match.group(1)
                    args_text = match.group(2)
                    
                    # Parse simple key=value arguments
                    args = {}
                    for pair in args_text.split(','):
                        if '=' in pair:
                            key, value = pair.split('=', 1)
                            args[key.strip()] = value.strip().strip('"\'')
                    
                    tool_call = {
                        "id": f"call_{time.time_ns()}",
                        "type": "function",
                        "function": {
                            "name": name,
                            "arguments": json.dumps(args)
                        }
                    }
                    tool_calls.append(tool_call)
                except Exception:
                    continue
            
            return tool_calls
            
        except Exception as e:
            logger.error(f"Error parsing tool calls: {e}")
            return []
    
    def _format_tool_call(self, call_data: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Format a tool call into standard structure."""
        try:
            if "name" in call_data:
                name = call_data["name"]
                args = call_data.get("arguments", {})
            elif "function" in call_data:
                func_data = call_data["function"]
                name = func_data.get("name")
                args = func_data.get("arguments", {})
            else:
                return None
            
            if not name:
                return None
            
            # Ensure arguments is a JSON string
            if isinstance(args, dict):
                args_str = json.dumps(args)
            elif isinstance(args, str):
                args_str = args
            else:
                args_str = json.dumps({})
            
            return {
                "id": call_data.get("id", f"call_{time.time_ns()}"),
                "type": "function",
                "function": {
                    "name": name,
                    "arguments": args_str
                }
            }
        except Exception as e:
            logger.error(f"Error formatting tool call: {e}")
            return None
    
    def _extract_content_from_response(self, text: str) -> str:
        """Extract regular content from response, excluding tool calls."""
        # Remove JSON blocks that contain tool_calls
        json_pattern = r'```(?:json)?\s*\{.*?"tool_calls".*?\}\s*```'
        text = re.sub(json_pattern, '', text, flags=re.DOTALL | re.IGNORECASE)
        
        # Remove standalone JSON with tool_calls
        try:
            data = json.loads(text.strip())
            if "tool_calls" in data:
                return data.get("content", "")
        except json.JSONDecodeError:
            pass
        
        # Remove tool call patterns
        tool_call_pattern = r'\{"name":\s*"[^"]+",\s*"arguments":\s*\{[^}]*\}\}'
        text = re.sub(tool_call_pattern, '', text)
        
        return text.strip()
    
    async def ask(self, messages: Union[str, List[Dict[str, Any]]], **kwargs) -> str:
        """Basic ask method without tool support."""
        try:
            if isinstance(messages, str):
                messages = [{"role": "user", "content": messages}]
            
            prompt = self._format_prompt_for_llama(messages)
            
            # Run in executor to avoid blocking
            loop = asyncio.get_event_loop()
            response = await loop.run_in_executor(
                self._executor,
                lambda: self.text_model(
                    prompt,
                    max_tokens=self.max_tokens,
                    temperature=self.temperature,
                    stop=["<|user|>", "<|system|>", "<|eot_id|>"],
                    **kwargs
                )
            )
            
            content = response['choices'][0]['text'].strip()
            
            # Update token counter
            usage = response.get('usage', {})
            self.token_counter.update(
                usage.get('prompt_tokens', 0),
                usage.get('completion_tokens', 0)
            )
            
            return content
            
        except Exception as e:
            logger.error(f"Error in local LLM ask: {e}")
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
            
            # Add system messages
            if system_msgs:
                for msg in system_msgs:
                    if isinstance(msg, Message):
                        formatted_messages.append(msg.to_dict())
                    else:
                        formatted_messages.append(msg)
            
            # Add regular messages
            for msg in messages:
                if isinstance(msg, Message):
                    formatted_messages.append(msg.to_dict())
                else:
                    formatted_messages.append(msg)
            
            # Create enhanced prompt with tools
            base_prompt = self._format_prompt_for_llama(formatted_messages)
            tools_text = self._format_tools_for_prompt(tools) if tools else ""
            tool_instructions = self._get_tool_instructions(tool_choice)
            
            enhanced_prompt = f"{base_prompt}{tools_text}{tool_instructions}"
            
            # Use provided temperature or default
            temperature = temp if temp is not None else self.temperature
            
            # Run model
            loop = asyncio.get_event_loop()
            response = await loop.run_in_executor(
                self._executor,
                lambda: self.text_model(
                    enhanced_prompt,
                    max_tokens=self.max_tokens,
                    temperature=temperature,
                    stop=["<|user|>", "<|system|>", "<|eot_id|>"],
                    **kwargs
                )
            )
            
            completion_text = response['choices'][0]['text'].strip()
            
            # Parse tool calls and content
            tool_calls = self._parse_tool_calls_from_response(completion_text)
            content = self._extract_content_from_response(completion_text)
            
            # Update token counter
            usage = response.get('usage', {})
            self.token_counter.update(
                usage.get('prompt_tokens', 0),
                usage.get('completion_tokens', 0)
            )
            
            return {
                "content": content,
                "tool_calls": tool_calls,
                "usage": {
                    "prompt_tokens": usage.get('prompt_tokens', 0),
                    "completion_tokens": usage.get('completion_tokens', 0),
                    "total_tokens": usage.get('total_tokens', 0),
                }
            }
            
        except Exception as e:
            logger.error(f"Error in local LLM ask_tool: {e}")
            raise
    
    async def ask_vision(
        self,
        messages: Union[str, List[Dict[str, Any]]],
        images: Optional[List[str]] = None,
        **kwargs
    ) -> str:
        """Ask vision model with image support - uses the same unified model."""
        if not self.vision_enabled or not self.vision_model:
            raise ValueError("Vision model not available")
        
        try:
            if isinstance(messages, str):
                prompt = messages
            else:
                prompt = self._format_prompt_for_llama(messages)
            
            # For Llama 3.2 Vision, image handling is built into the model
            # The model can process both text and images in the same prompt
            
            loop = asyncio.get_event_loop()
            response = await loop.run_in_executor(
                self._executor,
                lambda: self.vision_model(
                    prompt,
                    max_tokens=self.max_tokens,
                    temperature=self.temperature,
                    **kwargs
                )
            )
            
            content = response['choices'][0]['text'].strip()
            
            # Update token counter
            usage = response.get('usage', {})
            self.token_counter.update(
                usage.get('prompt_tokens', 0),
                usage.get('completion_tokens', 0)
            )
            
            return content
            
        except Exception as e:
            logger.error(f"Error in unified vision model: {e}")
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


class OllamaLLMWithTools:
    """Ollama implementation with tool calling support."""
    
    def __init__(self, config):
        if not OPENAI_AVAILABLE:
            raise ImportError("openai package not available. Install with: pip install openai")
        
        self.config = config
        self.token_counter = TokenCounter()
        
        self.client = AsyncOpenAI(
            base_url=config.base_url,
            api_key=config.api_key,
        )
        
        logger.info(f"Initialized Ollama LLM: {config.base_url}")
    
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
                model=self.config.model,
                messages=formatted_messages,
                max_tokens=self.config.max_tokens,
                temperature=self.config.temperature,
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
                "model": self.config.model,
                "messages": formatted_messages,
                "max_tokens": self.config.max_tokens,
                "temperature": temp if temp is not None else self.config.temperature,
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
                        # Object format
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
                        formatted_tool_calls.append({
                            "id": tool_call.get('id', f"call_{len(formatted_tool_calls)}"),
                            "type": "function",
                            "function": {
                                "name": tool_call.get('function', {}).get('name', ''),
                                "arguments": tool_call.get('function', {}).get('arguments', '{}')
                            }
                        })
                    else:
                        # Handle malformed tool call
                        logger.warning(f"Malformed tool call: {tool_call}")
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
            logger.error(f"Error in Ollama ask_tool: {e}")
            raise
    
    async def ask_vision(self, messages: Union[str, List[Dict[str, Any]]], images: List[str] = None, **kwargs) -> str:
        """Ask vision model."""
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
    """Factory function to create appropriate LLM with tool support."""
    api_type = getattr(config, 'api_type', 'local').lower()
    
    if api_type == 'local':
        if not LLAMA_CPP_AVAILABLE:
            logger.error("llama-cpp-python not available for local models")
            raise ImportError("llama-cpp-python required for local models")
        return LocalLLMWithTools(config)
    elif api_type == 'ollama':
        if not OPENAI_AVAILABLE:
            logger.error("openai package not available for Ollama")
            raise ImportError("openai package required for Ollama")
        return OllamaLLMWithTools(config)
    else:
        raise ValueError(f"Unsupported API type: {api_type}")


# Alias for backward compatibility
LLM = create_llm_with_tools

