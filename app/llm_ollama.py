"""
Ollama LLM Backend for ParManus
Provides integration with Ollama for local LLM inference
"""

import asyncio
import base64
import json
import time
from io import BytesIO
from typing import Any, Dict, Generator, List, Optional, Union

import httpx
import ollama
from PIL import Image
from pydantic import BaseModel

from app.config import LLMSettings, config
from app.exceptions import TokenLimitExceeded
from app.logger import logger
from app.schema import Message, ToolChoice


class TokenCounter:
    """Track token usage across requests."""

    def __init__(self):
        self.prompt_tokens = 0
        self.completion_tokens = 0
        self.total_tokens = 0

    def update(self, prompt_tokens: int, completion_tokens: int):
        """Update token counts."""
        self.prompt_tokens += prompt_tokens
        self.completion_tokens += completion_tokens
        self.total_tokens = self.prompt_tokens + self.completion_tokens

    def reset(self):
        """Reset all counters."""
        self.prompt_tokens = 0
        self.completion_tokens = 0
        self.total_tokens = 0


class OllamaLLMSettings(BaseModel):
    """Ollama-specific LLM settings."""

    model: str = "llama3.2-vision:11b"
    base_url: str = "http://localhost:11434"
    max_tokens: int = 2048
    temperature: float = 0.0
    timeout: int = 120


class OllamaLLM:
    """
    Ollama LLM class for handling interactions with Ollama-served models.
    Provides vision capabilities and tool calling support.
    """

    def __init__(
        self, settings: Optional[Union[LLMSettings, OllamaLLMSettings]] = None
    ):
        """Initialize the Ollama LLM with settings."""
        if settings is None:
            # Create default Ollama settings
            settings = OllamaLLMSettings()
        elif isinstance(settings, LLMSettings):
            # Convert LLMSettings to OllamaLLMSettings
            settings = OllamaLLMSettings(
                model=getattr(settings, "model", "llama3.2-vision:11b"),
                max_tokens=getattr(settings, "max_tokens", 2048),
                temperature=getattr(settings, "temperature", 0.0),
            )

        self.settings = settings
        self.model = settings.model
        self.max_tokens = settings.max_tokens
        self.temperature = settings.temperature
        self.base_url = settings.base_url
        self.timeout = settings.timeout  # Initialize Ollama client
        self.client = ollama.Client(host=self.base_url)

        # Vision capabilities
        self.vision_enabled = True  # Ollama supports vision models

        # Token counter for compatibility with agents
        self.token_counter = TokenCounter()

        logger.info(f"Initialized Ollama LLM: {self.model} at {self.base_url}")

    def count_tokens(self, text: str) -> int:
        """Estimate token count (approximate)."""
        if not text:
            return 0
        # Simple approximation: 1 token ≈ 4 characters for English text
        return max(1, len(text.encode("utf-8")) // 4)

    def _format_messages_for_ollama(
        self, messages: List[Union[Message, Dict[str, Any]]]
    ) -> List[Dict[str, Any]]:
        """Format messages for Ollama API."""
        formatted_messages = []

        for msg in messages:
            if isinstance(msg, Message):
                msg_dict = {"role": msg.role, "content": msg.content}
            else:
                msg_dict = {
                    "role": msg.get("role", "user"),
                    "content": msg.get("content", ""),
                }

            # Handle image content for vision models
            if isinstance(msg_dict["content"], list):
                # Multi-modal content (text + images)
                text_parts = []
                images = []

                for item in msg_dict["content"]:
                    if isinstance(item, dict):
                        if item.get("type") == "text":
                            text_parts.append(item.get("text", ""))
                        elif item.get("type") == "image_url":
                            # Extract base64 image data
                            image_url = item.get("image_url", {}).get("url", "")
                            if image_url.startswith("data:image"):
                                # Extract base64 data
                                base64_data = (
                                    image_url.split(",", 1)[1]
                                    if "," in image_url
                                    else image_url
                                )
                                images.append(base64_data)

                # Set content as text and add images
                msg_dict["content"] = " ".join(text_parts)
                if images:
                    msg_dict["images"] = images

            formatted_messages.append(msg_dict)

        return formatted_messages

    def _format_tools_for_ollama(
        self, tools: Optional[List[Dict[str, Any]]]
    ) -> Optional[List[Dict[str, Any]]]:
        """Format tools for Ollama (if supported by the model)."""
        if not tools:
            return None

        # Ollama tools format (simplified)
        formatted_tools = []
        for tool in tools:
            if tool.get("type") == "function":
                function = tool.get("function", {})
                formatted_tool = {
                    "type": "function",
                    "function": {
                        "name": function.get("name", ""),
                        "description": function.get("description", ""),
                        "parameters": function.get("parameters", {}),
                    },
                }
                formatted_tools.append(formatted_tool)

        return formatted_tools

    async def ask(
        self,
        messages: List[Union[Message, Dict[str, Any]]],
        system_msgs: Optional[List[Union[Message, Dict[str, Any]]]] = None,
        temperature: Optional[float] = None,
        stream: bool = False,
        timeout: int = 120,
        **kwargs,
    ) -> Union[str, Generator[str, None, None]]:
        """
        Send a request to the Ollama model and get a response.

        Args:
            messages: List of messages to send to the model
            system_msgs: Optional system messages to prepend
            temperature: Temperature for sampling
            stream: Whether to stream the response
            timeout: Timeout in seconds
            **kwargs: Additional arguments

        Returns:
            The model's response as a string or a generator if streaming
        """
        try:
            # Combine system messages with regular messages
            all_messages = []
            if system_msgs:
                all_messages.extend(system_msgs)
            all_messages.extend(messages)

            # Format messages for Ollama
            formatted_messages = self._format_messages_for_ollama(all_messages)

            # Set temperature
            temp = temperature if temperature is not None else self.temperature

            # Prepare request parameters
            request_params = {
                "model": self.model,
                "messages": formatted_messages,
                "options": {
                    "temperature": temp,
                    "num_predict": self.max_tokens,
                },
                "stream": stream,
            }

            if stream:
                return self._generate_stream_response(request_params, timeout)
            else:
                return await self._generate_single_response(request_params, timeout)

        except Exception as e:
            logger.error(f"Error in Ollama ask: {e}")
            raise

    def _generate_stream_response(
        self, request_params: Dict[str, Any], timeout: int
    ) -> Generator[str, None, None]:
        """Generate streaming response from Ollama."""
        try:
            for chunk in self.client.chat(**request_params):
                if chunk.get("done", False):
                    break
                content = chunk.get("message", {}).get("content", "")
                if content:
                    yield content
        except Exception as e:
            logger.error(f"Error in streaming response: {e}")
            yield f"[Error: {str(e)}]"

    async def _generate_single_response(
        self, request_params: Dict[str, Any], timeout: int
    ) -> str:
        """Generate single response from Ollama."""
        try:
            # Run in thread pool to avoid blocking
            loop = asyncio.get_event_loop()
            response = await asyncio.wait_for(
                loop.run_in_executor(None, lambda: self.client.chat(**request_params)),
                timeout=timeout,
            )

            return response.get("message", {}).get("content", "")

        except asyncio.TimeoutError:
            logger.error(f"Ollama request timed out after {timeout} seconds")
            return f"[Response incomplete due to timeout after {timeout} seconds]"
        except Exception as e:
            logger.error(f"Error in single response: {e}")
            raise

    async def ask_tool(
        self,
        messages: List[Union[Message, Dict[str, Any]]],
        tools: Optional[List[Dict[str, Any]]] = None,
        tool_choice: Union[str, ToolChoice] = ToolChoice.AUTO,
        temp: float = 0.1,
        max_retries: int = 2,
        timeout: Optional[int] = None,
        **kwargs,
    ) -> Dict[str, Any]:
        """
        Ask the model with tool calling capability.

        Args:
            messages: List of messages
            tools: Available tools
            tool_choice: Tool choice strategy
            temp: Temperature
            max_retries: Maximum retry attempts
            timeout: Request timeout
            **kwargs: Additional arguments

        Returns:
            Response dict with content and tool_calls
        """
        start_time = time.time()

        if timeout is None:
            timeout = self.timeout

        try:
            # Format messages
            formatted_messages = self._format_messages_for_ollama(messages)

            # Add tool instructions to the last user message if tools are provided
            if tools:
                tool_descriptions = self._format_tool_descriptions(tools)

                # Find the last user message and append tool instructions
                for i in range(len(formatted_messages) - 1, -1, -1):
                    if formatted_messages[i]["role"] == "user":
                        formatted_messages[i]["content"] += f"\n\n{tool_descriptions}"
                        break

            # Make request to Ollama
            request_params = {
                "model": self.model,
                "messages": formatted_messages,
                "options": {
                    "temperature": temp,
                    "num_predict": self.max_tokens,
                },
                "stream": False,
            }

            response = await self._generate_single_response(
                request_params, timeout
            )  # Parse tool calls from response
            tool_calls = (
                self._parse_tool_calls(response) if tools else []
            )  # Calculate token usage
            prompt_tokens = sum(
                self.count_tokens(
                    getattr(msg, "content", "")
                    if hasattr(msg, "content")
                    else msg.get("content", "")
                )
                for msg in formatted_messages
            )
            completion_tokens = self.count_tokens(response)
            total_tokens = prompt_tokens + completion_tokens

            # Update token counter
            self.token_counter.update(prompt_tokens, completion_tokens)

            result = {
                "content": response,
                "tool_calls": tool_calls,
                "usage": {
                    "prompt_tokens": prompt_tokens,
                    "completion_tokens": completion_tokens,
                    "total_tokens": total_tokens,
                },
                "elapsed_time": time.time() - start_time,
                "attempts": 1,
            }

            return result

        except Exception as e:
            logger.error(f"Error in ask_tool: {e}")
            return {
                "content": "",
                "tool_calls": [],
                "error": str(e),
                "elapsed_time": time.time() - start_time,
                "attempts": 1,
            }

    def _format_tool_descriptions(self, tools: List[Dict[str, Any]]) -> str:
        """Format tools as text descriptions for the model."""
        if not tools:
            return ""

        tool_descriptions = ["Available tools:"]
        for tool in tools:
            if tool.get("type") == "function":
                func = tool.get("function", {})
                name = func.get("name", "")
                description = func.get("description", "")
                parameters = func.get("parameters", {})

                tool_desc = f"\n- {name}: {description}"
                if parameters and parameters.get("properties"):
                    props = parameters["properties"]
                    required = parameters.get("required", [])

                    params_desc = []
                    for param_name, param_info in props.items():
                        param_type = param_info.get("type", "string")
                        param_desc = param_info.get("description", "")
                        is_required = param_name in required

                        param_str = f"{param_name} ({param_type})"
                        if is_required:
                            param_str += " [required]"
                        if param_desc:
                            param_str += f": {param_desc}"

                        params_desc.append(param_str)

                    if params_desc:
                        tool_desc += f"\n  Parameters: {', '.join(params_desc)}"

                tool_descriptions.append(tool_desc)

        tool_descriptions.append(
            "\nTo use a tool, respond with a function call in this format:\n"
            'tool_name(param1="value1", param2="value2")\n'
            "You can call multiple tools by listing them on separate lines."
        )

        return "\n".join(tool_descriptions)

    def _parse_tool_calls(self, content: str) -> List[Dict[str, Any]]:
        """Parse tool calls from model output."""
        tool_calls = []

        if not content:
            return tool_calls

        # Look for function call patterns like: function_name(arg1="value1", arg2="value2")
        import re

        # Pattern to match function calls
        pattern = r"(\w+)\s*\(\s*([^)]*)\s*\)"
        matches = re.finditer(pattern, content)

        for i, match in enumerate(matches):
            func_name = match.group(1)
            args_str = match.group(2)

            # Skip if this looks like regular text, not a tool call
            if func_name.lower() in [
                "if",
                "for",
                "while",
                "def",
                "class",
                "print",
                "len",
                "str",
                "int",
                "float",
                "bool",
                "list",
                "dict",
                "set",
                "tuple",
            ]:
                continue

            # Parse arguments
            args = {}
            if args_str.strip():
                # Simple argument parsing (could be improved)
                try:
                    # Handle the case where it's malformed like 'import requests; import json; response = requests.get('
                    if (
                        "=" in args_str
                        and not args_str.endswith('"')
                        and not args_str.endswith("'")
                    ):
                        # This looks like incomplete code, skip it
                        logger.warning(
                            f"Skipping malformed tool call: {func_name}({args_str})"
                        )
                        continue

                    # Try to parse as key=value pairs
                    for arg_pair in args_str.split(","):
                        if "=" in arg_pair:
                            key, value = arg_pair.split("=", 1)
                            key = key.strip().strip("\"'")
                            value = value.strip().strip("\"'")
                            args[key] = value
                except:
                    # If parsing fails, treat the whole string as a single argument
                    if len(args_str.strip()) > 100:  # Skip very long malformed strings
                        logger.warning(
                            f"Skipping overly long malformed argument: {args_str[:50]}..."
                        )
                        continue
                    args = {"input": args_str.strip()}

            tool_call = {
                "id": f"call_{i}",
                "type": "function",
                "function": {"name": func_name, "arguments": json.dumps(args)},
            }
            tool_calls.append(tool_call)

        return tool_calls

    def check_model_availability(self) -> bool:
        """Check if the specified model is available in Ollama."""
        try:
            models = self.client.list()
            # Handle the response format
            if hasattr(models, "models"):
                available_models = [
                    model.model for model in models.models
                ]  # Use .model instead of .name
            elif isinstance(models, dict) and "models" in models:
                available_models = [
                    model.get("model", "") for model in models["models"]
                ]  # Use "model" instead of "name"
            else:
                available_models = []

            logger.debug(f"Available models: {available_models}")
            logger.debug(f"Looking for model: {self.model}")

            return self.model in available_models
        except Exception as e:
            logger.error(f"Error checking model availability: {e}")
            return False

            logger.debug(f"Available models: {available_models}")
            logger.debug(f"Looking for model: {self.model}")

            return self.model in available_models
        except Exception as e:
            logger.error(f"Error checking model availability: {e}")
            return False

    async def pull_model_if_needed(self) -> bool:
        """Pull the model if it's not available locally."""
        if self.check_model_availability():
            logger.info(f"Model {self.model} is already available")
            return True

        try:
            logger.info(f"Pulling model {self.model} from Ollama...")
            # Run pull in thread pool to avoid blocking
            loop = asyncio.get_event_loop()
            await loop.run_in_executor(None, lambda: self.client.pull(self.model))
            logger.info(f"Successfully pulled model {self.model}")
            return True
        except Exception as e:
            logger.error(f"Error pulling model {self.model}: {e}")
            return False
