import math
import os
from typing import Dict, List, Optional, Union, Any, AsyncIterator
import base64
from io import BytesIO
import asyncio
from pydantic import BaseModel
from tenacity import (
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_random_exponential,
)
from llama_cpp import Llama
from app.config import LLMSettings, config
from app.exceptions import TokenLimitExceeded
from app.logger import logger
from app.schema import (
    ROLE_VALUES,
    TOOL_CHOICE_TYPE,
    TOOL_CHOICE_VALUES,
    Message,
    ToolChoice,
)

# Define models that support vision capabilities
MULTIMODAL_MODELS = ["qwen-vl-7b"]

class TokenCounter:
    # Token constants
    BASE_MESSAGE_TOKENS = 4
    FORMAT_TOKENS = 2
    LOW_DETAIL_IMAGE_TOKENS = 85
    HIGH_DETAIL_TILE_TOKENS = 170

    def __init__(self):
        self.prompt_tokens = 0
        self.completion_tokens = 0
        self.total_tokens = 0

    def update(self, prompt_tokens: int, completion_tokens: int):
        self.prompt_tokens += prompt_tokens
        self.completion_tokens += completion_tokens
        self.total_tokens = self.prompt_tokens + self.completion_tokens

    def reset(self):
        self.prompt_tokens = 0
        self.completion_tokens = 0
        self.total_tokens = 0

class ChatCompletionMessage(BaseModel):
    role: str
    content: Optional[str] = None
    tool_calls: Optional[List[Dict[str, Any]]] = None
    function_call: Optional[Dict[str, Any]] = None

class LLM:
    """
    LLM class for handling interactions with local GGUF models using llama-cpp-python.
    """

    # Define the context window sizes as class variables
    TEXT_MODEL_CONTEXT_SIZE = 16384  # Increased from 8192
    VISION_MODEL_CONTEXT_SIZE = 8192  # Increased from 4096

    def __init__(self, settings: Optional[LLMSettings] = None, config_name: str = None):
        """
        Initialize the LLM with settings.
        Args:
            settings: LLM configuration settings
            config_name: Optional name for config lookup
        """
        # Ensure settings is an LLMSettings instance
        if settings is None:
            settings = config.llm

        # Double-check that settings is an LLMSettings instance
        if not isinstance(settings, LLMSettings):
            raise TypeError(f"Expected LLMSettings instance, got {type(settings)}")

        self.model = settings.model
        self.model_path = settings.model_path
        self.max_tokens = settings.max_tokens
        self.temperature = settings.temperature
        self.token_counter = TokenCounter()

        # Initialize model instances as None, will be loaded on demand
        self._text_model = None
        self._vision_model = None

        # Vision model settings
        self.vision_settings = settings.vision

        logger.info(f"Initialized LLM with model: {self.model}, path: {self.model_path}")

    @property
    def text_model(self):
        """Lazy-load the text model"""
        if self._text_model is None:
            logger.info(f"Loading text model from {self.model_path}")
            self._text_model = Llama(
                model_path=self.model_path,
                n_ctx=self.TEXT_MODEL_CONTEXT_SIZE,  # Use the class variable
                n_gpu_layers=-1  # Use all GPU layers
            )
        return self._text_model

    @property
    def vision_model(self):
        """Lazy-load the vision model"""
        if self._vision_model is None and self.vision_settings:
            logger.info(f"Loading vision model from {self.vision_settings.model_path}")
            self._vision_model = Llama(
                model_path=self.vision_settings.model_path,
                n_ctx=self.VISION_MODEL_CONTEXT_SIZE,  # Use the class variable
                n_gpu_layers=-1,
                verbose=False
            )
        return self._vision_model

    def get_current_context_size(self, has_images: bool = False) -> int:
        """
        Get the current context size based on whether images are being processed.
        Args:
            has_images: Whether the request contains images
        Returns:
            Current context window size
        """
        if has_images and self.model in MULTIMODAL_MODELS:
            return self.VISION_MODEL_CONTEXT_SIZE
        return self.TEXT_MODEL_CONTEXT_SIZE

    def format_messages(self, messages: List[Union[dict, Message]], supports_images: bool = False) -> List[dict]:
        """
        Format messages for the model.
        Args:
            messages: List of messages to format
            supports_images: Whether the model supports images
        Returns:
            List of formatted messages
        """
        formatted_messages = []
        for msg in messages:
            if isinstance(msg, Message):
                msg_dict = msg.model_dump()
            else:
                msg_dict = msg.copy()

            # Handle image content if model supports it
            if supports_images and isinstance(msg_dict.get("content"), list):
                content_list = []
                for item in msg_dict["content"]:
                    if isinstance(item, dict) and item.get("type") == "image_url":
                        # Handle image URLs or base64 data
                        image_url = item.get("image_url", {})
                        if isinstance(image_url, dict):
                            url = image_url.get("url", "")
                            if url.startswith("data:image"):
                                # Keep base64 images as is
                                content_list.append(item)
                            else:
                                # For file paths, convert to base64
                                try:
                                    if os.path.exists(url):
                                        with open(url, "rb") as img_file:
                                            img_data = img_file.read()
                                            base64_data = base64.b64encode(img_data).decode("utf-8")
                                            mime_type = "image/jpeg"  # Assume JPEG, could be improved
                                            data_url = f"data:{mime_type};base64,{base64_data}"
                                            content_list.append({
                                                "type": "image_url",
                                                "image_url": {"url": data_url}
                                            })
                                    else:
                                        logger.warning(f"Image file not found: {url}")
                                        content_list.append({"type": "text", "text": f"[Image: {url} not found]"})
                                except Exception as e:
                                    logger.error(f"Error processing image: {e}")
                                    content_list.append({"type": "text", "text": f"[Image processing error: {str(e)}]"})
                        else:
                            # Direct URL string
                            content_list.append(item)
                    else:
                        # Text or other content
                        content_list.append(item)
                msg_dict["content"] = content_list

            formatted_messages.append(msg_dict)

        return formatted_messages

    def count_tokens(self, text: str) -> int:
        """
        Count tokens in text using a simple approximation.
        Args:
            text: Text to count tokens for
        Returns:
            Approximate token count
        """
        # Simple approximation: 1 token ≈ 4 characters for English text
        return len(text) // 4 + 1

    def count_message_tokens(self, messages: List[dict]) -> int:
        """
        Count tokens in a list of messages.
        Args:
            messages: List of messages
        Returns:
            Total token count
        """
        token_count = 0
        for msg in messages:
            # Count role tokens
            token_count += TokenCounter.BASE_MESSAGE_TOKENS

            # Count content tokens
            content = msg.get("content", "")
            if isinstance(content, str):
                token_count += self.count_tokens(content)
            elif isinstance(content, list):
                for item in content:
                    if isinstance(item, dict):
                        if item.get("type") == "text":
                            token_count += self.count_tokens(item.get("text", ""))
                        elif item.get("type") == "image_url":
                            # Approximate token count for images
                            token_count += TokenCounter.LOW_DETAIL_IMAGE_TOKENS

            # Count function/tool tokens if present
            if "function_call" in msg:
                token_count += self.count_tokens(str(msg["function_call"]))
            if "tool_calls" in msg:
                token_count += self.count_tokens(str(msg["tool_calls"]))

        return token_count

    def check_token_limit(self, input_tokens: int, has_images: bool = False) -> bool:
        """
        Check if input tokens are within limits.
        Args:
            input_tokens: Number of input tokens
            has_images: Whether the request contains images
        Returns:
            True if within limits, False otherwise
        """
        context_size = self.get_current_context_size(has_images)
        return input_tokens + self.max_tokens <= context_size

    def get_limit_error_message(self, input_tokens: int, has_images: bool = False) -> str:
        """
        Get error message for token limit exceeded.
        Args:
            input_tokens: Number of input tokens
            has_images: Whether the request contains images
        Returns:
            Error message
        """
        context_size = self.get_current_context_size(has_images)
        return f"Token limit exceeded: {input_tokens} input tokens + {self.max_tokens} max output tokens > {context_size} context window"

    def update_token_count(self, prompt_tokens: int, completion_tokens: int):
        """
        Update token counter.
        Args:
            prompt_tokens: Number of prompt tokens
            completion_tokens: Number of completion tokens
        """
        self.token_counter.update(prompt_tokens, completion_tokens)

    def _format_prompt_for_llama(self, messages: List[dict]) -> str:
        """
        Format messages into a prompt string for Llama models.
        Args:
            messages: List of messages
        Returns:
            Formatted prompt string
        """
        prompt = ""
        for msg in messages:
            role = msg.get("role", "user")
            content = msg.get("content", "")

            if role == "system":
                prompt += f"<|system|>\n{content}\n"
            elif role == "user":
                prompt += f"<|user|>\n{content}\n"
            elif role == "assistant":
                prompt += f"<|assistant|>\n{content}\n"
            else:
                # Handle other roles as user
                prompt += f"<|user|>\n{content}\n"

        # Add final assistant marker for completion
        prompt += "<|assistant|>\n"
        return prompt

    def _format_vision_prompt(self, messages: List[dict]) -> str:
        """
        Format messages for vision models, handling both text and images.
        Args:
            messages: List of messages with potential image content
        Returns:
            Formatted prompt string with image references
        """
        prompt = ""
        for msg in messages:
            role = msg.get("role", "user")
            content = msg.get("content", "")

            if role == "system":
                prompt += f"<|system|>\n{content}\n"
            elif role == "user":
                if isinstance(content, str):
                    prompt += f"<|user|>\n{content}\n"
                elif isinstance(content, list):
                    prompt += "<|user|>\n"
                    for item in content:
                        if isinstance(item, dict):
                            if item.get("type") == "text":
                                prompt += f"{item.get('text', '')}\n"
                            elif item.get("type") == "image_url":
                                image_url = item.get("image_url", {})
                                if isinstance(image_url, dict):
                                    url = image_url.get("url", "")
                                    prompt += f"[IMAGE: {url}]\n"
                                else:
                                    prompt += f"[IMAGE: {image_url}]\n"
            elif role == "assistant":
                prompt += f"<|assistant|>\n{content}\n"

        # Add final assistant marker for completion
        prompt += "<|assistant|>\n"
        return prompt

    def _extract_tool_calls(self, completion_text: str) -> List[Dict[str, Any]]:
        """
        Extract tool calls from completion text.
        Args:
            completion_text: Text to extract tool calls from
        Returns:
            List of extracted tool calls
        """
        tool_calls = []

        # Simple parsing for function calls in the format:
        # ```
        # {"name": "function_name", "arguments": {"arg1": "value1"}}
        # ```
        import re
        import json

        # Find all JSON blocks
        json_blocks = re.findall(r'``````', completion_text, re.DOTALL)

        for i, block in enumerate(json_blocks):
            try:
                data = json.loads(block)
                if isinstance(data, dict) and "name" in data and "arguments" in data:
                    tool_calls.append({
                        "id": f"call_{i}",
                        "type": "function",
                        "function": {
                            "name": data["name"],
                            "arguments": json.dumps(data["arguments"])
                        }
                    })
            except json.JSONDecodeError:
                logger.warning(f"Failed to parse JSON block: {block}")

        return tool_calls

    @retry(
        wait=wait_random_exponential(min=1, max=60),
        stop=stop_after_attempt(6),
        retry=retry_if_exception_type(Exception),
    )
    async def ask_tool(
        self,
        messages: List[Union[dict, Message]],
        system_msgs: Optional[List[Union[dict, Message]]] = None,
        timeout: int = 300,
        tools: Optional[List[dict]] = None,
        tool_choice: TOOL_CHOICE_TYPE = ToolChoice.AUTO,
        temperature: Optional[float] = None,
        **kwargs,
    ) -> Optional[ChatCompletionMessage]:
        """
        Ask LLM using functions/tools and return the response.
        Args:
            messages: List of conversation messages
            system_msgs: Optional system messages to prepend
            timeout: Request timeout in seconds
            tools: List of tools to use
            tool_choice: Tool choice strategy
            temperature: Sampling temperature for the response
            **kwargs: Additional completion arguments
        Returns:
            ChatCompletionMessage: The model's response
        Raises:
            TokenLimitExceeded: If token limits are exceeded
            ValueError: If tools, tool_choice, or messages are invalid
            Exception: For unexpected errors
        """
        try:
            # Validate tool_choice
            if tool_choice not in TOOL_CHOICE_VALUES:
                raise ValueError(f"Invalid tool_choice: {tool_choice}")

            # Check if the model supports images
            supports_images = self.model in MULTIMODAL_MODELS

            # Format messages
            if system_msgs:
                system_msgs = self.format_messages(system_msgs, supports_images)
                messages = system_msgs + self.format_messages(messages, supports_images)
            else:
                messages = self.format_messages(messages, supports_images)

            # Check for images in messages
            has_images = False
            for msg in messages:
                content = msg.get("content", "")
                if isinstance(content, list):
                    for item in content:
                        if isinstance(item, dict) and item.get("type") == "image_url":
                            has_images = True
                            break

            # Calculate input token count
            input_tokens = self.count_message_tokens(messages)

            # If there are tools, calculate token count for tool descriptions
            tools_tokens = 0
            if tools:
                for tool in tools:
                    tools_tokens += self.count_tokens(str(tool))
            input_tokens += tools_tokens

            # Check if token limits are exceeded
            if not self.check_token_limit(input_tokens, has_images):
                error_message = self.get_limit_error_message(input_tokens, has_images)
                # Raise a special exception that won't be retried
                raise TokenLimitExceeded(error_message)

            # Validate tools if provided
            if tools:
                for tool in tools:
                    if not isinstance(tool, dict) or "type" not in tool:
                        raise ValueError("Each tool must be a dict with 'type' field")

            # Prepare tool instructions if needed
            tool_instructions = ""
            if tools:
                import json
                tool_instructions = "You have access to the following tools:\n"
                for tool in tools:
                    tool_instructions += json.dumps(tool, indent=2) + "\n\n"

                if tool_choice == ToolChoice.REQUIRED:
                    tool_instructions += "You MUST use one of these tools in your response.\n"
                elif tool_choice == ToolChoice.AUTO:
                    tool_instructions += "Use these tools if needed to complete the task.\n"

            # Add tool instructions to system message if present
            if tool_instructions:
                has_system = False
                for msg in messages:
                    if msg.get("role") == "system":
                        msg["content"] = msg.get("content", "") + "\n\n" + tool_instructions
                        has_system = True
                        break

                if not has_system:
                    messages.insert(0, {"role": "system", "content": tool_instructions})

            # Use vision model if content has images and vision model is available
            if has_images and supports_images and self.vision_model:
                logger.info("Using vision model for image content")
                prompt = self._format_vision_prompt(messages)
                model = self.vision_model
            else:
                prompt = self._format_prompt_for_llama(messages)
                model = self.text_model

            # Set temperature
            temp = temperature if temperature is not None else self.temperature

            # Create a task for model completion with timeout
            completion_task = asyncio.create_task(
                asyncio.to_thread(
                    model.create_completion,
                    prompt=prompt,
                    max_tokens=self.max_tokens,
                    temperature=temp,
                    stop=["<|user|>", "<|system|>"],
                    **kwargs
                )
            )

            try:
                # Wait for completion with timeout
                completion = await asyncio.wait_for(completion_task, timeout=timeout)
            except asyncio.TimeoutError:
                logger.error(f"Model completion timed out after {timeout} seconds")
                raise Exception(f"Model completion timed out after {timeout} seconds")

            # Extract completion text
            completion_text = completion.get("choices", [{}])[0].get("text", "").strip()

            # Estimate token counts
            prompt_tokens = self.count_tokens(prompt)
            completion_tokens = self.count_tokens(completion_text)

            # Update token counter
            self.update_token_count(prompt_tokens, completion_tokens)

            # Check for tool calls in the response
            tool_calls = None
            if tools and completion_text:
                tool_calls = self._extract_tool_calls(completion_text)

            # Create response message
            response = ChatCompletionMessage(
                role="assistant",
                content=completion_text if not tool_calls else None,
                tool_calls=tool_calls
            )

            return response

        except TokenLimitExceeded:
            # Re-raise token limit errors without logging
            raise
        except ValueError as ve:
            logger.error(f"Validation error in ask_tool: {ve}")
            raise
        except Exception as e:
            logger.error(f"Unexpected error in ask_tool: {e}")
            raise

    @retry(
        wait=wait_random_exponential(min=1, max=60),
        stop=stop_after_attempt(6),
        retry=retry_if_exception_type(Exception),
    )
    async def ask(
        self,
        messages: List[Union[dict, Message]],
        system_msgs: Optional[List[Union[dict, Message]]] = None,
        stream: bool = False,
        timeout: int = 300,
        temperature: Optional[float] = None,
        **kwargs,
    ) -> Union[str, AsyncIterator[str]]:
        """
        Ask LLM and return the response.
        Args:
            messages: List of conversation messages
            system_msgs: Optional system messages to prepend
            stream: Whether to stream the response
            timeout: Request timeout in seconds
            temperature: Sampling temperature for the response
            **kwargs: Additional completion arguments
        Returns:
            String response or async iterator of response chunks
        Raises:
            TokenLimitExceeded: If token limits are exceeded
            Exception: For unexpected errors
        """
        try:
            # Check if the model supports images
            supports_images = self.model in MULTIMODAL_MODELS

            # Format messages
            if system_msgs:
                system_msgs = self.format_messages(system_msgs, supports_images)
                messages = system_msgs + self.format_messages(messages, supports_images)
            else:
                messages = self.format_messages(messages, supports_images)

            # Check for images in messages
            has_images = False
            for msg in messages:
                content = msg.get("content", "")
                if isinstance(content, list):
                    for item in content:
                        if isinstance(item, dict) and item.get("type") == "image_url":
                            has_images = True
                            break

            # Calculate input token count
            input_tokens = self.count_message_tokens(messages)

            # Check if token limits are exceeded
            if not self.check_token_limit(input_tokens, has_images):
                error_message = self.get_limit_error_message(input_tokens, has_images)
                # Raise a special exception that won't be retried
                raise TokenLimitExceeded(error_message)

            # Use vision model if content has images and vision model is available
            if has_images and supports_images and self.vision_model:
                logger.info("Using vision model for image content")
                prompt = self._format_vision_prompt(messages)
                model = self.vision_model
            else:
                prompt = self._format_prompt_for_llama(messages)
                model = self.text_model

            # Set temperature
            temp = temperature if temperature is not None else self.temperature

            if stream:
                # Create streaming response
                async def response_stream():
                    try:
                        # Create a generator that yields completion chunks
                        completion_generator = model.create_completion(
                            prompt=prompt,
                            max_tokens=self.max_tokens,
                            temperature=temp,
                            stop=["<|user|>", "<|system|>"],
                            stream=True,
                            **kwargs
                        )

                        total_text = ""
                        for chunk in completion_generator:
                            chunk_text = chunk.get("choices", [{}])[0].get("text", "")
                            total_text += chunk_text
                            yield chunk_text

                        # Update token counts after streaming completes
                        prompt_tokens = self.count_tokens(prompt)
                        completion_tokens = self.count_tokens(total_text)
                        self.update_token_count(prompt_tokens, completion_tokens)

                    except Exception as e:
                        logger.error(f"Error in streaming response: {e}")
                        raise

                return response_stream()
            else:
                # Create a task for model completion with timeout
                completion_task = asyncio.create_task(
                    asyncio.to_thread(
                        model.create_completion,
                        prompt=prompt,
                        max_tokens=self.max_tokens,
                        temperature=temp,
                        stop=["<|user|>", "<|system|>"],
                        **kwargs
                    )
                )

                try:
                    # Wait for completion with timeout
                    completion = await asyncio.wait_for(completion_task, timeout=timeout)
                except asyncio.TimeoutError:
                    logger.error(f"Model completion timed out after {timeout} seconds")
                    raise Exception(f"Model completion timed out after {timeout} seconds")

                # Extract completion text
                completion_text = completion.get("choices", [{}])[0].get("text", "").strip()

                # Estimate token counts
                prompt_tokens = self.count_tokens(prompt)
                completion_tokens = self.count_tokens(completion_text)

                # Update token counter
                self.update_token_count(prompt_tokens, completion_tokens)

                return completion_text

        except TokenLimitExceeded:
            # Re-raise token limit errors without logging
            raise
        except Exception as e:
            logger.error(f"Unexpected error in ask: {e}")
            raise
