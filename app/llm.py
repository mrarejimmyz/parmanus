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

# Default maximum output tokens (can be overridden in config)
DEFAULT_MAX_OUTPUT_TOKENS = 8192  # Half of typical 16K context window

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

class LLM:
    """
    LLM interface for local models.
    """
    def __init__(
        self,
        model: str = "llama-jb",
        model_path: Optional[str] = None,
        temperature: float = 0.7,
        max_tokens: Optional[int] = None,
    ):
        """
        Initialize LLM with model and parameters.
        Args:
            model: Model name
            model_path: Path to model file
            temperature: Temperature for sampling
            max_tokens: Maximum number of tokens to generate
        """
        self.model = model
        self.temperature = temperature
        self.token_counter = TokenCounter()
        # Use config value if available, otherwise use default or provided value
        self.max_tokens = config.llm_settings.get("max_output_tokens", max_tokens or DEFAULT_MAX_OUTPUT_TOKENS)
        self.TEXT_MODEL_CONTEXT_SIZE = 16384  # Default context size for text models
        self.VISION_MODEL_CONTEXT_SIZE = 16384  # Default context size for vision models

        # Initialize models
        self.text_model = None
        self.vision_model = None

        # Store adjusted max tokens for dynamic adjustment
        self._adjusted_max_tokens = self.max_tokens

        # Initialize models
        self._initialize_models()
        logger.info(f"Initialized LLM with model: {model}, path: {model_path}")
    def _initialize_models(self):
        """
        Initialize text and vision models.
        """
        # Initialize text model
        try:
            self.text_model = Llama(
                model_path="/models/llama-jb.gguf",
                n_ctx=self.TEXT_MODEL_CONTEXT_SIZE,
                n_gpu_layers=0,
            )
        except Exception as e:
            logger.error(f"Failed to initialize text model: {e}")
            raise
        # Initialize vision model if needed
        if self.model in MULTIMODAL_MODELS:
            try:
                self.vision_model = Llama(
                    model_path="/models/qwen-vl-7b.gguf",
                    n_ctx=self.VISION_MODEL_CONTEXT_SIZE,
                    n_gpu_layers=0,
                )
            except Exception as e:
                logger.error(f"Failed to initialize vision model: {e}")
                # Continue without vision model
    def count_tokens(self, text: str) -> int:
        """
        Count tokens in text.
        Args:
            text: Text to count tokens for
        Returns:
            Number of tokens
        """
        if not text:
            return 0
        if self.text_model:
            return len(self.text_model.tokenize(text.encode("utf-8")))
        # Fallback to rough estimate
        return math.ceil(len(text) / 4)
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
    def _format_prompt_for_llama(self, messages: List[Dict]) -> str:
        """
        Format messages for Llama model.
        Args:
            messages: List of message dictionaries
        Returns:
            Formatted prompt string
        """
        prompt = ""
        for msg in messages:
            role = msg.get("role", "")
            content = msg.get("content", "")
            if role == "system":
                prompt += f"<|system|>\n{content}\n"
            elif role == "user":
                prompt += f"<|user|>\n{content}\n"
            elif role == "assistant":
                prompt += f"<|assistant|>\n{content}\n"
            elif role == "tool":
                prompt += f"<|tool|>\n{content}\n"
        prompt += "<|assistant|>\n"
        return prompt
    def _format_vision_prompt(self, messages: List[Dict]) -> str:
        """
        Format messages for vision model.
        Args:
            messages: List of message dictionaries
        Returns:
            Formatted prompt string
        """
        # Similar to _format_prompt_for_llama but with image handling
        # This is a simplified implementation
        prompt = ""
        for msg in messages:
            role = msg.get("role", "")
            content = msg.get("content", "")
            if isinstance(content, list):
                # Process multimodal content
                text_parts = []
                for item in content:
                    if isinstance(item, dict):
                        if item.get("type") == "text":
                            text_parts.append(item.get("text", ""))
                        # Image URLs are handled separately
                content = " ".join(text_parts)
            if role == "system":
                prompt += f"<|system|>\n{content}\n"
            elif role == "user":
                prompt += f"<|user|>\n{content}\n"
            elif role == "assistant":
                prompt += f"<|assistant|>\n{content}\n"
            elif role == "tool":
                prompt += f"<|tool|>\n{content}\n"
        prompt += "<|assistant|>\n"
        return prompt
    def format_messages(self, messages: List[Dict], supports_images: bool = False) -> List[Dict]:
        """
        Format messages for model input.
        Args:
            messages: List of message dictionaries
            supports_images: Whether the model supports images
        Returns:
            Formatted messages
        """
        formatted_messages = []
        for msg in messages:
            if not isinstance(msg, dict):
                continue
            role = msg.get("role", "")
            content = msg.get("content", "")
            if role not in ROLE_VALUES:
                continue
            formatted_msg = {"role": role}
            # Handle content
            if isinstance(content, list) and supports_images:
                # Process multimodal content
                formatted_content = []
                for item in content:
                    if isinstance(item, dict):
                        if item.get("type") == "text":
                            formatted_content.append({"type": "text", "text": item.get("text", "")})
                        elif item.get("type") == "image_url" and supports_images:
                            image_url = item.get("image_url", {})
                            if isinstance(image_url, dict) and "url" in image_url:
                                url = image_url["url"]
                                if url.startswith("data:image/"):
                                    # Handle base64 images
                                    formatted_content.append({"type": "image_url", "image_url": {"url": url}})
                formatted_msg["content"] = formatted_content
            else:
                # Text-only content
                formatted_msg["content"] = str(content) if content is not None else ""
            formatted_messages.append(formatted_msg)
        return formatted_messages
    def count_message_tokens(self, messages: List[Dict]) -> int:
        """
        Count tokens in messages.
        Args:
            messages: List of message dictionaries
        Returns:
            Number of tokens
        """
        token_count = 0
        for msg in messages:
            role = msg.get("role", "")
            content = msg.get("content", "")
            # Count role tokens
            token_count += self.BASE_MESSAGE_TOKENS
            # Count content tokens
            if isinstance(content, list):
                for item in content:
                    if isinstance(item, dict):
                        if item.get("type") == "text":
                            token_count += self.count_tokens(item.get("text", ""))
                        elif item.get("type") == "image_url":
                            # Estimate image tokens
                            token_count += self.LOW_DETAIL_IMAGE_TOKENS
            else:
                token_count += self.count_tokens(str(content) if content is not None else "")
            # Count function call tokens
            if "function_call" in msg:
                token_count += self.count_tokens(str(msg["function_call"]))
            if "tool_calls" in msg:
                token_count += self.count_tokens(str(msg["tool_calls"]))
        return token_count
    def check_token_limit(self, input_tokens: int, has_images: bool = False) -> bool:
        """
        Check if input tokens are within limits.
        Dynamically adjusts max_tokens if needed to fit within context window.

        Args:
            input_tokens: Number of input tokens
            has_images: Whether the request contains images

        Returns:
            True if within limits, False otherwise
        """
        context_size = self.get_current_context_size(has_images)
        safety_buffer = 100  # Add a small buffer for safety

        # If input tokens alone are too large, we can't proceed
        if input_tokens + safety_buffer >= context_size:
            return False

        # Dynamically adjust max_tokens based on input size
        self._adjusted_max_tokens = min(self.max_tokens, context_size - input_tokens - safety_buffer)

        # Check if we have a reasonable amount of output tokens available
        return self._adjusted_max_tokens >= 100  # Ensure we have at least 100 tokens for output

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

        # Calculate how many tokens would be available for output
        available_for_output = max(0, context_size - input_tokens - 100)  # 100 token safety buffer

        return (f"Token limit exceeded: {input_tokens} input tokens + {self.max_tokens} max output tokens > "
                f"{context_size} context window. The input alone requires {input_tokens} tokens, leaving only "
                f"{available_for_output} tokens available for output (requested {self.max_tokens}). "
                f"Consider reducing input size or adjusting max_output_tokens in configuration.")

    def update_token_count(self, prompt_tokens: int, completion_tokens: int):
        """
        Update token counter.
        Args:
            prompt_tokens: Number of prompt tokens
            completion_tokens: Number of completion tokens
        """
        self.token_counter.update(prompt_tokens, completion_tokens)
    def reset_token_count(self):
        """
        Reset token counter.
        """
        self.token_counter.reset()
    @retry(
        retry=retry_if_exception_type(Exception),
        stop=stop_after_attempt(3),
        wait=wait_random_exponential(min=1, max=10),
    )
    async def chat_completion(
        self,
        messages: List[Dict],
        system_msgs: Optional[List[Dict]] = None,
        temperature: Optional[float] = None,
        stream: bool = False,
        timeout: int = 60,
        tools: Optional[List[Dict]] = None,
        tool_choice: TOOL_CHOICE_TYPE = ToolChoice.AUTO,  # type: ignore
        **kwargs,
    ) -> Union[str, AsyncIterator[str]]:
        """
        Generate chat completion.
        Args:
            messages: List of message dictionaries
            system_msgs: Optional system messages
            temperature: Temperature for sampling
            stream: Whether to stream the response
            timeout: Timeout in seconds
            tools: Optional list of tools
            tool_choice: Tool choice strategy
            **kwargs: Additional arguments
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
                elif tool_choice == ToolChoice.NONE:
                    tool_instructions += "Do not use any tools in your response.\n"
                # Add tool instructions to system message
                if system_msgs:
                    # Append to last system message
                    for i in range(len(system_msgs) - 1, -1, -1):
                        if system_msgs[i]["role"] == "system":
                            content = system_msgs[i].get("content", "")
                            if isinstance(content, str):
                                system_msgs[i]["content"] = content + "\n\n" + tool_instructions
                            break
                else:
                    # Add new system message
                    system_msgs = [{"role": "system", "content": tool_instructions}]
                    messages = system_msgs + messages
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
                            max_tokens=self._adjusted_max_tokens,  # Use dynamically adjusted value
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
                        max_tokens=self._adjusted_max_tokens,  # Use dynamically adjusted value
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
    async def ask(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        temperature: Optional[float] = None,
        stream: bool = False,
        timeout: int = 60,
        **kwargs,
    ) -> Union[str, AsyncIterator[str]]:
        """
        Ask a question and get a response.
        Args:
            prompt: User prompt
            system_prompt: Optional system prompt
            temperature: Temperature for sampling
            stream: Whether to stream the response
            timeout: Timeout in seconds
            **kwargs: Additional arguments
        Returns:
            String response or async iterator of response chunks
        """
        messages = [{"role": "user", "content": prompt}]
        system_msgs = [{"role": "system", "content": system_prompt}] if system_prompt else None
        return await self.chat_completion(
            messages=messages,
            system_msgs=system_msgs,
            temperature=temperature,
            stream=stream,
            timeout=timeout,
            **kwargs,
        )
