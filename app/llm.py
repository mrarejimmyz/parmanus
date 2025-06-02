import asyncio
import base64
import json
import os
import re
import time
from concurrent.futures import ThreadPoolExecutor
from functools import lru_cache
from io import BytesIO
from typing import Any, Dict, Generator, List, Optional, Union
import gc

import tiktoken
from llama_cpp import Llama
from pydantic import BaseModel

from app.config import LLMSettings, config
from app.exceptions import TokenLimitExceeded
from app.logger import logger
from app.gpu_manager import CUDAGPUManager
from app.schema import (
    ROLE_VALUES,
    TOOL_CHOICE_TYPE,
    TOOL_CHOICE_VALUES,
    Message,
    ToolChoice,
)

# Define models that support vision capabilities
MULTIMODAL_MODELS = ["qwen-vl-7b"]

# Global model cache to persist models between requests
MODEL_CACHE = {}

# Global locks to prevent concurrent loading of the same model
MODEL_LOCKS = {}

# GPU memory management
GPU_MEMORY_THRESHOLD = 0.8  # 80% GPU memory usage threshold
ENABLE_GPU_MONITORING = True


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


class LLMOptimized:
    """
    Optimized LLM wrapper for llama-cpp-python with enhanced GPU management.
    
    Features:
    - Intelligent GPU memory management
    - Graceful vision model fallback
    - Adaptive layer allocation
    - Memory monitoring and cleanup
    """

    # Model context sizes
    TEXT_MODEL_CONTEXT_SIZE = 4096
    VISION_MODEL_CONTEXT_SIZE = 2048
    MAX_ALLOWED_OUTPUT_TOKENS = 2048

    def __init__(self, settings: LLMSettings = None):
        """Initialize the optimized LLM with GPU management."""
        self.settings = settings or config.llm
        self.model = self.settings.model
        self.model_path = self.settings.model_path
        self.max_tokens = min(self.settings.max_tokens, self.MAX_ALLOWED_OUTPUT_TOKENS)
        self.temperature = self.settings.temperature

        # GPU management with configuration
        force_cuda = getattr(config, 'gpu', {}).get('force_cuda', False)
        force_gpu_layers = getattr(config, 'gpu', {}).get('force_gpu_layers', 0)
        self.gpu_manager = CUDAGPUManager(force_cuda=force_cuda, force_gpu_layers=force_gpu_layers)
        
        # Thread pool for model operations
        self._executor = ThreadPoolExecutor(max_workers=2)

        # Token tracking
        self.token_counter = TokenCounter()

        # Vision model settings with fallback
        self.vision_settings = settings.vision if settings else None
        self.vision_enabled = self._validate_vision_model()

        # Cache keys for model instances
        self._text_model_key = f"{self.model_path}_{self.TEXT_MODEL_CONTEXT_SIZE}"
        self._vision_model_key = None
        if self.vision_enabled:
            self._vision_model_key = (
                f"{self.vision_settings.model_path}_{self.VISION_MODEL_CONTEXT_SIZE}"
            )

        logger.info(
            f"Initialized optimized LLM: {self.model}, GPU: {self.gpu_manager.cuda_available}, "
            f"Vision: {self.vision_enabled}"
        )

        # Initialize locks if they don't exist
        if self._text_model_key not in MODEL_LOCKS:
            MODEL_LOCKS[self._text_model_key] = asyncio.Lock()

        if self._vision_model_key and self._vision_model_key not in MODEL_LOCKS:
            MODEL_LOCKS[self._vision_model_key] = asyncio.Lock()

        # Preload models with GPU optimization
        if self._text_model_key not in MODEL_CACHE:
            asyncio.create_task(self._preload_text_model_safe())

        if self._vision_model_key and self._vision_model_key not in MODEL_CACHE:
            asyncio.create_task(self._preload_vision_model_safe())

    def _validate_vision_model(self) -> bool:
        """Validate if vision model can be loaded."""
        if not self.vision_settings:
            return False
        
        vision_path = self.vision_settings.model_path
        if not os.path.exists(vision_path):
            logger.warning(f"Vision model not found: {vision_path}, disabling vision capabilities")
            return False
        
        # Check file size and integrity
        try:
            file_size = os.path.getsize(vision_path) / (1024**3)  # GB
            if file_size < 0.1:  # Less than 100MB is suspicious
                logger.warning(f"Vision model file too small: {file_size:.1f}GB, disabling vision")
                return False
            
            logger.info(f"Vision model validated: {vision_path} ({file_size:.1f}GB)")
            return True
            
        except Exception as e:
            logger.warning(f"Failed to validate vision model: {e}")
            return False

    async def _preload_text_model_safe(self):
        """Preload text model with GPU optimization."""
        async with MODEL_LOCKS[self._text_model_key]:
            if self._text_model_key not in MODEL_CACHE:
                await self._preload_text_model()
            else:
                logger.info(f"Text model {self._text_model_key} already loaded")

    async def _preload_vision_model_safe(self):
        """Preload vision model with fallback handling."""
        if not self.vision_enabled:
            return

        async with MODEL_LOCKS[self._vision_model_key]:
            if self._vision_model_key not in MODEL_CACHE:
                await self._preload_vision_model()
            else:
                logger.info(f"Vision model {self._vision_model_key} already loaded")

    async def _preload_text_model(self):
        """Preload text model with GPU optimization."""
        try:
            logger.info(f"Preloading text model: {self.model_path}")
            await asyncio.get_event_loop().run_in_executor(
                self._executor, self._load_text_model
            )
            logger.info("Text model preloaded successfully")
        except Exception as e:
            logger.error(f"Error preloading text model: {e}")
            raise

    async def _preload_vision_model(self):
        """Preload vision model with graceful fallback."""
        if not self.vision_enabled:
            return

        try:
            logger.info(f"Preloading vision model: {self.vision_settings.model_path}")
            await asyncio.get_event_loop().run_in_executor(
                self._executor, self._load_vision_model
            )
            logger.info("Vision model preloaded successfully")
        except Exception as e:
            logger.error(f"Error preloading vision model: {e}")
            logger.warning("Disabling vision capabilities due to loading failure")
            self.vision_enabled = False

    def _load_text_model(self):
        """Load text model with optimized GPU settings."""
        if self._text_model_key not in MODEL_CACHE:
            logger.info(f"Loading text model: {self.model_path}")
            start_time = time.time()

            # Estimate model size (rough approximation)
            model_size_gb = os.path.getsize(self.model_path) / (1024**3)
            
            # Determine GPU usage
            use_gpu = self.gpu_manager.should_use_gpu(model_size_gb)
            gpu_layers = 0
            
            if use_gpu:
                # Estimate total layers (rough approximation for common models)
                estimated_layers = int(model_size_gb * 10)  # Rough estimate
                gpu_layers = self.gpu_manager.optimize_gpu_layers(estimated_layers, model_size_gb, self.model_path)
            
            logger.info(f"Loading text model with {gpu_layers} GPU layers")

            try:
                model = Llama(
                    model_path=self.model_path,
                    n_ctx=self.TEXT_MODEL_CONTEXT_SIZE,
                    n_gpu_layers=gpu_layers,
                    n_threads=min(os.cpu_count(), 8),  # Limit CPU threads
                    use_mmap=True,
                    use_mlock=False,  # Disable mlock to reduce memory pressure
                    verbose=False,  # Reduce verbosity
                )

                MODEL_CACHE[self._text_model_key] = model
                load_time = time.time() - start_time
                
                # Log memory usage
                memory_info = self.gpu_manager.get_gpu_memory_info()
                logger.info(
                    f"Text model loaded in {load_time:.2f}s, "
                    f"GPU memory: {memory_info['used']:.1f}/{memory_info['total']:.1f}GB"
                )
                
            except Exception as e:
                logger.error(f"Failed to load text model: {e}")
                # Fallback to CPU-only
                if gpu_layers > 0:
                    logger.warning("Retrying with CPU-only mode")
                    model = Llama(
                        model_path=self.model_path,
                        n_ctx=self.TEXT_MODEL_CONTEXT_SIZE,
                        n_gpu_layers=0,
                        n_threads=os.cpu_count(),
                        use_mmap=True,
                        use_mlock=False,
                        verbose=False,
                    )
                    MODEL_CACHE[self._text_model_key] = model
                    logger.info("Text model loaded in CPU-only mode")
                else:
                    raise

        return MODEL_CACHE[self._text_model_key]

    def _load_vision_model(self):
        """Load vision model with graceful fallback."""
        if not self.vision_enabled:
            return None

        if self._vision_model_key not in MODEL_CACHE:
            logger.info(f"Loading vision model: {self.vision_settings.model_path}")
            start_time = time.time()

            try:
                # Check if we have enough GPU memory for vision model
                model_size_gb = os.path.getsize(self.vision_settings.model_path) / (1024**3)
                use_gpu = self.gpu_manager.should_use_gpu(model_size_gb)
                
                # Use fewer GPU layers for vision model to conserve memory
                gpu_layers = 0
                if use_gpu:
                    estimated_layers = int(model_size_gb * 8)  # Conservative estimate
                    gpu_layers = min(
                        self.gpu_manager.optimize_gpu_layers(estimated_layers, model_size_gb, self.vision_settings.model_path),
                        estimated_layers // 2  # Use only half the layers for vision
                    )

                logger.info(f"Loading vision model with {gpu_layers} GPU layers")

                model = Llama(
                    model_path=self.vision_settings.model_path,
                    n_ctx=self.VISION_MODEL_CONTEXT_SIZE,
                    n_gpu_layers=gpu_layers,
                    n_threads=min(os.cpu_count(), 4),  # Fewer threads for vision
                    use_mmap=True,
                    use_mlock=False,
                    verbose=False,
                )

                MODEL_CACHE[self._vision_model_key] = model
                load_time = time.time() - start_time
                
                memory_info = self.gpu_manager.get_gpu_memory_info()
                logger.info(
                    f"Vision model loaded in {load_time:.2f}s, "
                    f"GPU memory: {memory_info['used']:.1f}/{memory_info['total']:.1f}GB"
                )

            except Exception as e:
                logger.error(f"Failed to load vision model: {e}")
                logger.warning("Vision capabilities disabled due to loading failure")
                self.vision_enabled = False
                return None

        return MODEL_CACHE[self._vision_model_key]

    @property
    def text_model(self):
        """Get cached text model or load if not available."""
        if self._text_model_key in MODEL_CACHE:
            return MODEL_CACHE[self._text_model_key]
        return self._load_text_model()

    @property
    def vision_model(self):
        """Get cached vision model or return None if not available."""
        if not self.vision_enabled:
            return None

        if self._vision_model_key in MODEL_CACHE:
            return MODEL_CACHE[self._vision_model_key]
        return self._load_vision_model()

    def count_tokens(self, text: str) -> int:
        """Count tokens with improved estimation."""
        if not text:
            return 0
        
        # More accurate token estimation
        # Average of 3.5 characters per token for English
        return max(1, len(text.encode('utf-8')) // 4)

    def update_token_count(self, prompt_tokens: int, completion_tokens: int):
        """Update token counter and check limits."""
        self.token_counter.update(prompt_tokens, completion_tokens)
        
        # Check token limits
        if self.settings.max_input_tokens:
            if self.token_counter.total_tokens > self.settings.max_input_tokens:
                raise TokenLimitExceeded(
                    f"Total tokens ({self.token_counter.total_tokens}) "
                    f"exceeded limit ({self.settings.max_input_tokens})"
                )

    def cleanup_models(self):
        """Clean up models and free GPU memory."""
        try:
            # Clear model cache for this instance
            if self._text_model_key in MODEL_CACHE:
                del MODEL_CACHE[self._text_model_key]
            
            if self._vision_model_key and self._vision_model_key in MODEL_CACHE:
                del MODEL_CACHE[self._vision_model_key]
            
            # Force garbage collection
            gc.collect()
            
            # Clean up GPU memory
            self.gpu_manager.cleanup_gpu_memory()
            
            logger.info("Model cleanup completed")
            
        except Exception as e:
            logger.warning(f"Error during model cleanup: {e}")

    @classmethod
    def cleanup_all_models(cls):
        """Clean up all models in the global cache (class method for cleanup handlers)."""
        try:
            # Clear all models from global cache
            MODEL_CACHE.clear()
            
            # Force garbage collection
            gc.collect()
            
            logger.info("All models cleanup completed")
            
        except Exception as e:
            logger.warning(f"Error during global model cleanup: {e}")

    def get_memory_stats(self) -> Dict[str, Any]:
        """Get current memory statistics."""
        gpu_info = self.gpu_manager.get_gpu_memory_info()
        
        return {
            "gpu_memory": gpu_info,
            "models_loaded": len(MODEL_CACHE),
            "text_model_loaded": self._text_model_key in MODEL_CACHE,
            "vision_model_loaded": self._vision_model_key in MODEL_CACHE if self._vision_model_key else False,
            "vision_enabled": self.vision_enabled,
            "token_count": {
                "prompt": self.token_counter.prompt_tokens,
                "completion": self.token_counter.completion_tokens,
                "total": self.token_counter.total_tokens,
            }
        }

    def update_token_count(self, prompt_tokens: int, completion_tokens: int) -> None:
        """Update the token counter with the latest usage."""
        self.token_counter.update(prompt_tokens, completion_tokens)

    def check_token_limit(self, input_tokens: int, has_images: bool = False) -> bool:
        """Check if the input tokens exceed the model's context window."""
        # Adjust context size based on whether we're using vision model
        context_size = (
            self.VISION_MODEL_CONTEXT_SIZE
            if has_images
            else self.TEXT_MODEL_CONTEXT_SIZE
        )
        # Reserve space for completion tokens
        available_tokens = context_size - self.max_tokens
        return input_tokens <= available_tokens

    def get_limit_error_message(self, input_tokens: int, has_images: bool = False) -> str:
        """Get error message for token limit exceeded."""
        context_size = (
            self.VISION_MODEL_CONTEXT_SIZE
            if has_images
            else self.TEXT_MODEL_CONTEXT_SIZE
        )
        available_tokens = context_size - self.max_tokens
        return (
            f"Input tokens ({input_tokens}) exceed available context "
            f"({available_tokens} tokens available, {context_size} total context, "
            f"{self.max_tokens} reserved for completion)"
        )

    def format_messages(
        self,
        messages: List[Union[Message, Dict[str, Any]]],
        supports_images: bool = False,
    ) -> List[Dict[str, Any]]:
        """Format messages for the model."""
        formatted_messages = []
        for message in messages:
            if isinstance(message, Message):
                message_dict = message.model_dump()
            else:
                message_dict = message.copy()

            # Ensure role is valid
            if "role" not in message_dict or message_dict["role"] not in ROLE_VALUES:
                message_dict["role"] = "user"

            formatted_messages.append(message_dict)

        return formatted_messages

    def _format_prompt_for_llama(self, messages: List[Dict[str, Any]]) -> str:
        """Format messages into a prompt string for Llama models."""
        prompt = ""
        for message in messages:
            role = message["role"]
            content = message.get("content", "")

            if role == "system":
                prompt += f"<|system|>\n{content}\n"
            elif role == "user":
                prompt += f"<|user|>\n{content}\n"
            elif role == "assistant":
                prompt += f"<|assistant|>\n{content}\n"
            else:
                # Default to user for unknown roles
                prompt += f"<|user|>\n{content}\n"

        # Add the final assistant prefix to prompt the model to respond
        prompt += "<|assistant|>\n"
        return prompt

    def _format_vision_prompt(self, messages: List[Dict[str, Any]]) -> str:
        """Format messages for vision models, handling image content."""
        prompt = ""
        for message in messages:
            role = message["role"]
            content = message.get("content", "")

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
                                # For now, just indicate an image was here
                                # Vision handling will need to be model-specific
                                prompt += "[IMAGE]\n"
            elif role == "assistant":
                prompt += f"<|assistant|>\n{content}\n"
            else:
                # Default to user for unknown roles
                prompt += f"<|user|>\n{content}\n"

        # Add the final assistant prefix to prompt the model to respond
        prompt += "<|assistant|>\n"
        return prompt

    def count_message_tokens(self, messages: List[Dict[str, Any]]) -> int:
        """
        Count the number of tokens in a list of messages.
        This is an estimate and may not match the exact tokenization of the model.
        """
        total_tokens = 0
        for message in messages:
            content = message.get("content", "")
            if isinstance(content, str):
                total_tokens += self.count_tokens(content)
            elif isinstance(content, list):
                for item in content:
                    if isinstance(item, dict):
                        if item.get("type") == "text":
                            total_tokens += self.count_tokens(item.get("text", ""))
                        elif item.get("type") == "image_url":
                            # Add token estimate for images
                            total_tokens += TokenCounter.LOW_DETAIL_IMAGE_TOKENS
        return total_tokens

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
        Send a request to the model and get a response.

        Args:
            messages: List of messages to send to the model
            system_msgs: Optional system messages to prepend
            temperature: Temperature for sampling (0.0 to 1.0)
            stream: Whether to stream the response
            timeout: Timeout in seconds for the request
            **kwargs: Additional arguments to pass to the model

        Returns:
            The model's response as a string or a generator of strings if streaming

        Raises:
            TokenLimitExceeded: If the input exceeds the model's token limit
            Exception: For other unexpected errors
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
            if has_images and supports_images and self.vision_enabled:
                logger.info("Using vision model for image content")
                prompt = self._format_vision_prompt(messages)
                model = self.vision_model
            else:
                prompt = self._format_prompt_for_llama(messages)
                model = self.text_model

            # Set temperature
            temp = temperature if temperature is not None else self.temperature

            # Apply safe max tokens limit
            safe_max_tokens = min(self.max_tokens, self.MAX_ALLOWED_OUTPUT_TOKENS)

            if stream:
                # Create streaming response
                def generate_stream():
                    try:
                        for chunk in model.create_completion(
                            prompt=prompt,
                            max_tokens=safe_max_tokens,
                            temperature=temp,
                            stream=True,
                            stop=["<|user|>", "<|system|>"],
                            **kwargs,
                        ):
                            yield chunk["choices"][0]["text"]
                    except Exception as e:
                        logger.error(f"Error in streaming completion: {e}")
                        yield f"[Error: {str(e)}]"

                return generate_stream()
            else:
                # Create single response
                try:
                    # Run model inference in thread pool to avoid blocking the event loop
                    completion = await asyncio.wait_for(
                        asyncio.get_event_loop().run_in_executor(
                            self._executor,
                            lambda: model.create_completion(
                                prompt=prompt,
                                max_tokens=safe_max_tokens,
                                temperature=temp,
                                stop=["<|user|>", "<|system|>"],
                                **kwargs,
                            ),
                        ),
                        timeout=timeout,
                    )

                    # Extract completion text
                    completion_text = completion["choices"][0]["text"]

                    # Estimate token counts
                    prompt_tokens = self.count_tokens(prompt)
                    completion_tokens = self.count_tokens(completion_text)

                    # Update token counter
                    self.update_token_count(prompt_tokens, completion_tokens)

                    return completion_text
                except asyncio.TimeoutError:
                    logger.error(f"Model completion timed out after {timeout} seconds")
                    return (
                        f"[Response incomplete due to timeout after {timeout} seconds]"
                    )
                except Exception as e:
                    logger.error(f"Error in model completion: {e}")
                    return f"[Error: {str(e)}]"

        except TokenLimitExceeded:
            # Re-raise token limit exceptions without modification
            raise
        except Exception as e:
            logger.error(f"Unexpected error in ask method: {e}")
            return f"[Unexpected error: {str(e)}]"

    async def ask_tool(
        self,
        messages: List[Union[Message, Dict[str, Any]]],
        system_msgs: Optional[List[Union[Message, Dict[str, Any]]]] = None,
        tools: Optional[List[Dict[str, Any]]] = None,
        tool_choice: Optional[str] = None,
        temperature: Optional[float] = None,
        timeout: int = 120,
        **kwargs,
    ) -> Dict[str, Any]:
        """
        Send a request to the model with tool support and get a response.

        Args:
            messages: List of messages to send to the model
            system_msgs: Optional system messages to prepend
            tools: List of available tools in OpenAI format
            tool_choice: Tool choice strategy ("auto", "none", or specific tool)
            temperature: Temperature for sampling (0.0 to 1.0)
            timeout: Timeout in seconds for the request
            **kwargs: Additional arguments to pass to the model

        Returns:
            Dictionary with 'content' and optionally 'tool_calls'

        Raises:
            TokenLimitExceeded: If the input exceeds the model's token limit
            Exception: For other unexpected errors
        """
        try:
            # For now, since llama-cpp-python doesn't natively support function calling,
            # we'll simulate it by including tool information in the system prompt
            # and parsing the response for tool calls
            
            # Build enhanced system message with tool information
            enhanced_system_msgs = system_msgs or []
            
            if tools:
                tool_descriptions = []
                for tool in tools:
                    func = tool.get('function', {})
                    name = func.get('name', 'unknown')
                    desc = func.get('description', 'No description')
                    params = func.get('parameters', {})
                    
                    tool_desc = f"- {name}: {desc}"
                    if params and params.get('properties'):
                        param_names = list(params['properties'].keys())
                        tool_desc += f" (Parameters: {', '.join(param_names)})"
                    tool_descriptions.append(tool_desc)
                
                tool_system_msg = {
                    "role": "system",
                    "content": f"""You have access to the following tools:

{chr(10).join(tool_descriptions)}

To use a tool, respond with a JSON object in this format:
{{"tool_calls": [{{"function": {{"name": "tool_name", "arguments": {{"param": "value"}}}}}}]}}

You can also provide regular text responses. If you need to use a tool, include the tool call JSON in your response."""
                }
                enhanced_system_msgs = [tool_system_msg] + enhanced_system_msgs
            
            # Get regular response
            response_text = await self.ask(
                messages=messages,
                system_msgs=enhanced_system_msgs,
                temperature=temperature,
                timeout=timeout,
                **kwargs
            )
            
            # Try to parse tool calls from the response
            tool_calls = []
            content = response_text
            
            # Simple parsing for tool calls (this is a basic implementation)
            import json
            import re
            import uuid
            from app.schema import ToolCall, Function
            
            # Look for JSON tool call patterns
            tool_call_pattern = r'\{"tool_calls":\s*\[.*?\]\}'
            matches = re.findall(tool_call_pattern, response_text, re.DOTALL)
            
            for match in matches:
                try:
                    parsed = json.loads(match)
                    if 'tool_calls' in parsed:
                        for tc in parsed['tool_calls']:
                            if 'function' in tc:
                                # Convert to expected ToolCall format
                                tool_call = ToolCall(
                                    id=str(uuid.uuid4()),
                                    type="function",
                                    function=Function(
                                        name=tc['function'].get('name', ''),
                                        arguments=json.dumps(tc['function'].get('arguments', {}))
                                    )
                                )
                                tool_calls.append(tool_call)
                        # Remove the tool call JSON from content
                        content = content.replace(match, '').strip()
                except json.JSONDecodeError:
                    continue
            
            return {
                'content': content,
                'tool_calls': tool_calls
            }
            
        except TokenLimitExceeded:
            # Re-raise token limit exceptions without modification
            raise
        except Exception as e:
            logger.error(f"Unexpected error in ask_tool method: {e}")
            return {
                'content': f"[Unexpected error: {str(e)}]",
                'tool_calls': []
            }