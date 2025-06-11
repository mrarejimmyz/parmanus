import asyncio
import base64
import gc
import json
import os
import re
import time
from concurrent.futures import ThreadPoolExecutor
from functools import lru_cache
from io import BytesIO
from typing import Any, Dict, Generator, List, Optional, Union

import tiktoken
from llama_cpp import Llama
from pydantic import BaseModel, Field

from app.config import LLMSettings, config
from app.exceptions import ParManusError, TokenLimitExceeded
from app.gpu_manager import GPUManager, get_gpu_manager
from app.logger import logger
from app.schema import (
    ROLE_VALUES,
    TOOL_CHOICE_TYPE,
    TOOL_CHOICE_VALUES,
    Function,
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


class GPUMemoryManager:
    """Manages GPU memory allocation and monitoring for CUDA."""

    def __init__(self):
        self.cuda_available = self._check_cuda_availability()
        self.memory_stats = {}

    def _check_cuda_availability(self) -> bool:
        """Check if CUDA is available."""
        try:
            import torch

            return torch.cuda.is_available()
        except ImportError:
            logger.warning("PyTorch not available, GPU monitoring disabled")
            return False

    def get_gpu_memory_info(self) -> Dict[str, float]:
        """Get current GPU memory usage information."""
        if not self.cuda_available:
            return {"total": 0, "used": 0, "free": 0, "utilization": 0}

        try:
            import torch

            torch.cuda.synchronize()

            total = torch.cuda.get_device_properties(0).total_memory
            reserved = torch.cuda.memory_reserved(0)
            allocated = torch.cuda.memory_allocated(0)
            free = total - reserved

            return {
                "total": total / 1024**3,  # GB
                "used": reserved / 1024**3,  # GB
                "allocated": allocated / 1024**3,  # GB
                "free": free / 1024**3,  # GB
                "utilization": reserved / total if total > 0 else 0,
            }
        except Exception as e:
            logger.debug(f"Failed to get GPU memory info: {e}")
            return {"total": 0, "used": 0, "free": 0, "utilization": 0}

    def should_use_gpu(self, model_size_estimate: float = 1.0) -> bool:
        """Determine if GPU should be used based on memory availability."""
        if not self.cuda_available:
            return False

        memory_info = self.get_gpu_memory_info()
        available_memory = memory_info["free"]

        # Conservative estimate: need at least 2x model size for safe operation
        required_memory = model_size_estimate * 2

        if available_memory < required_memory:
            logger.warning(
                f"Insufficient GPU memory: {available_memory:.1f}GB available, "
                f"{required_memory:.1f}GB required"
            )
            return False

        return memory_info["utilization"] < GPU_MEMORY_THRESHOLD

    def optimize_gpu_layers(self, total_layers: int, model_size_gb: float) -> int:
        """Calculate optimal number of GPU layers based on available memory."""
        if not self.cuda_available:
            return 0

        memory_info = self.get_gpu_memory_info()
        available_memory = memory_info["free"]

        # Estimate memory per layer (rough approximation)
        memory_per_layer = model_size_gb / total_layers if total_layers > 0 else 0

        # Calculate how many layers we can fit
        max_layers = (
            int(available_memory / (memory_per_layer * 1.5))
            if memory_per_layer > 0
            else 0
        )

        # Use conservative approach
        optimal_layers = min(max_layers, total_layers)

        logger.info(
            f"GPU optimization: {optimal_layers}/{total_layers} layers on GPU "
            f"(available: {available_memory:.1f}GB)"
        )

        return optimal_layers

    def cleanup_gpu_memory(self):
        """Clean up GPU memory."""
        if not self.cuda_available:
            return

        try:
            import torch

            torch.cuda.empty_cache()
            torch.cuda.synchronize()
            logger.debug("GPU memory cache cleared")
        except Exception as e:
            logger.debug(f"Failed to cleanup GPU memory: {e}")


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


class LLMOptimized:
    """Optimized LLM implementation with improved memory management and GPU utilization."""

    DEFAULT_TEMPERATURE = 0.7
    DEFAULT_TOP_P = 0.95
    DEFAULT_CONTEXT_SIZE = 4096
    VISION_MODEL_CONTEXT_SIZE = 2048
    MAX_ALLOWED_OUTPUT_TOKENS = 2048
    max_tokens = 2048  # Default max tokens for completion

    # Thread pool for model loading and inference - required for tool execution
    _executor = ThreadPoolExecutor(max_workers=2)

    def __init__(self, settings: LLMSettings, gpu_manager=None):
        """Initialize LLM with optimized settings."""
        self.settings = settings
        self.model_path = settings.model_path
        self.model = settings.model
        self.max_tokens = min(settings.max_tokens, self.MAX_ALLOWED_OUTPUT_TOKENS)
        self.temperature = settings.temperature
        self.gpu_manager = gpu_manager or get_gpu_manager()
        self._text_model_key = f"{self.model}_text"
        self._model = None  # Store model instance
        self.tokenizer = None
        self.token_counter = TokenCounter()  # Initialize token counter

        # Vision model settings - required for compatibility
        self.vision_settings = settings.vision
        self.vision_enabled = settings.vision.enabled if settings.vision else False
        self._vision_model_key = None
        if self.vision_settings and self.vision_settings.enabled:
            self._vision_model_key = f"{self.vision_settings.model}_vision"

        logger.info(
            f"Initialized optimized LLM: {self.model}, "
            f"GPU: {self.gpu_manager.cuda_available}, "
            f"Vision: {self.vision_enabled}"
        )

    def format_messages(
        self,
        messages: List[Message],
        tools=None,
        tool_choice: Optional[ToolChoice] = None,
    ) -> str:
        """Format a list of messages into a single string for the model input."""
        formatted = ""
        for msg in messages:
            if msg.role == "system":
                formatted += f"System: {msg.content}\n"
            elif msg.role == "user":
                formatted += f"User: {msg.content}\n"
            elif msg.role == "assistant":
                formatted += f"Assistant: {msg.content}\n"
            if msg.tool_calls:
                for tc in msg.tool_calls:
                    if tc.function:
                        func_str = f"{tc.function.name}({tc.function.arguments})"
                        formatted += f"Tool call: {func_str}\n"
        if tools:
            formatted += "\nAvailable tools:\n"
            for tool in tools:
                formatted += f"- {tool['name']}: {tool['description']}\n"
        if tool_choice:
            formatted += f"\nTool choice: {tool_choice}\n"
        return formatted

    async def _preload_text_model(self):
        """Preload text model with optimized settings."""
        try:
            logger.info(f"Preloading text model: {self.model_path}")

            if self._text_model_key not in MODEL_CACHE:
                # Load model synchronously since Llama doesn't support async
                self._model = self._load_text_model()
                MODEL_CACHE[self._text_model_key] = self._model

            return MODEL_CACHE[self._text_model_key]

        except Exception as e:
            logger.error(f"Error preloading text model: {e}")
            raise

    def _load_text_model(self):
        """Load text model with GPU optimization."""
        try:
            # Check GPU availability
            model_size = os.path.getsize(self.model_path) / (1024**3)
            use_gpu = bool(self.gpu_manager.should_use_gpu)  # Access as property
            gpu_layers = 35 if use_gpu else 0

            # Initialize model
            model = Llama(
                model_path=self.model_path,
                n_ctx=4096,
                n_gpu_layers=gpu_layers,
                n_threads=min(os.cpu_count(), 8),
                use_mmap=True,
            )

            logger.info(f"Model loaded with {gpu_layers} GPU layers")
            return model

        except Exception as e:
            logger.error(f"GPU load failed: {e}, falling back to CPU")
            return self._load_cpu_fallback()

    @property
    def text_model(self):
        """Get cached text model or load if not available."""
        if self._text_model_key in MODEL_CACHE:
            return MODEL_CACHE[self._text_model_key]
        return self._load_text_model()

    def count_tokens(self, text: str) -> int:
        """Count tokens with improved estimation."""
        if not text:
            return 0

        # More accurate token estimation
        # Average of 3.5 characters per token for English
        return max(1, len(text.encode("utf-8")) // 4)

    def update_token_count(self, prompt_tokens: int, completion_tokens: int):
        """Update token counter and check limits."""
        self.token_counter.update(prompt_tokens, completion_tokens)

        # Check token limits against max_tokens (which is available in settings)
        if (
            hasattr(self.settings, "max_input_tokens")
            and self.settings.max_input_tokens
        ):
            if self.token_counter.total_tokens > self.settings.max_input_tokens:
                raise TokenLimitExceeded(
                    f"Total tokens ({self.token_counter.total_tokens}) "
                    f"exceeded limit ({self.settings.max_input_tokens})"
                )
        elif self.token_counter.total_tokens > self.DEFAULT_CONTEXT_SIZE:
            # Fallback to default context size limit
            raise TokenLimitExceeded(
                f"Total tokens ({self.token_counter.total_tokens}) "
                f"exceeded default limit ({self.DEFAULT_CONTEXT_SIZE})"
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

            # Clean up GPU memory if method exists
            if hasattr(self.gpu_manager, "cleanup_gpu_memory"):
                self.gpu_manager.cleanup_gpu_memory()
            elif hasattr(self.gpu_manager, "cleanup"):
                self.gpu_manager.cleanup()

            logger.info("Model cleanup completed")

        except Exception as e:
            logger.warning(f"Error during model cleanup: {e}")

    def get_memory_stats(self) -> Dict[str, Any]:
        """Get current memory statistics."""
        gpu_info = self.gpu_manager.get_gpu_memory_info()

        return {
            "gpu_memory": gpu_info,
            "models_loaded": len(MODEL_CACHE),
            "text_model_loaded": self._text_model_key in MODEL_CACHE,
            "vision_model_loaded": (
                self._vision_model_key in MODEL_CACHE
                if self._vision_model_key
                else False
            ),
            "vision_enabled": self.vision_enabled,
            "token_count": {
                "prompt": self.token_counter.prompt_tokens,
                "completion": self.token_counter.completion_tokens,
                "total": self.token_counter.total_tokens,
            },
        }

    def _load_cpu_fallback(self):
        """Fallback to CPU-only loading with minimal context."""
        model = Llama(
            model_path=self.model_path,
            n_ctx=4096,  # Minimum viable context
            n_gpu_layers=0,
            n_threads=os.cpu_count(),
            use_mmap=True,
        )
        MODEL_CACHE[self._text_model_key] = model
        logger.info("Model loaded in CPU-only mode")
        return model

    async def initialize_browser(self):
        """Initialize browser with async support."""
        try:
            # Import async version of playwright
            import subprocess

            from playwright.async_api import async_playwright

            # Kill existing Chrome processes
            subprocess.run(
                ["taskkill", "/F", "/IM", "chrome.exe"], capture_output=True, shell=True
            )

            # Initialize playwright in async mode
            self.playwright = await async_playwright().start()
            self.browser = await self.playwright.chromium.launch(
                headless=False,
                args=[
                    "--remote-debugging-port=9222",
                    "--no-sandbox",
                    "--disable-dev-shm-usage",
                ],
            )

            # Create isolated context and page
            self.context = await self.browser.new_context()
            self.page = await self.context.new_page()

            logger.info("Browser initialized successfully in async mode")
            return self.browser

        except Exception as e:
            logger.error(f"Browser initialization failed: {e}")
            return None

    async def navigate(self, url: str) -> bool:
        """Navigate to a URL with proper error handling."""
        try:
            if not hasattr(self, "page") or not self.page:
                logger.info("No active browser session, initializing...")
                await self.initialize_browser()

            if not self.page:
                raise Exception("Failed to create browser page")

            # Add http:// if missing
            if not url.startswith(("http://", "https://")):
                url = "https://" + url

            logger.info(f"Navigating to: {url}")

            # Set longer timeout and wait for network idle
            await self.page.goto(
                url, wait_until="networkidle", timeout=30000  # 30 seconds timeout
            )

            # Wait for page to be fully loaded
            await self.page.wait_for_load_state("networkidle")

            logger.info(f"Successfully loaded {url}")
            return True

        except Exception as e:
            logger.error(f"Navigation failed: {e}")
            await self.cleanup_browser()  # Cleanup on error
            return False

    async def cleanup_browser(self):
        """Clean up browser resources properly."""
        try:
            if hasattr(self, "page") and self.page:
                await self.page.close()

            if hasattr(self, "context") and self.context:
                await self.context.close()

            if hasattr(self, "browser") and self.browser:
                await self.browser.close()

            if hasattr(self, "playwright") and self.playwright:
                await self.playwright.stop()

            # Clear instance attributes
            for attr in ["page", "context", "browser", "playwright"]:
                if hasattr(self, attr):
                    delattr(self, attr)

            logger.info("Browser resources cleaned up successfully")

        except Exception as e:
            logger.error(f"Browser cleanup failed: {e}")

    def _parse_tool_calls(self, content: str) -> Optional[List[Dict]]:
        """Parse tool calls from model output."""
        tool_calls = []

        # Look for function call patterns
        pattern = r"(\w+)\((.*?)\)"
        matches = re.finditer(pattern, content)

        for match in matches:
            func_name = match.group(1)
            args_str = match.group(2)

            # Try to parse arguments as JSON
            try:
                args = json.loads(args_str)
            except json.JSONDecodeError:
                # If not valid JSON, treat as plain string
                args = args_str

            tool_calls.append(
                {
                    "id": f"call_{len(tool_calls)}",
                    "type": "function",
                    "function": {
                        "name": func_name,
                        "arguments": (
                            json.dumps(args) if isinstance(args, (dict, list)) else args
                        ),
                    },
                }
            )

        return tool_calls if tool_calls else None

    def _format_prompt_for_llama(
        self,
        messages: List[Message],
        tools=None,
        tool_choice: Optional[ToolChoice] = None,
    ) -> str:
        """Format messages for llama.cpp models using the chat template."""
        formatted_tokens = []

        # Process each message
        for msg in messages:
            # Add BOS token for first message
            if len(formatted_tokens) == 0:
                formatted_tokens.append("<|begin_of_text|>")

            # Add header
            formatted_tokens.append("<|start_header_id|>")
            formatted_tokens.append(msg.role)
            formatted_tokens.append("<|end_header_id|>\n\n")

            # Add content
            formatted_tokens.append(msg.content)

            # Add EOT token
            formatted_tokens.append("<|eot_id|>")

        # Add final assistant header for response
        formatted_tokens.append("<|start_header_id|>assistant<|end_header_id|>\n\n")

        # Join all tokens
        prompt = "".join(formatted_tokens)

        # Add tool context if present
        if tools:
            tool_str = "\nAvailable tools:\n"
            for tool in tools:
                tool_str += f"- {tool['name']}: {tool['description']}\n"
            prompt += tool_str

        if tool_choice:
            prompt += f"\nTool choice: {tool_choice}\n"

        return prompt

    def _format_prompt_for_llama(self, messages: list) -> str:
        """Format messages for Llama chat model format."""
        formatted_messages = []
        for msg in messages:
            if isinstance(msg, str):
                # Convert string messages to dict format
                msg = {"role": "user", "content": msg}

            if not isinstance(msg, dict) or "role" not in msg or "content" not in msg:
                raise ValueError(f"Invalid message format: {msg}")

            role = msg["role"]
            content = msg["content"]

            # Format according to Llama chat template
            formatted_msg = (
                f"<|start_header_id|>{role}<|end_header_id|>\n\n{content}<|eot_id|>"
            )
            formatted_messages.append(formatted_msg)

        # Add system message if not present
        if not any(msg.get("role") == "system" for msg in messages):
            system_msg = "<|start_header_id|>system<|end_header_id|>\n\nYou are a helpful AI assistant.<|eot_id|>"
            formatted_messages.insert(0, system_msg)

        # Add final assistant header
        formatted_messages.append("<|start_header_id|>assistant<|end_header_id|>\n\n")

        return "".join(formatted_messages)

    async def ask(
        self,
        messages: List[Union[Message, Dict[str, Any]]],
        system_msgs: Optional[List[Union[Message, Dict[str, Any]]]] = None,
        temperature: Optional[float] = None,
        stream: bool = False,
        timeout: int = 120,
        **kwargs,
    ) -> Union[str, Generator[str, None, None]]:
        """Ask the model to generate a response."""
        try:
            # Format messages
            formatted_messages = []
            if system_msgs:
                formatted_messages.extend(
                    [
                        msg if isinstance(msg, dict) else msg.to_dict()
                        for msg in system_msgs
                    ]
                )
            formatted_messages.extend(
                [msg if isinstance(msg, dict) else msg.to_dict() for msg in messages]
            )

            # Format prompt
            prompt = self._format_prompt_for_llama(formatted_messages)

            # Get model
            model = self.text_model

            # Set temperature
            temp = temperature if temperature is not None else self.temperature

            # Calculate safe max tokens
            prompt_tokens = self.count_tokens(prompt)
            safe_max_tokens = min(
                self.max_tokens, self.DEFAULT_CONTEXT_SIZE - prompt_tokens - 100
            )

            if safe_max_tokens <= 0:
                raise TokenLimitExceeded("Prompt too long for available context")

            if stream:
                # Streaming response
                def generate_stream():
                    try:
                        for chunk in model.create_completion(
                            prompt=prompt,
                            max_tokens=safe_max_tokens,
                            temperature=temp,
                            stream=True,
                            stop=["<|user|>", "<|system|>", "<|eot_id|>"],
                            **kwargs,
                        ):
                            yield chunk["choices"][0]["text"]
                    except Exception as e:
                        logger.error(f"Error in streaming completion: {e}")
                        yield f"[Error: {str(e)}]"

                return generate_stream()
            else:
                # Non-streaming response
                completion = await asyncio.wait_for(
                    asyncio.get_event_loop().run_in_executor(
                        self._executor,
                        lambda: model.create_completion(
                            prompt=prompt,
                            max_tokens=safe_max_tokens,
                            temperature=temp,
                            stop=["<|user|>", "<|system|>", "<|eot_id|>"],
                            **kwargs,
                        ),
                    ),
                    timeout=timeout,
                )

                # Extract completion text
                completion_text = completion["choices"][0]["text"]

                # Update token counter
                completion_tokens = self.count_tokens(completion_text)
                self.update_token_count(prompt_tokens, completion_tokens)

                return completion_text

        except TokenLimitExceeded:
            raise
        except asyncio.TimeoutError:
            logger.error(f"Model completion timed out after {timeout} seconds")
            return f"[Response incomplete due to timeout after {timeout} seconds]"
        except Exception as e:
            logger.error(f"Error in ask: {e}")
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
        """Ask the model with tool calling support."""
        if timeout is None:
            timeout = 120  # Default timeout for tool calls

        start_time = time.time()
        last_exception = None

        for attempt in range(max_retries + 1):
            try:
                # Format messages
                formatted_messages = []
                if system_msgs:
                    formatted_messages.extend(
                        [
                            msg if isinstance(msg, dict) else msg.to_dict()
                            for msg in system_msgs
                        ]
                    )
                formatted_messages.extend(
                    [
                        msg if isinstance(msg, dict) else msg.to_dict()
                        for msg in messages
                    ]
                )

                # Create enhanced prompt with tool information
                prompt = self._format_prompt_for_llama(formatted_messages)

                # Add tool definitions if provided
                if tools:
                    tool_definitions = "\n\nAvailable tools:\n"
                    for tool in tools:
                        tool_definitions += f"- {tool['name']}: {tool['description']}\n"
                    prompt += tool_definitions

                    # Add tool choice instructions
                    if tool_choice == ToolChoice.AUTO:
                        prompt += "\nYou may use these tools if appropriate to help answer the user's question.\n"
                    elif tool_choice == ToolChoice.REQUIRED:
                        prompt += "\nYou must use one of these tools to answer the user's question.\n"

                # Get model and set temperature
                model = self.text_model
                temperature = temp if temp is not None else self.temperature

                # Calculate safe max tokens
                prompt_tokens = self.count_tokens(prompt)
                safe_max_tokens = min(
                    self.max_tokens, self.DEFAULT_CONTEXT_SIZE - prompt_tokens - 100
                )

                if safe_max_tokens <= 0:
                    raise TokenLimitExceeded("Prompt too long for available context")

                # Run model with timeout
                completion = await asyncio.wait_for(
                    asyncio.get_event_loop().run_in_executor(
                        self._executor,
                        lambda: model.create_completion(
                            prompt=prompt,
                            max_tokens=safe_max_tokens,
                            temperature=temperature,
                            stop=["<|user|>", "<|system|>", "<|eot_id|>"],
                            **kwargs,
                        ),
                    ),
                    timeout=timeout,
                )

                # Extract completion text
                completion_text = completion["choices"][0]["text"].strip()

                # Parse tool calls from response
                tool_calls = self._parse_tool_calls(completion_text) if tools else []

                # Update token counter
                completion_tokens = self.count_tokens(completion_text)
                self.update_token_count(prompt_tokens, completion_tokens)

                # Return structured response
                return {
                    "content": completion_text,
                    "tool_calls": tool_calls or [],
                    "usage": {
                        "prompt_tokens": prompt_tokens,
                        "completion_tokens": completion_tokens,
                        "total_tokens": prompt_tokens + completion_tokens,
                    },
                    "elapsed_time": time.time() - start_time,
                    "attempts": attempt + 1,
                }

            except asyncio.TimeoutError as e:
                last_exception = e
                elapsed = time.time() - start_time

                if attempt < max_retries:
                    # Increase timeout for retry
                    timeout = min(timeout * 1.5, 180)  # Cap at 3 minutes
                    logger.warning(
                        f"Tool call timed out after {elapsed:.1f}s, "
                        f"retrying with {timeout}s timeout (attempt {attempt + 1}/{max_retries})"
                    )
                    continue
                else:
                    logger.error(
                        f"Tool call failed after {max_retries + 1} attempts "
                        f"(total time: {elapsed:.1f}s)"
                    )
                    # Return partial result with timeout indication
                    return {
                        "content": f"[Response incomplete due to timeout after {elapsed:.1f}s and {max_retries + 1} attempts]",
                        "tool_calls": [],
                        "usage": {
                            "prompt_tokens": 0,
                            "completion_tokens": 0,
                            "total_tokens": 0,
                        },
                        "elapsed_time": elapsed,
                        "attempts": attempt + 1,
                        "error": "timeout",
                    }

            except Exception as e:
                last_exception = e
                elapsed = time.time() - start_time

                if attempt < max_retries and not isinstance(e, TokenLimitExceeded):
                    logger.warning(f"Tool call error on attempt {attempt + 1}: {e}")
                    continue
                else:
                    logger.error(f"Tool call failed permanently: {e}")
                    raise

        # This should not be reached, but just in case
        raise last_exception or RuntimeError("Unexpected tool call failure")
