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

import psutil
import tiktoken
from llama_cpp import Llama
from pydantic import BaseModel

from app.config import LLMSettings, config
from app.exceptions import TokenLimitExceeded
from app.gpu_manager import CUDAGPUManager, get_gpu_manager
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
        """Update token counts."""
        self.prompt_tokens += prompt_tokens
        self.completion_tokens += completion_tokens
        self.total_tokens = self.prompt_tokens + self.completion_tokens


class LLM:
    """Base LLM class for handling interactions with local GGUF models."""

    # Define the context window sizes as class variables
    TEXT_MODEL_CONTEXT_SIZE = 8192
    VISION_MODEL_CONTEXT_SIZE = 4096
    MAX_ALLOWED_OUTPUT_TOKENS = 2048

    # Thread pool for model loading and inference
    _executor = ThreadPoolExecutor(max_workers=2)

    def __init__(self, settings: Optional[LLMSettings] = None):
        """Initialize the LLM with settings."""
        if settings is None:
            settings = config.llm
        if not isinstance(settings, LLMSettings):
            raise TypeError(f"Expected LLMSettings instance, got {type(settings)}")

        self.settings = settings
        self.model = settings.model
        self.model_path = settings.model_path
        self.max_tokens = min(settings.max_tokens, self.MAX_ALLOWED_OUTPUT_TOKENS)
        self.temperature = settings.temperature
        self.token_counter = TokenCounter()

        # Vision model settings
        self.vision_settings = settings.vision
        self.vision_enabled = False if not settings.vision else True

        # Cache keys for model instances
        self._text_model_key = f"{self.model_path}_{self.TEXT_MODEL_CONTEXT_SIZE}"
        self._vision_model_key = None
        if self.vision_settings:
            self._vision_model_key = (
                f"{self.vision_settings.model_path}_{self.VISION_MODEL_CONTEXT_SIZE}"
            )

        # Initialize locks if they don't exist
        if self._text_model_key not in MODEL_LOCKS:
            MODEL_LOCKS[self._text_model_key] = asyncio.Lock()

        if self._vision_model_key and self._vision_model_key not in MODEL_LOCKS:
            MODEL_LOCKS[self._vision_model_key] = asyncio.Lock()

    def _load_text_model(self):
        """Load text model with memory mapping for faster loading"""
        if self._text_model_key not in MODEL_CACHE:
            logger.info(f"Loading text model from {self.model_path}")
            start_time = time.time()

            model = Llama(
                model_path=self.model_path,
                n_ctx=self.TEXT_MODEL_CONTEXT_SIZE,
                n_gpu_layers=-1,  # Use all GPU layers
                n_threads=os.cpu_count(),  # Use all available CPU threads
                use_mmap=True,  # Use memory mapping for faster loading
                use_mlock=True,  # Lock memory to prevent swapping
            )

            MODEL_CACHE[self._text_model_key] = model
            load_time = time.time() - start_time
            logger.info(f"Text model loaded in {load_time:.2f} seconds")

        return MODEL_CACHE[self._text_model_key]

    @property
    def text_model(self):
        """Get cached text model or load if not available"""
        if self._text_model_key in MODEL_CACHE:
            return MODEL_CACHE[self._text_model_key]
        return self._load_text_model()

    def count_tokens(self, text: str) -> int:
        """
        Count the number of tokens in a text string.
        This is an estimate and may not match the exact tokenization of the model.
        """
        # Simple approximation: 1 token ≈ 4 characters for English text
        return len(text) // 4 + 1

    def update_token_count(self, prompt_tokens: int, completion_tokens: int) -> None:
        """Update the token counter with the latest usage."""
        self.token_counter.update(prompt_tokens, completion_tokens)

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


class LLMOptimized(LLM):
    """
    Optimized LLM wrapper with enhanced GPU management and memory optimization.

    Features:
    - Intelligent GPU memory management
    - Graceful vision model fallback
    - Adaptive layer allocation
    - Memory monitoring and cleanup
    - Dynamic context sizing
    """

    def __init__(self, settings: Optional[LLMSettings] = None):
        """Initialize the optimized LLM with GPU management."""
        super().__init__(settings)

        # GPU management with configuration
        force_cuda = getattr(config, "gpu", {}).get("force_cuda", False)
        force_gpu_layers = getattr(config, "gpu", {}).get("force_gpu_layers", 0)
        self.gpu_manager = CUDAGPUManager(
            force_cuda=force_cuda, force_gpu_layers=force_gpu_layers
        )

        logger.info(
            f"Initialized optimized LLM: {self.model}, GPU: {self.gpu_manager.cuda_available}, "
            f"Vision: {self.vision_enabled}"
        )


# For backward compatibility - keep the LLM name as an alias to LLMOptimized
LLM = LLMOptimized
