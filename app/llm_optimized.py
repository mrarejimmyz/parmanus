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
                "utilization": reserved / total if total > 0 else 0
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
        max_layers = int(available_memory / (memory_per_layer * 1.5)) if memory_per_layer > 0 else 0
        
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

        # GPU management
        self.gpu_manager = GPUMemoryManager()
        
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
                gpu_layers = self.gpu_manager.optimize_gpu_layers(estimated_layers, model_size_gb)
            
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
                        self.gpu_manager.optimize_gpu_layers(estimated_layers, model_size_gb),
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


# Alias for backward compatibility
LLM = LLMOptimized

