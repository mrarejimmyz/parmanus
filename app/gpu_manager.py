"""
GPU Memory Manager for CUDA optimization in ParManusAI.

This module provides intelligent GPU memory management, monitoring,
and optimization for CUDA environments.
"""

import gc
import os
import threading
import time
from dataclasses import dataclass
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple

import torch
from loguru import logger


class GPUState(Enum):
    """GPU state enumeration."""

    AVAILABLE = "available"
    BUSY = "busy"
    OVERLOADED = "overloaded"
    ERROR = "error"


@dataclass
class GPUMemoryInfo:
    """GPU memory information structure."""

    total: float  # Total GPU memory in GB
    used: float  # Used GPU memory in GB
    free: float  # Free GPU memory in GB
    allocated: float  # Allocated by current process in GB
    reserved: float  # Reserved by current process in GB
    utilization: float  # Memory utilization ratio (0-1)
    timestamp: float  # Timestamp of measurement


@dataclass
class ModelMemoryProfile:
    """Memory profile for a model."""

    model_path: str
    estimated_size: float  # Estimated size in GB
    actual_size: float  # Actual loaded size in GB
    optimal_layers: int  # Optimal number of GPU layers
    load_time: float  # Time taken to load
    performance_score: float  # Performance score (0-1)


class CUDAGPUManager:
    """Manages CUDA GPU resources and memory."""

    def __init__(self, force_cuda: bool = None, force_gpu_layers: int = None):
        """Initialize the GPU manager.

        Args:
            force_cuda (bool, optional): Force CUDA mode if True, force CPU if False,
                                       or auto-detect if None
            force_gpu_layers (int, optional): Number of layers to force on GPU,
                                            or None for auto
        """
        self.cuda_available = self._check_cuda_availability()
        self.gpu_layers = force_gpu_layers

        # Handle force_cuda parameter
        if force_cuda is not None:
            self.should_use_gpu = bool(force_cuda) and self.cuda_available
        else:
            self.should_use_gpu = self.cuda_available

        self.device = torch.device("cuda" if self.should_use_gpu else "cpu")
        self.text_model_limit = 6.0  # Default 6GB for text models
        self.vision_model_limit = 2.0  # Default 2GB for vision models

        if self.should_use_gpu:
            self.start_monitoring()

    def _check_cuda_availability(self) -> bool:
        """Check if CUDA is available."""
        try:
            has_cuda = torch.cuda.is_available()
            if has_cuda:
                logger.info("CUDA detected via PyTorch")
            return has_cuda
        except Exception as e:
            logger.warning(f"CUDA check failed: {e}")
            return False

    def get_memory_info(self) -> dict:
        """Get current GPU memory usage."""
        if not self.cuda_available:
            return {"total": 0, "used": 0, "free": 0}

        try:
            total = torch.cuda.get_device_properties(0).total_memory / (1024**3)
            reserved = torch.cuda.memory_reserved(0) / (1024**3)
            allocated = torch.cuda.memory_allocated(0) / (1024**3)
            free = total - reserved

            return {
                "total": total,
                "used": reserved,
                "free": free,
                "allocated": allocated,
            }
        except Exception as e:
            logger.error(f"Failed to get GPU memory info: {e}")
            return {"total": 0, "used": 0, "free": 0}

    def get_gpu_memory_info(self) -> Dict[str, float]:
        """Get detailed GPU memory information.

        Returns:
            Dict containing total, used, free and cached memory in GB
        """
        if not self.cuda_available:
            return {
                "total": 0.0,
                "used": 0.0,
                "free": 0.0,
                "cached": 0.0,
            }

        try:
            total = torch.cuda.get_device_properties(0).total_memory / (1024**3)
            reserved = torch.cuda.memory_reserved(0) / (1024**3)
            allocated = torch.cuda.memory_allocated(0) / (1024**3)
            cached = reserved - allocated
            free = total - reserved

            return {
                "total": total,
                "used": reserved,
                "free": free,
                "cached": cached,
            }
        except Exception as e:
            logger.warning(f"Failed to get GPU memory info: {e}")
            return {
                "total": 0.0,
                "used": 0.0,
                "free": 0.0,
                "cached": 0.0,
            }

    def configure_memory_limits(
        self, text_limit: float = None, vision_limit: float = None
    ):
        """Configure memory limits for models."""
        if text_limit is not None:
            self.text_model_limit = float(text_limit)
        if vision_limit is not None:
            self.vision_model_limit = float(vision_limit)
        logger.info(
            f"GPU memory limits configured: Text={self.text_model_limit}GB, Vision={self.vision_model_limit}GB"
        )

    def start_monitoring(self):
        """Start GPU memory monitoring."""
        if not self.cuda_available:
            return
        try:
            memory_info = self.get_memory_info()
            logger.info(
                f"GPU monitoring started. Total memory: {memory_info['total']:.1f}GB"
            )
        except Exception as e:
            logger.error(f"Failed to start GPU monitoring: {e}")

    async def cleanup(self):
        """Clean up GPU resources."""
        if not self.cuda_available:
            return
        try:
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
            logger.info("GPU memory cache cleared")
        except Exception as e:
            logger.error(f"Failed to cleanup GPU memory: {e}")

    def force_cpu(self):
        """Force CPU-only mode regardless of CUDA availability."""
        self.should_use_gpu = False
        self.device = torch.device("cpu")
        logger.info("Forced CPU-only mode")

    def force_gpu(self):
        """Try to force GPU mode if CUDA is available."""
        if self.cuda_available:
            self.should_use_gpu = True
            self.device = torch.device("cuda")
            logger.info("Forced GPU mode")
        else:
            logger.warning("Cannot force GPU mode - CUDA not available")
            self.should_use_gpu = False
            self.device = torch.device("cpu")

    def get_available_memory(self) -> float:
        """Get available GPU memory in GB."""
        info = self.get_gpu_memory_info()
        return info["free"] + info["cached"]

    def get_total_memory(self) -> float:
        """Get total GPU memory in GB."""
        info = self.get_gpu_memory_info()
        return info["total"]

    def estimate_memory_required(self, model_path: str) -> float:
        """Estimate memory required for model in GB."""
        try:
            model_size = os.path.getsize(model_path) / (1024**3)
            # Add 20% overhead for loading
            return model_size * 1.2
        except Exception as e:
            logger.warning(f"Failed to estimate model memory: {e}")
            return 0.0

    def __bool__(self) -> bool:
        """Boolean representation of GPU availability."""
        return self.cuda_available


# Type alias for backward compatibility
GPUManager = CUDAGPUManager

# Global instance
_gpu_manager = None


def get_gpu_manager() -> CUDAGPUManager:
    """Get or create GPU manager instance."""
    global _gpu_manager
    if _gpu_manager is None:
        _gpu_manager = CUDAGPUManager()
    return _gpu_manager
