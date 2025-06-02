"""
GPU Memory Manager for CUDA optimization in ParManusAI.

This module provides intelligent GPU memory management, monitoring,
and optimization for CUDA environments.
"""

import gc
import time
import threading
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass
from enum import Enum

from app.logger import logger


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
    used: float   # Used GPU memory in GB
    free: float   # Free GPU memory in GB
    allocated: float  # Allocated by current process in GB
    reserved: float   # Reserved by current process in GB
    utilization: float  # Memory utilization ratio (0-1)
    timestamp: float   # Timestamp of measurement


@dataclass
class ModelMemoryProfile:
    """Memory profile for a model."""
    model_path: str
    estimated_size: float  # Estimated size in GB
    actual_size: float     # Actual loaded size in GB
    optimal_layers: int    # Optimal number of GPU layers
    load_time: float       # Time taken to load
    performance_score: float  # Performance score (0-1)


class CUDAGPUManager:
    """
    Advanced GPU memory manager for CUDA environments.
    
    Features:
    - Real-time memory monitoring
    - Adaptive layer allocation
    - Model memory profiling
    - Automatic cleanup and optimization
    - Performance tracking
    """
    
    def __init__(self, 
                 memory_threshold: float = 0.8,
                 cleanup_threshold: float = 0.9,
                 monitoring_interval: float = 5.0,
                 force_cuda: bool = False,
                 force_gpu_layers: int = 0):
        """
        Initialize GPU manager.
        
        Args:
            memory_threshold: Memory usage threshold for warnings (0-1)
            cleanup_threshold: Memory usage threshold for automatic cleanup (0-1)
            monitoring_interval: Monitoring interval in seconds
            force_cuda: Force CUDA usage even if detection fails
            force_gpu_layers: Force minimum GPU layers when CUDA is available
        """
        self.memory_threshold = memory_threshold
        self.cleanup_threshold = cleanup_threshold
        self.monitoring_interval = monitoring_interval
        self.force_cuda = force_cuda
        self.force_gpu_layers = force_gpu_layers
        
        # CUDA availability check
        self.cuda_available = self._check_cuda_availability()
        
        # Override CUDA detection if forced
        if force_cuda and not self.cuda_available:
            logger.warning("Force CUDA enabled - overriding CUDA detection failure")
            self.cuda_available = True
        
        self.device_count = self._get_device_count()
        self.primary_device = 0
        
        # Memory tracking
        self.memory_history: List[GPUMemoryInfo] = []
        self.model_profiles: Dict[str, ModelMemoryProfile] = {}
        self.current_state = GPUState.AVAILABLE
        
        # Threading for monitoring
        self._monitoring_active = False
        self._monitor_thread: Optional[threading.Thread] = None
        self._lock = threading.Lock()
        
        # Performance metrics
        self.metrics = {
            "total_allocations": 0,
            "successful_allocations": 0,
            "cleanup_operations": 0,
            "oom_events": 0,
            "fallback_to_cpu": 0,
        }
        
        logger.info(f"GPU Manager initialized: CUDA={self.cuda_available}, Devices={self.device_count}")
        
        if self.cuda_available:
            self.start_monitoring()
    
    def _check_cuda_availability(self) -> bool:
        """Check if CUDA is available and functional through llama-cpp-python."""
        try:
            # First try PyTorch if available (preferred method)
            try:
                import torch
                if torch.cuda.is_available():
                    torch.cuda.synchronize()
                    logger.info("CUDA detected via PyTorch")
                    return True
            except ImportError:
                logger.info("PyTorch not available, checking llama-cpp-python CUDA support")
            except Exception as e:
                logger.warning(f"PyTorch CUDA check failed: {e}, trying llama-cpp-python")
            
            # Fallback: Test CUDA through llama-cpp-python
            from llama_cpp import Llama
            
            # Method 1: Check if n_gpu_layers parameter is available
            try:
                import inspect
                init_signature = inspect.signature(Llama.__init__)
                if 'n_gpu_layers' not in init_signature.parameters:
                    logger.warning("llama-cpp-python compiled without CUDA support (no n_gpu_layers parameter)")
                    return False
            except Exception:
                pass
            
            # Method 2: Try to detect CUDA through nvidia-smi
            try:
                import subprocess
                result = subprocess.run(['nvidia-smi'], capture_output=True, text=True, timeout=5)
                if result.returncode == 0:
                    logger.info("CUDA detected via nvidia-smi")
                    return True
            except Exception:
                pass
            
            # Method 3: Check for CUDA libraries
            try:
                import ctypes
                try:
                    # Try to load CUDA runtime library
                    ctypes.CDLL('libcudart.so')
                    logger.info("CUDA runtime library detected")
                    return True
                except OSError:
                    try:
                        # Windows CUDA library
                        ctypes.CDLL('cudart64_110.dll')
                        logger.info("CUDA runtime library detected (Windows)")
                        return True
                    except OSError:
                        pass
            except Exception:
                pass
            
            logger.warning("CUDA not detected through any method")
            return False
                
        except Exception as e:
            logger.error(f"CUDA availability check failed: {e}")
            return False
    
    def _get_device_count(self) -> int:
        """Get number of available CUDA devices."""
        if not self.cuda_available:
            return 0
        
        try:
            # Try PyTorch first
            import torch
            return torch.cuda.device_count()
        except ImportError:
            # Fallback: Try to detect devices through other means
            try:
                import subprocess
                result = subprocess.run(['nvidia-smi', '-L'], capture_output=True, text=True, timeout=5)
                if result.returncode == 0:
                    # Count GPU lines
                    gpu_count = len([line for line in result.stdout.split('\n') if line.startswith('GPU')])
                    logger.info(f"Detected {gpu_count} CUDA devices via nvidia-smi")
                    return gpu_count
            except Exception:
                pass
            
            # Default to 1 if CUDA is available but we can't count devices
            logger.warning("CUDA available but device count unknown, assuming 1 device")
            return 1
        except Exception:
            return 0
    
    def get_memory_info(self, device: int = None) -> GPUMemoryInfo:
        """
        Get detailed GPU memory information.
        
        Args:
            device: CUDA device index (default: primary device)
            
        Returns:
            GPUMemoryInfo object with current memory statistics
        """
        if not self.cuda_available:
            return GPUMemoryInfo(0, 0, 0, 0, 0, 0, time.time())
        
        device = device or self.primary_device
        
        try:
            import torch
            torch.cuda.synchronize(device)
            
            # Get memory statistics
            total = torch.cuda.get_device_properties(device).total_memory
            reserved = torch.cuda.memory_reserved(device)
            allocated = torch.cuda.memory_allocated(device)
            free = total - reserved
            
            # Convert to GB
            total_gb = total / (1024**3)
            used_gb = reserved / (1024**3)
            free_gb = free / (1024**3)
            allocated_gb = allocated / (1024**3)
            reserved_gb = reserved / (1024**3)
            
            utilization = reserved / total if total > 0 else 0
            
            return GPUMemoryInfo(
                total=total_gb,
                used=used_gb,
                free=free_gb,
                allocated=allocated_gb,
                reserved=reserved_gb,
                utilization=utilization,
                timestamp=time.time()
            )
            
        except Exception as e:
            logger.debug(f"PyTorch GPU memory info failed: {e}, trying nvidia-smi fallback")
            
            # Fallback: Try to get memory info via nvidia-smi
            try:
                import subprocess
                result = subprocess.run([
                    'nvidia-smi', '--query-gpu=memory.total,memory.used,memory.free', 
                    '--format=csv,noheader,nounits'
                ], capture_output=True, text=True, timeout=5)
                
                if result.returncode == 0:
                    lines = result.stdout.strip().split('\n')
                    if lines and device < len(lines):
                        memory_data = lines[device].split(', ')
                        if len(memory_data) == 3:
                            total_mb = float(memory_data[0])
                            used_mb = float(memory_data[1])
                            free_mb = float(memory_data[2])
                            
                            # Convert to GB
                            total_gb = total_mb / 1024
                            used_gb = used_mb / 1024
                            free_gb = free_mb / 1024
                            
                            utilization = used_mb / total_mb if total_mb > 0 else 0
                            
                            logger.info(f"GPU memory via nvidia-smi: {free_gb:.1f}GB free / {total_gb:.1f}GB total")
                            
                            return GPUMemoryInfo(
                                total=total_gb,
                                used=used_gb,
                                free=free_gb,
                                allocated=used_gb,  # Approximate
                                reserved=used_gb,   # Approximate
                                utilization=utilization,
                                timestamp=time.time()
                            )
            except Exception as nvidia_error:
                logger.debug(f"nvidia-smi fallback failed: {nvidia_error}")
            
            # Final fallback: Assume reasonable defaults for CUDA systems
            if self.cuda_available:
                logger.warning("Using default GPU memory estimates (8GB total, 6GB free)")
                return GPUMemoryInfo(
                    total=8.0,      # Assume 8GB GPU
                    used=2.0,       # Assume 2GB used
                    free=6.0,       # Assume 6GB free
                    allocated=1.0,  # Conservative estimate
                    reserved=2.0,   # Conservative estimate
                    utilization=0.25,  # 25% utilization
                    timestamp=time.time()
                )
            
            return GPUMemoryInfo(0, 0, 0, 0, 0, 0, time.time())
    
    def estimate_model_size(self, model_path: str) -> float:
        """
        Estimate model size in GB from file size.
        
        Args:
            model_path: Path to model file
            
        Returns:
            Estimated size in GB
        """
        try:
            import os
            file_size = os.path.getsize(model_path)
            # Model typically uses 1.2-1.5x file size when loaded
            return (file_size * 1.3) / (1024**3)
        except Exception as e:
            logger.warning(f"Failed to estimate model size for {model_path}: {e}")
            return 1.0  # Default estimate
    
    def calculate_optimal_layers(self, 
                                model_path: str, 
                                total_layers: int = None,
                                target_memory_usage: float = 0.7) -> int:
        """
        Calculate optimal number of GPU layers for a model.
        
        Args:
            model_path: Path to model file
            total_layers: Total number of layers in model
            target_memory_usage: Target memory usage ratio (0-1)
            
        Returns:
            Optimal number of GPU layers
        """
        if not self.cuda_available:
            return 0
        
        memory_info = self.get_memory_info()
        available_memory = memory_info.free
        
        # Estimate model size
        model_size = self.estimate_model_size(model_path)
        
        # Check if we have a profile for this model
        if model_path in self.model_profiles:
            profile = self.model_profiles[model_path]
            model_size = profile.actual_size or profile.estimated_size
        
        # Estimate total layers if not provided
        if total_layers is None:
            # Rough estimation based on model size
            # Typical transformer: ~100-200MB per layer
            total_layers = max(1, int(model_size * 1024 / 150))  # MB per layer
        
        # Calculate target memory for this model
        target_memory = available_memory * target_memory_usage
        
        if model_size > target_memory:
            # Model too large, calculate partial loading
            memory_per_layer = model_size / total_layers
            optimal_layers = int(target_memory / memory_per_layer)
        else:
            # Model fits, use all layers
            optimal_layers = total_layers
        
        # Ensure we don't exceed available memory
        optimal_layers = max(0, min(optimal_layers, total_layers))
        
        # Apply force_gpu_layers if specified
        if self.force_gpu_layers > 0 and self.cuda_available:
            forced_layers = min(self.force_gpu_layers, total_layers)
            if forced_layers > optimal_layers:
                logger.warning(
                    f"Force GPU layers enabled: using {forced_layers} layers instead of {optimal_layers} "
                    f"(may exceed memory estimates)"
                )
                optimal_layers = forced_layers
        
        logger.info(
            f"Optimal layers for {model_path}: {optimal_layers}/{total_layers} "
            f"(model: {model_size:.1f}GB, available: {available_memory:.1f}GB)"
        )
        
        return optimal_layers
    
    def get_gpu_memory_info(self) -> Dict[str, float]:
        """
        Get GPU memory information in the format expected by LLM class.
        Backward compatibility wrapper for get_memory_info.
        
        Returns:
            Dictionary with memory information in GB
        """
        memory_info = self.get_memory_info()
        return {
            "total": memory_info.total,
            "used": memory_info.used,
            "free": memory_info.free,
            "utilization": memory_info.utilization
        }
    
    def optimize_gpu_layers(self, total_layers: int, model_size_gb: float, model_path: str = None) -> int:
        """
        Backward compatibility wrapper for calculate_optimal_layers.
        
        Args:
            total_layers: Total number of layers in model
            model_size_gb: Model size in GB
            model_path: Optional path to model file
            
        Returns:
            Optimal number of GPU layers
        """
        # Use provided model path or a dummy path for the calculation
        path = model_path or "dummy_model"
        return self.calculate_optimal_layers(path, total_layers)
    
    def can_allocate(self, required_memory_gb: float) -> bool:
        """
        Check if we can allocate the required memory.
        
        Args:
            required_memory_gb: Required memory in GB
            
        Returns:
            True if allocation is possible
        """
        if not self.cuda_available:
            return False
        
        memory_info = self.get_memory_info()
        
        # Add safety margin
        available_with_margin = memory_info.free * 0.9
        
        return available_with_margin >= required_memory_gb
    
    def should_use_gpu(self, model_size_gb: float) -> bool:
        """
        Determine if GPU should be used for a model.
        
        Args:
            model_size_gb: Model size in GB
            
        Returns:
            True if GPU should be used
        """
        if not self.cuda_available:
            return False
        
        memory_info = self.get_memory_info()
        
        # Check if we have enough memory
        if not self.can_allocate(model_size_gb):
            self.metrics["fallback_to_cpu"] += 1
            return False
        
        # Check if GPU is not overloaded
        if memory_info.utilization > self.memory_threshold:
            logger.warning(f"GPU memory usage high: {memory_info.utilization:.1%}")
            return False
        
        return True
    
    def cleanup_memory(self, force: bool = False) -> bool:
        """
        Clean up GPU memory.
        
        Args:
            force: Force cleanup even if not needed
            
        Returns:
            True if cleanup was performed
        """
        if not self.cuda_available:
            return False
        
        memory_info = self.get_memory_info()
        
        if not force and memory_info.utilization < self.cleanup_threshold:
            return False
        
        try:
            import torch
            
            # Clear cache
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
            
            # Force garbage collection
            gc.collect()
            
            # Update metrics
            self.metrics["cleanup_operations"] += 1
            
            # Check improvement
            new_memory_info = self.get_memory_info()
            freed_memory = memory_info.used - new_memory_info.used
            
            logger.info(f"GPU memory cleanup: freed {freed_memory:.1f}GB")
            
            return True
            
        except Exception as e:
            logger.warning(f"GPU memory cleanup failed: {e}")
            return False
    
    def profile_model_loading(self, 
                            model_path: str, 
                            actual_size: float,
                            load_time: float,
                            gpu_layers: int) -> ModelMemoryProfile:
        """
        Profile model loading performance.
        
        Args:
            model_path: Path to model file
            actual_size: Actual memory usage in GB
            load_time: Time taken to load in seconds
            gpu_layers: Number of GPU layers used
            
        Returns:
            ModelMemoryProfile object
        """
        estimated_size = self.estimate_model_size(model_path)
        
        # Calculate performance score
        # Based on load time and memory efficiency
        time_score = max(0, 1 - (load_time / 60))  # Penalty for >60s load time
        memory_score = estimated_size / max(actual_size, 0.1)  # Efficiency ratio
        performance_score = (time_score + memory_score) / 2
        
        profile = ModelMemoryProfile(
            model_path=model_path,
            estimated_size=estimated_size,
            actual_size=actual_size,
            optimal_layers=gpu_layers,
            load_time=load_time,
            performance_score=performance_score
        )
        
        self.model_profiles[model_path] = profile
        
        logger.info(
            f"Model profile: {model_path} - "
            f"Size: {actual_size:.1f}GB, "
            f"Load time: {load_time:.1f}s, "
            f"Performance: {performance_score:.2f}"
        )
        
        return profile
    
    def start_monitoring(self):
        """Start background memory monitoring."""
        # Debug: Check if monitoring is disabled in config
        logger.info(f"GPU Manager start_monitoring called. Has config: {hasattr(self, 'config') and self.config is not None}")
        
        if hasattr(self, 'config') and self.config:
            logger.info(f"Config object found. Type: {type(self.config)}")
            
            # Check multiple possible config paths for monitoring disable
            monitoring_disabled = False
            
            # Check gpu.enable_monitoring
            if hasattr(self.config, 'gpu'):
                logger.info(f"Config has gpu section. Has enable_monitoring: {hasattr(self.config.gpu, 'enable_monitoring')}")
                if hasattr(self.config.gpu, 'enable_monitoring'):
                    logger.info(f"gpu.enable_monitoring value: {self.config.gpu.enable_monitoring}")
                    if not self.config.gpu.enable_monitoring:
                        monitoring_disabled = True
            
            # Check monitoring.gpu_monitoring  
            if hasattr(self.config, 'monitoring'):
                logger.info(f"Config has monitoring section. Has gpu_monitoring: {hasattr(self.config.monitoring, 'gpu_monitoring')}")
                if hasattr(self.config.monitoring, 'gpu_monitoring'):
                    logger.info(f"monitoring.gpu_monitoring value: {self.config.monitoring.gpu_monitoring}")
                    if not self.config.monitoring.gpu_monitoring:
                        monitoring_disabled = True
                        
            logger.info(f"Monitoring disabled by config: {monitoring_disabled}")
            if monitoring_disabled:
                logger.info("GPU memory monitoring disabled by configuration")
                return
        else:
            logger.info("No config found or config is None")
        
        if self._monitoring_active or not self.cuda_available:
            return
        
        self._monitoring_active = True
        self._monitor_thread = threading.Thread(target=self._monitor_loop, daemon=True)
        self._monitor_thread.start()
        
        logger.info("GPU memory monitoring started")
    
    def stop_monitoring(self):
        """Stop background memory monitoring."""
        self._monitoring_active = False
        if self._monitor_thread:
            self._monitor_thread.join(timeout=1.0)
        
        logger.info("GPU memory monitoring stopped")
    
    def _monitor_loop(self):
        """Background monitoring loop."""
        while self._monitoring_active:
            try:
                memory_info = self.get_memory_info()
                
                with self._lock:
                    self.memory_history.append(memory_info)
                    
                    # Keep only recent history (last hour)
                    cutoff_time = time.time() - 3600
                    self.memory_history = [
                        info for info in self.memory_history 
                        if info.timestamp > cutoff_time
                    ]
                
                # Update state based on memory usage
                if memory_info.utilization > 0.95:
                    self.current_state = GPUState.OVERLOADED
                    logger.warning(f"GPU overloaded: {memory_info.utilization:.1%} usage")
                elif memory_info.utilization > self.memory_threshold:
                    self.current_state = GPUState.BUSY
                else:
                    self.current_state = GPUState.AVAILABLE
                
                # Automatic cleanup if needed
                if memory_info.utilization > self.cleanup_threshold:
                    self.cleanup_memory()
                
                time.sleep(self.monitoring_interval)
                
            except Exception as e:
                logger.debug(f"Monitoring error: {e}")
                time.sleep(self.monitoring_interval)
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get comprehensive GPU statistics."""
        memory_info = self.get_memory_info()
        
        # Calculate average utilization from history
        with self._lock:
            if self.memory_history:
                avg_utilization = sum(info.utilization for info in self.memory_history) / len(self.memory_history)
                peak_utilization = max(info.utilization for info in self.memory_history)
            else:
                avg_utilization = memory_info.utilization
                peak_utilization = memory_info.utilization
        
        return {
            "cuda_available": self.cuda_available,
            "device_count": self.device_count,
            "current_state": self.current_state.value,
            "memory": {
                "current": memory_info.__dict__,
                "average_utilization": avg_utilization,
                "peak_utilization": peak_utilization,
            },
            "models": {
                "profiles_count": len(self.model_profiles),
                "total_estimated_size": sum(p.estimated_size for p in self.model_profiles.values()),
                "average_performance": sum(p.performance_score for p in self.model_profiles.values()) / max(len(self.model_profiles), 1),
            },
            "metrics": self.metrics.copy(),
            "thresholds": {
                "memory_threshold": self.memory_threshold,
                "cleanup_threshold": self.cleanup_threshold,
            }
        }
    
    def optimize_for_models(self, model_paths: List[str]) -> Dict[str, int]:
        """
        Optimize GPU allocation for multiple models.
        
        Args:
            model_paths: List of model paths to optimize for
            
        Returns:
            Dictionary mapping model paths to optimal GPU layers
        """
        if not self.cuda_available:
            return {path: 0 for path in model_paths}
        
        memory_info = self.get_memory_info()
        available_memory = memory_info.free * 0.8  # Safety margin
        
        # Estimate sizes and prioritize
        model_info = []
        for path in model_paths:
            size = self.estimate_model_size(path)
            priority = 1.0  # Default priority
            
            # Adjust priority based on model type
            if "text" in path.lower() or "llama" in path.lower():
                priority = 2.0  # Higher priority for text models
            elif "vision" in path.lower() or "llava" in path.lower():
                priority = 1.0  # Lower priority for vision models
            
            model_info.append((path, size, priority))
        
        # Sort by priority (higher first)
        model_info.sort(key=lambda x: x[2], reverse=True)
        
        # Allocate memory
        allocations = {}
        remaining_memory = available_memory
        
        for path, size, priority in model_info:
            if remaining_memory >= size:
                # Full allocation
                allocations[path] = -1  # Use all layers
                remaining_memory -= size
            elif remaining_memory > size * 0.3:
                # Partial allocation
                ratio = remaining_memory / size
                estimated_layers = int(50 * ratio)  # Rough estimate
                allocations[path] = max(1, estimated_layers)
                remaining_memory -= size * ratio
            else:
                # CPU only
                allocations[path] = 0
        
        logger.info(f"GPU allocation optimized for {len(model_paths)} models")
        return allocations
    
    def __del__(self):
        """Cleanup on destruction."""
        try:
            if hasattr(self, '_monitoring_active'):
                self.stop_monitoring()
        except Exception:
            # Ignore errors during cleanup
            pass


# Global GPU manager instance
gpu_manager = None


def get_gpu_manager(config=None) -> CUDAGPUManager:
    """Get the global GPU manager instance."""
    global gpu_manager
    if gpu_manager is None:
        gpu_manager = CUDAGPUManager()
        if config:
            gpu_manager.config = config
    return gpu_manager

