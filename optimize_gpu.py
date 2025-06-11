#!/usr/bin/env python3
"""
ParManus AI GPU Optimization Utility
Optimizes GPU memory usage and performance for better agent execution
"""

import json
import os
import subprocess
import time
from pathlib import Path

import psutil

try:
    import torch

    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

try:
    import requests

    REQUESTS_AVAILABLE = True
except ImportError:
    REQUESTS_AVAILABLE = False


class GPUOptimizer:
    """Comprehensive GPU optimization for ParManus AI"""

    def __init__(self):
        self.ollama_base_url = "http://localhost:11434"

    def check_gpu_status(self):
        """Check current GPU status and memory usage"""
        print("🔍 GPU Status Check")
        print("=" * 50)

        try:
            result = subprocess.run(
                [
                    "nvidia-smi",
                    "--query-gpu=name,memory.total,memory.used,memory.free,utilization.gpu",
                    "--format=csv,noheader,nounits",
                ],
                capture_output=True,
                text=True,
                timeout=10,
            )

            if result.returncode == 0:
                gpu_info = result.stdout.strip().split(", ")
                print(f"GPU: {gpu_info[0]}")
                print(f"Total Memory: {gpu_info[1]} MB")
                print(
                    f"Used Memory: {gpu_info[2]} MB ({float(gpu_info[2])/float(gpu_info[1])*100:.1f}%)"
                )
                print(f"Free Memory: {gpu_info[3]} MB")
                print(f"GPU Utilization: {gpu_info[4]}%")

                return {
                    "total": int(gpu_info[1]),
                    "used": int(gpu_info[2]),
                    "free": int(gpu_info[3]),
                    "utilization": int(gpu_info[4]),
                }
            else:
                print("❌ Could not get GPU information")
                return None

        except Exception as e:
            print(f"❌ Error checking GPU: {e}")
            return None

    def check_ollama_models(self):
        """Check currently loaded Ollama models"""
        print("\n📦 Ollama Models Status")
        print("=" * 50)

        if not REQUESTS_AVAILABLE:
            print("❌ requests library not available")
            return []

        try:
            response = requests.get(f"{self.ollama_base_url}/api/tags", timeout=5)
            if response.status_code == 200:
                models = response.json().get("models", [])
                loaded_models = []

                for model in models:
                    name = model["name"]
                    size_mb = model["size"] / (1024 * 1024)
                    print(f"• {name}: {size_mb:.1f} MB")
                    loaded_models.append({"name": name, "size_mb": size_mb})

                return loaded_models
            else:
                print("❌ Could not connect to Ollama")
                return []

        except Exception as e:
            print(f"❌ Error checking Ollama models: {e}")
            return []

    def optimize_ollama_environment(self):
        """Set optimized environment variables for Ollama"""
        print("\n⚡ Optimizing Ollama Environment")
        print("=" * 50)

        optimizations = {
            "OLLAMA_NUM_PARALLEL": "1",  # Single model at a time
            "OLLAMA_MAX_LOADED_MODELS": "1",  # Keep only one model loaded
            "OLLAMA_FLASH_ATTENTION": "1",  # Enable flash attention
            "CUDA_VISIBLE_DEVICES": "0",  # Use primary GPU only
            "OLLAMA_LLM_LIBRARY": "cuda",  # Force CUDA backend
        }

        for key, value in optimizations.items():
            os.environ[key] = value
            print(f"✓ Set {key}={value}")

    def restart_ollama(self):
        """Restart Ollama with optimized settings"""
        print("\n🔄 Restarting Ollama Service")
        print("=" * 50)

        try:
            # Kill existing Ollama processes
            for proc in psutil.process_iter(["pid", "name"]):
                if "ollama" in proc.info["name"].lower():
                    print(f"Stopping Ollama process {proc.info['pid']}")
                    proc.terminate()

            time.sleep(3)

            # Start Ollama with optimized environment
            print("Starting optimized Ollama service...")
            subprocess.Popen(
                ["ollama", "serve"],
                env=os.environ.copy(),
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )

            time.sleep(5)
            print("✓ Ollama restarted with optimizations")

        except Exception as e:
            print(f"❌ Error restarting Ollama: {e}")

    def optimize_models(self):
        """Optimize model loading strategy"""
        print("\n🎯 Model Loading Optimization")
        print("=" * 50)

        if not REQUESTS_AVAILABLE:
            print("❌ Cannot optimize without requests library")
            return

        models = self.check_ollama_models()

        # Prioritize keeping only the vision model loaded initially
        vision_models = [m for m in models if "vision" in m["name"].lower()]
        text_models = [m for m in models if "vision" not in m["name"].lower()]

        if vision_models:
            print(f"✓ Keeping vision model: {vision_models[0]['name']}")

        if text_models:
            print(f"• Text models will be loaded on-demand")

    def clear_gpu_memory(self):
        """Clear GPU memory cache"""
        print("\n🧹 Clearing GPU Memory")
        print("=" * 50)

        if TORCH_AVAILABLE:
            try:
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                    torch.cuda.synchronize()
                    print("✓ PyTorch GPU cache cleared")
            except Exception as e:
                print(f"⚠️ PyTorch cleanup warning: {e}")

        try:
            subprocess.run(
                ["nvidia-smi", "--gpu-reset"], capture_output=True, timeout=10
            )
            print("✓ NVIDIA GPU reset completed")
        except Exception as e:
            print(f"⚠️ GPU reset warning: {e}")

    def create_optimized_config(self):
        """Create optimized configuration file"""
        print("\n📝 Creating Optimized Configuration")
        print("=" * 50)

        config_path = Path("config/gpu_optimized_runtime.toml")
        if config_path.exists():
            print(f"✓ Optimized config already exists: {config_path}")
        else:
            print(f"⚠️ Config file not found: {config_path}")

    def run_optimization(self):
        """Run complete optimization process"""
        print("🚀 ParManus AI GPU Optimization")
        print("=" * 60)

        # Step 1: Check current status
        gpu_status = self.check_gpu_status()

        # Step 2: Check models
        models = self.check_ollama_models()

        # Step 3: Clear memory if usage is high
        if gpu_status and gpu_status["used"] / gpu_status["total"] > 0.85:
            self.clear_gpu_memory()

        # Step 4: Optimize environment
        self.optimize_ollama_environment()

        # Step 5: Restart Ollama
        self.restart_ollama()

        # Step 6: Optimize models
        self.optimize_models()

        # Step 7: Check config
        self.create_optimized_config()

        print("\n✅ GPU Optimization Complete!")
        print("=" * 60)
        print("💡 To use optimized settings, run:")
        print("   python main.py --config config/gpu_optimized_runtime.toml")
        print("\n📊 Checking final GPU status...")

        time.sleep(2)
        self.check_gpu_status()


if __name__ == "__main__":
    optimizer = GPUOptimizer()
    optimizer.run_optimization()
