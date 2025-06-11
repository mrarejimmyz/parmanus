"""
ParManus AI GPU Dashboard
Real-time GPU monitoring and optimization for your agent
"""

import asyncio
import json
import subprocess
import time
from datetime import datetime
from pathlib import Path

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


class GPUDashboard:
    """Real-time GPU monitoring dashboard for ParManus AI"""

    def __init__(self):
        self.ollama_url = "http://localhost:11434"
        self.monitoring_active = False

    def get_gpu_metrics(self):
        """Get comprehensive GPU metrics"""
        try:
            result = subprocess.run(
                [
                    "nvidia-smi",
                    "--query-gpu=name,memory.total,memory.used,memory.free,utilization.gpu,utilization.memory,temperature.gpu,power.draw",
                    "--format=csv,noheader,nounits",
                ],
                capture_output=True,
                text=True,
                timeout=5,
            )

            if result.returncode == 0:
                metrics = result.stdout.strip().split(", ")
                return {
                    "name": metrics[0],
                    "memory_total_mb": int(metrics[1]),
                    "memory_used_mb": int(metrics[2]),
                    "memory_free_mb": int(metrics[3]),
                    "gpu_utilization": int(metrics[4]),
                    "memory_utilization": int(metrics[5]),
                    "temperature": int(metrics[6]),
                    "power_draw": float(metrics[7]),
                    "memory_usage_percent": round(
                        int(metrics[2]) / int(metrics[1]) * 100, 1
                    ),
                    "timestamp": datetime.now().strftime("%H:%M:%S"),
                }
        except Exception as e:
            return {"error": str(e)}

    def get_ollama_status(self):
        """Get Ollama model status"""
        if not REQUESTS_AVAILABLE:
            return {"error": "requests not available"}

        try:
            response = requests.get(f"{self.ollama_url}/api/tags", timeout=3)
            if response.status_code == 200:
                models = response.json().get("models", [])
                return {
                    "models_count": len(models),
                    "models": [
                        {
                            "name": m["name"],
                            "size_gb": round(m["size"] / (1024**3), 2),
                            "modified": m.get("modified_at", "unknown"),
                        }
                        for m in models
                    ],
                }
        except Exception as e:
            return {"error": str(e)}

    def get_torch_metrics(self):
        """Get PyTorch GPU metrics"""
        if not TORCH_AVAILABLE or not torch.cuda.is_available():
            return {"available": False}

        try:
            return {
                "available": True,
                "device_count": torch.cuda.device_count(),
                "current_device": torch.cuda.current_device(),
                "memory_allocated_gb": round(
                    torch.cuda.memory_allocated() / (1024**3), 2
                ),
                "memory_reserved_gb": round(
                    torch.cuda.memory_reserved() / (1024**3), 2
                ),
                "memory_cached_gb": round(torch.cuda.memory_cached() / (1024**3), 2),
            }
        except Exception as e:
            return {"error": str(e)}

    def display_dashboard(self):
        """Display real-time dashboard"""
        gpu_metrics = self.get_gpu_metrics()
        ollama_status = self.get_ollama_status()
        torch_metrics = self.get_torch_metrics()

        # Clear screen (Windows)
        subprocess.run(["cls"], shell=True)

        print("🚀 ParManus AI - GPU Performance Dashboard")
        print("=" * 60)
        print(f"⏰ Last Update: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print()

        # GPU Status
        if "error" not in gpu_metrics:
            status_color = (
                "🟢"
                if gpu_metrics["memory_usage_percent"] < 75
                else "🟡" if gpu_metrics["memory_usage_percent"] < 90 else "🔴"
            )
            print(f"🎮 GPU Status {status_color}")
            print(f"   Name: {gpu_metrics['name']}")
            print(
                f"   Memory: {gpu_metrics['memory_used_mb']:,}MB / {gpu_metrics['memory_total_mb']:,}MB ({gpu_metrics['memory_usage_percent']}%)"
            )
            print(
                f"   Utilization: GPU {gpu_metrics['gpu_utilization']}% | Memory {gpu_metrics['memory_utilization']}%"
            )
            print(f"   Temperature: {gpu_metrics['temperature']}°C")
            print(f"   Power: {gpu_metrics['power_draw']}W")
        else:
            print(f"🎮 GPU Status: ❌ {gpu_metrics['error']}")

        print()

        # Ollama Status
        if "error" not in ollama_status:
            print(f"🤖 Ollama Models ({ollama_status['models_count']} loaded)")
            for model in ollama_status["models"]:
                size_indicator = (
                    "🟢"
                    if model["size_gb"] < 5
                    else "🟡" if model["size_gb"] < 8 else "🔴"
                )
                print(f"   {size_indicator} {model['name']}: {model['size_gb']}GB")
        else:
            print(f"🤖 Ollama Status: ❌ {ollama_status['error']}")

        print()

        # PyTorch Status
        if torch_metrics.get("available"):
            print(f"🔥 PyTorch CUDA")
            print(
                f"   Devices: {torch_metrics['device_count']} | Current: {torch_metrics['current_device']}"
            )
            print(f"   Allocated: {torch_metrics['memory_allocated_gb']}GB")
            print(f"   Reserved: {torch_metrics['memory_reserved_gb']}GB")
            print(f"   Cached: {torch_metrics['memory_cached_gb']}GB")
        else:
            print(f"🔥 PyTorch CUDA: ❌ Not available")

        print()

        # Optimization Recommendations
        print("💡 Optimization Status")
        recommendations = []

        if "error" not in gpu_metrics:
            if gpu_metrics["memory_usage_percent"] > 85:
                recommendations.append(
                    "🔴 High memory usage - consider unloading unused models"
                )
            elif gpu_metrics["memory_usage_percent"] < 50:
                recommendations.append(
                    "🟢 Memory usage optimal - can load additional models"
                )

            if gpu_metrics["temperature"] > 80:
                recommendations.append("🔴 High temperature - check cooling")
            elif gpu_metrics["temperature"] < 70:
                recommendations.append("🟢 Temperature optimal")

        if "error" not in ollama_status and ollama_status["models_count"] > 2:
            recommendations.append(
                "🟡 Multiple models loaded - consider selective loading"
            )

        if not recommendations:
            recommendations.append("🟢 All systems optimal")

        for rec in recommendations:
            print(f"   {rec}")

        print()
        print("📊 Commands: [R]efresh | [O]ptimize | [Q]uit")

    async def start_monitoring(self, interval=5):
        """Start continuous monitoring"""
        print("Starting GPU monitoring... Press 'q' to quit")
        self.monitoring_active = True

        while self.monitoring_active:
            self.display_dashboard()

            # Non-blocking input check
            await asyncio.sleep(interval)

    def optimize_now(self):
        """Run optimization commands"""
        print("\n🔧 Running optimization...")

        # Clear PyTorch cache
        if TORCH_AVAILABLE and torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
            print("✓ PyTorch cache cleared")

        # Set Ollama optimizations
        import os

        os.environ["OLLAMA_MAX_LOADED_MODELS"] = "1"
        os.environ["OLLAMA_FLASH_ATTENTION"] = "1"
        print("✓ Ollama environment optimized")

        print("✅ Optimization complete")
        time.sleep(2)


async def main():
    """Main dashboard function"""
    dashboard = GPUDashboard()

    print("🚀 ParManus AI GPU Dashboard")
    print("Loading...")

    try:
        while True:
            dashboard.display_dashboard()

            print(
                "\nEnter command (r=refresh, o=optimize, m=monitor, q=quit): ", end=""
            )
            choice = input().lower().strip()

            if choice == "q":
                break
            elif choice == "r":
                continue
            elif choice == "o":
                dashboard.optimize_now()
            elif choice == "m":
                await dashboard.start_monitoring()
            else:
                print("Invalid choice")
                time.sleep(1)

    except KeyboardInterrupt:
        print("\n👋 Dashboard closed")


if __name__ == "__main__":
    asyncio.run(main())
