#!/usr/bin/env python3
"""Test PyTorch CUDA availability"""

import os
import sys


def test_torch():
    print("=== PyTorch CUDA Test ===")

    try:
        import torch

        print(f"✅ PyTorch imported successfully")
        print(f"PyTorch version: {torch.__version__}")

        # Check CUDA availability
        cuda_available = torch.cuda.is_available()
        print(f"CUDA available: {cuda_available}")

        if cuda_available:
            print(f"CUDA device count: {torch.cuda.device_count()}")
            print(f"CUDA version: {torch.version.cuda}")
            print(f"Current device: {torch.cuda.current_device()}")
            print(f"Device name: {torch.cuda.get_device_name(0)}")

            # Test GPU memory
            device_props = torch.cuda.get_device_properties(0)
            total_memory = device_props.total_memory / (1024**3)  # GB
            print(f"GPU memory: {total_memory:.1f} GB")

            # Test simple tensor operation
            x = torch.randn(1000, 1000).cuda()
            y = torch.randn(1000, 1000).cuda()
            z = x @ y
            print(f"✅ GPU tensor operation successful")

        else:
            print("❌ CUDA not available")

    except ImportError as e:
        print(f"❌ Failed to import PyTorch: {e}")
    except Exception as e:
        print(f"❌ Error testing PyTorch: {e}")


if __name__ == "__main__":
    test_torch()
