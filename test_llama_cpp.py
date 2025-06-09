#!/usr/bin/env python3
"""Test llama-cpp-python CUDA availability"""

import os
import sys


def test_llama_cpp():
    print("=== llama-cpp-python CUDA Test ===")

    try:
        from llama_cpp import llama_cpp

        print(f"✅ llama-cpp-python imported successfully")

        # Check CUDA compilation
        has_cublas = hasattr(llama_cpp, "_LLAMA_CUBLAS")
        if has_cublas:
            cublas_enabled = getattr(llama_cpp, "_LLAMA_CUBLAS", False)
            print(f"CUDA support compiled: {cublas_enabled}")
        else:
            print("❌ No CUDA support detected")

        # Try to get device count
        try:
            from llama_cpp import Llama

            # Check if there's a method to get device count
            if hasattr(Llama, "get_cuda_device_count"):
                device_count = Llama.get_cuda_device_count()
                print(f"CUDA device count: {device_count}")
            else:
                print("No device count method available")
        except Exception as e:
            print(f"Error checking device count: {e}")

    except ImportError as e:
        print(f"❌ Failed to import llama-cpp-python: {e}")
    except Exception as e:
        print(f"❌ Error testing llama-cpp-python: {e}")


if __name__ == "__main__":
    test_llama_cpp()
