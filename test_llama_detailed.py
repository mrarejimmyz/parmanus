#!/usr/bin/env python3
"""Detailed test for llama-cpp-python CUDA support"""

import os
import sys


def detailed_llama_cpp_test():
    print("=== Detailed llama-cpp-python CUDA Test ===")

    try:
        import llama_cpp

        print(f"✅ llama_cpp imported successfully")
        print(f"llama_cpp version: {llama_cpp.__version__}")

        # Check all available attributes
        print("\nAvailable llama_cpp attributes:")
        for attr in dir(llama_cpp):
            if "cuda" in attr.lower() or "gpu" in attr.lower():
                print(f"  - {attr}")

        # Check llama_cpp submodule
        try:
            from llama_cpp import llama_cpp as llama_core

            print(f"\nllama_cpp.llama_cpp module attributes:")
            for attr in dir(llama_core):
                if any(
                    keyword in attr.lower()
                    for keyword in ["cuda", "gpu", "cublas", "device"]
                ):
                    value = getattr(llama_core, attr, None)
                    print(f"  - {attr}: {value}")
        except Exception as e:
            print(f"Error accessing llama_cpp.llama_cpp: {e}")

        # Try to load a simple model to test GPU usage
        try:
            from llama_cpp import Llama

            print(f"\n✅ Llama class imported successfully")

            # Check if model exists
            model_path = "models/llama-jb.gguf"
            if os.path.exists(model_path):
                print(f"Found model at: {model_path}")
                print("Attempting to load with GPU layers...")

                # Try loading with verbose output
                llm = Llama(
                    model_path=model_path,
                    n_gpu_layers=1,  # Just 1 layer to test
                    verbose=True,
                    n_ctx=512,
                )
                print("✅ Model loaded successfully with GPU layers!")

                # Clean up
                del llm
            else:
                print(f"❌ Model not found at: {model_path}")

        except Exception as e:
            print(f"❌ Error loading model: {e}")

    except ImportError as e:
        print(f"❌ Failed to import llama_cpp: {e}")
    except Exception as e:
        print(f"❌ Error in detailed test: {e}")


if __name__ == "__main__":
    detailed_llama_cpp_test()
