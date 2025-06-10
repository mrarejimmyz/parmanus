#!/usr/bin/env python3
"""
Simple script to detect and handle Llama 3.2 Vision model architecture issues.
"""

import os
import sys
from pathlib import Path

# Add the project root to path
project_root = Path(__file__).parent.absolute()
sys.path.insert(0, str(project_root))


def test_model_loading():
    """Test if we can load the Llama 3.2 model and detect architecture issues."""

    print("🔍 Testing Llama 3.2 Vision Model Loading...")
    print("=" * 60)

    # Check if model file exists
    model_path = "models/Llama-3.2.gguf"
    if not os.path.exists(model_path):
        print(f"❌ Model file not found: {model_path}")
        return False

    print(f"✅ Model file found: {model_path}")

    # Try to load with llama-cpp-python
    try:
        from llama_cpp import Llama

        print(f"✅ llama-cpp-python imported successfully")

        # Try to load the model
        print(f"🔄 Attempting to load model...")

        model = Llama(
            model_path=model_path,
            n_ctx=512,  # Small context for testing
            n_gpu_layers=0,  # CPU only for testing
            verbose=False,
        )

        print(f"✅ Model loaded successfully!")
        return True

    except Exception as e:
        error_str = str(e)
        print(f"❌ Failed to load model: {error_str}")

        # Check if it's the mllama architecture issue
        if (
            "unknown model architecture" in error_str.lower()
            and "mllama" in error_str.lower()
        ):
            print("\n" + "🚨" * 20)
            print("🔍 DETECTED: mllama architecture not supported!")
            print("🚨" * 20)
            print()

            print("📋 YOUR MODEL DETAILS:")
            print(f"   • Model: Llama 3.2 11B Vision")
            print(f"   • Architecture: mllama")
            print(f"   • Status: ❌ Not supported by llama-cpp-python 0.3.9")
            print()

            print("💡 RECOMMENDED SOLUTIONS:")
            print()
            print("1. 🚀 USE OLLAMA (EASIEST):")
            print("   • Download Ollama from: https://ollama.ai")
            print("   • Run: ollama pull llama3.2-vision:11b")
            print("   • Update your config to use ollama backend")
            print()
            print("2. 🔄 WAIT FOR UPDATES:")
            print("   • Check for llama-cpp-python updates regularly")
            print("   • Monitor: https://github.com/abetlen/llama-cpp-python")
            print()
            print("3. 📝 USE TEXT-ONLY VERSION:")
            print("   • Download a text-only Llama 3.2 11B model")
            print("   • Vision features will be disabled")
            print()

            return False
        else:
            print(f"🔍 Different error: {error_str}")
            return False


def provide_next_steps():
    """Provide clear next steps for the user."""

    print("\n" + "🎯" * 20)
    print("IMMEDIATE NEXT STEPS:")
    print("🎯" * 20)
    print()

    print("Option A - Install Ollama (Recommended):")
    print("1. Download from https://ollama.ai/download")
    print("2. Install Ollama")
    print("3. Open terminal and run: ollama pull llama3.2-vision:11b")
    print("4. Update your config.toml:")
    print("   [llm]")
    print('   model = "llama3.2-vision:11b"')
    print('   model_path = "ollama"')
    print()

    print("Option B - Get a Compatible Model:")
    print("1. Download a text-only Llama 3.2 11B model")
    print("2. Or wait for llama-cpp-python to support mllama")
    print()

    print("Option C - Use Different AI:")
    print("1. Configure to use OpenAI, Anthropic, or other API")
    print("2. Update config.toml with API settings")


if __name__ == "__main__":
    print("🤖 ParManus AI - Model Compatibility Checker")
    print("=" * 50)
    print()

    success = test_model_loading()

    if not success:
        provide_next_steps()

    print("\n" + "✅" * 20)
    print("Model compatibility check completed!")
    print("✅" * 20)
