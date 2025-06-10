#!/usr/bin/env python3
"""
Detailed model architecture checker with verbose error capture.
"""

import os
import sys
from pathlib import Path

# Add the project root to path
project_root = Path(__file__).parent.absolute()
sys.path.insert(0, str(project_root))


def test_detailed_model_loading():
    """Test model loading with detailed error capture."""

    print("🔍 Detailed Model Loading Test...")
    print("=" * 60)

    model_path = "models/Llama-3.2.gguf"

    try:
        import llama_cpp

        print(f"✅ llama_cpp version: {llama_cpp.__version__}")

        # Enable verbose mode to capture more details
        print(f"🔄 Loading model with verbose output...")

        # Try to create Llama instance with verbose=True
        model = llama_cpp.Llama(
            model_path=model_path,
            n_ctx=512,
            n_gpu_layers=0,
            verbose=True,  # This should show more detailed error info
            use_mmap=True,
        )

        print(f"✅ Model loaded successfully!")
        return True

    except Exception as e:
        print(f"\n❌ DETAILED ERROR INFORMATION:")
        print(f"Error Type: {type(e).__name__}")
        print(f"Error Message: {str(e)}")
        print(f"Error Args: {e.args}")

        # Check for specific error patterns
        error_str = str(e).lower()

        if "mllama" in error_str:
            print(f"\n🎯 DETECTED: mllama architecture issue!")
            show_mllama_solutions()
            return False
        elif "architecture" in error_str:
            print(f"\n🎯 DETECTED: Architecture compatibility issue!")
            print(f"Full error: {str(e)}")
            show_mllama_solutions()
            return False
        else:
            print(f"\n❓ Different error type detected")
            return False


def show_mllama_solutions():
    """Show comprehensive solutions for mllama architecture issues."""

    print("\n" + "🔧" * 50)
    print("LLAMA 3.2 VISION COMPATIBILITY SOLUTIONS")
    print("🔧" * 50)
    print()

    print("🚨 ISSUE: Your Llama 3.2 Vision model uses 'mllama' architecture")
    print("   which is NOT supported by llama-cpp-python 0.3.9")
    print()

    print("💡 SOLUTION 1 - Use Ollama (RECOMMENDED):")
    print("   ✅ Pros: Easy setup, native Llama 3.2 Vision support")
    print("   📋 Steps:")
    print("      1. Download Ollama: https://ollama.ai/download")
    print("      2. Install and restart terminal")
    print("      3. Run: ollama pull llama3.2-vision:11b")
    print('      4. Test: ollama run llama3.2-vision:11b "Hello"')
    print("      5. Update config.toml:")
    print("         [llm]")
    print('         model = "llama3.2-vision:11b"')
    print('         model_path = "ollama"')
    print()

    print("💡 SOLUTION 2 - Get Text-Only Model:")
    print("   ✅ Pros: Works with current setup")
    print("   ❌ Cons: No vision capabilities")
    print("   📋 Steps:")
    print("      1. Download text-only Llama 3.2 11B model")
    print("      2. Replace Llama-3.2.gguf with text-only version")
    print("      3. Disable vision in config.toml")
    print()

    print("💡 SOLUTION 3 - Use Cloud AI:")
    print("   ✅ Pros: Most capable, latest models")
    print("   ❌ Cons: Requires API key, costs money")
    print("   📋 Options:")
    print("      • OpenAI GPT-4 Vision")
    print("      • Anthropic Claude 3.5 Sonnet")
    print("      • Google Gemini Pro Vision")
    print()

    print("⏳ SOLUTION 4 - Wait for Update:")
    print("   📋 Monitor: https://github.com/abetlen/llama-cpp-python")
    print("   🔍 Look for: mllama architecture support")


if __name__ == "__main__":
    print("🤖 ParManus AI - Detailed Model Compatibility Checker")
    print("=" * 60)
    print()

    test_detailed_model_loading()

    print("\n" + "🎯" * 20)
    print("NEXT STEPS:")
    print("🎯" * 20)
    print("1. Choose one of the solutions above")
    print("2. Follow the steps for your chosen solution")
    print("3. Re-run main.py after implementing the solution")
    print("4. If you need help, check the documentation or ask for assistance")
    print()
    print("✅ Compatibility check completed!")
