"""
Test script to validate ParManus AI system functionality.
"""

import sys
import os

def test_imports():
    """Test all required imports."""
    print("🧪 Testing imports...")
    
    # Core Python modules
    try:
        import asyncio
        import json
        import time
        from pathlib import Path
        print("✅ Core Python modules: OK")
    except ImportError as e:
        print(f"❌ Core Python modules: {e}")
        return False
    
    # Pydantic
    try:
        from pydantic import BaseModel, Field
        print("✅ Pydantic: OK")
    except ImportError as e:
        print(f"❌ Pydantic: {e}")
        return False
    
    # TOML parsing
    try:
        import tomllib
        print("✅ TOML parser (tomllib): OK")
    except ImportError:
        try:
            import tomli
            print("✅ TOML parser (tomli): OK")
        except ImportError as e:
            print(f"❌ TOML parser: {e}")
            return False
    
    # Optional: llama-cpp-python
    try:
        from llama_cpp import Llama
        print("✅ llama-cpp-python: OK")
    except ImportError as e:
        print(f"⚠️ llama-cpp-python: {e} (optional for local models)")
    
    # Optional: OpenAI
    try:
        from openai import AsyncOpenAI
        print("✅ OpenAI: OK")
    except ImportError as e:
        print(f"⚠️ OpenAI: {e} (optional for Ollama)")
    
    # Optional: Loguru
    try:
        from loguru import logger
        print("✅ Loguru: OK")
    except ImportError as e:
        print(f"⚠️ Loguru: {e} (will use standard logging)")
    
    return True

def test_config_loading():
    """Test configuration loading."""
    print("\n🧪 Testing configuration loading...")
    
    try:
        # Test if config file exists
        config_path = "config/config.toml"
        if os.path.exists(config_path):
            print(f"✅ Config file found: {config_path}")
            
            # Test loading
            with open(config_path, "rb") as f:
                try:
                    import tomllib
                    config_dict = tomllib.load(f)
                except ImportError:
                    import tomli
                    config_dict = tomli.load(f)
            
            print("✅ Config loading: OK")
            print(f"   API Type: {config_dict.get('llm', {}).get('api_type', 'not set')}")
            print(f"   Model: {config_dict.get('llm', {}).get('model', 'not set')}")
            return True
        else:
            print(f"⚠️ Config file not found: {config_path}")
            return True  # Not critical
    except Exception as e:
        print(f"❌ Config loading: {e}")
        return False

def test_model_files():
    """Test if model files exist."""
    print("\n🧪 Testing model files...")
    
    models_dir = "models"
    if not os.path.exists(models_dir):
        print(f"⚠️ Models directory not found: {models_dir}")
        return True  # Not critical for testing
    
    # Check for common model files
    model_files = [
        "Llama-3.2-11B-Vision-Instruct.Q4_K_M.gguf",
        "llava-1.6-mistral-7b-gguf/ggml-model-q4_k.gguf",
        "llava-1.6-mistral-7b-gguf/mmproj-model-f16.gguf"
    ]
    
    found_models = []
    for model_file in model_files:
        model_path = os.path.join(models_dir, model_file)
        if os.path.exists(model_path):
            found_models.append(model_file)
            print(f"✅ Model found: {model_file}")
        else:
            print(f"⚠️ Model not found: {model_file}")
    
    if found_models:
        print(f"✅ Found {len(found_models)} model file(s)")
    else:
        print("⚠️ No model files found (will need external models)")
    
    return True

def test_workspace():
    """Test workspace creation."""
    print("\n🧪 Testing workspace...")
    
    try:
        workspace_dir = "./workspace"
        os.makedirs(workspace_dir, exist_ok=True)
        
        if os.path.exists(workspace_dir):
            print(f"✅ Workspace created: {workspace_dir}")
            return True
        else:
            print(f"❌ Failed to create workspace: {workspace_dir}")
            return False
    except Exception as e:
        print(f"❌ Workspace creation: {e}")
        return False

def test_main_script():
    """Test main script syntax."""
    print("\n🧪 Testing main script...")
    
    try:
        # Test if main.py can be imported (syntax check)
        import main
        print("✅ Main script syntax: OK")
        return True
    except SyntaxError as e:
        print(f"❌ Main script syntax error: {e}")
        return False
    except ImportError as e:
        print(f"⚠️ Main script import issues: {e} (may be dependency-related)")
        return True  # Syntax is OK, just missing deps
    except Exception as e:
        print(f"❌ Main script error: {e}")
        return False

def main():
    """Run all tests."""
    print("🚀 ParManus AI System Test Suite")
    print("=" * 50)
    
    tests = [
        test_imports,
        test_config_loading,
        test_model_files,
        test_workspace,
        test_main_script,
    ]
    
    passed = 0
    total = len(tests)
    
    for test in tests:
        if test():
            passed += 1
    
    print("\n" + "=" * 50)
    print(f"🏁 Test Results: {passed}/{total} passed")
    
    if passed == total:
        print("🎉 All tests passed! System is ready.")
        return 0
    else:
        print("⚠️ Some tests failed. Check the issues above.")
        return 1

if __name__ == "__main__":
    sys.exit(main())

