#!/usr/bin/env python3
"""
ParManus AI - Quick Setup Script
Automates the installation and configuration process
"""

import os
import platform
import subprocess
import sys
from pathlib import Path


def run_command(command, description):
    """Run a command and handle errors"""
    print(f"🔧 {description}...")
    try:
        result = subprocess.run(command, shell=True, capture_output=True, text=True)
        if result.returncode == 0:
            print(f"✅ {description} completed successfully")
            return True
        else:
            print(f"❌ {description} failed: {result.stderr}")
            return False
    except Exception as e:
        print(f"❌ {description} failed: {str(e)}")
        return False


def check_python_version():
    """Check if Python version is compatible"""
    version = sys.version_info
    if version.major == 3 and version.minor >= 11:
        print(
            f"✅ Python {version.major}.{version.minor}.{version.micro} is compatible"
        )
        return True
    else:
        print(
            f"❌ Python {version.major}.{version.minor}.{version.micro} is not compatible"
        )
        print("   Please install Python 3.11 or higher")
        return False


def check_ollama():
    """Check if Ollama is installed and running"""
    print("🔍 Checking Ollama installation...")

    # Check if ollama command exists
    result = subprocess.run("ollama --version", shell=True, capture_output=True)
    if result.returncode != 0:
        print("❌ Ollama not found. Please install from https://ollama.ai")
        return False

    print("✅ Ollama is installed")

    # Check if ollama is running
    result = subprocess.run("ollama list", shell=True, capture_output=True)
    if result.returncode != 0:
        print("🔄 Starting Ollama service...")
        subprocess.Popen("ollama serve", shell=True)

    return True


def install_dependencies():
    """Install Python dependencies"""
    requirements = [
        "ollama",
        "pydantic",
        "asyncio",
        "psutil",
        "pyautogui",
        "pygetwindow",
        "pyperclip",
        "opencv-python",
        "screeninfo",
        "mss",
        "winshell",
        "numpy",
        "pillow",
        "requests",
        "browser-use",
    ]

    print("📦 Installing Python dependencies...")

    # Install from requirements.txt if it exists
    if Path("requirements.txt").exists():
        success = run_command(
            "pip install -r requirements.txt", "Installing from requirements.txt"
        )
        if success:
            return True

    # Install individual packages
    for package in requirements:
        success = run_command(f"pip install {package}", f"Installing {package}")
        if not success:
            print(f"⚠️ Warning: Failed to install {package}")

    return True


def download_model():
    """Download the Llama model"""
    print("🤖 Downloading AI model (this may take a while)...")
    models = ["llama3.2-vision:11b", "llama3.2:11b"]

    for model in models:
        success = run_command(f"ollama pull {model}", f"Downloading {model}")
        if success:
            print(f"✅ Model {model} downloaded successfully")
            return True

    print("❌ Failed to download any models")
    return False


def create_config():
    """Create configuration file"""
    config_content = """[llm]
model = "llama3.2-vision:11b"
base_url = "http://localhost:11434"
temperature = 0.1
max_tokens = 2048

[computer_control]
safety_mode = true
screenshot_quality = "high"
mouse_speed = "normal"
keyboard_delay = 0.1

[automation]
max_workflow_steps = 50
error_recovery = true
timeout_seconds = 30

[system]
workspace_root = "%s"
log_level = "INFO"
""" % str(
        Path.cwd()
    ).replace(
        "\\", "\\\\"
    )

    config_path = Path("config.toml")
    if not config_path.exists():
        print("📝 Creating configuration file...")
        with open(config_path, "w") as f:
            f.write(config_content)
        print("✅ Configuration file created")
    else:
        print("ℹ️ Configuration file already exists")


def run_tests():
    """Run validation tests"""
    print("🧪 Running validation tests...")

    test_files = [
        "test_computer_control_actions_async.py",
        "test_action_names_validation.py",
    ]

    all_passed = True
    for test_file in test_files:
        if Path(test_file).exists():
            success = run_command(f"python {test_file}", f"Running {test_file}")
            if not success:
                all_passed = False
        else:
            print(f"⚠️ Test file {test_file} not found")

    return all_passed


def main():
    """Main setup function"""
    print("🚀 ParManus AI - Quick Setup Script")
    print("=" * 50)

    # Check system requirements
    if platform.system() != "Windows":
        print("⚠️ Warning: This system is optimized for Windows")

    # Check Python version
    if not check_python_version():
        return False

    # Check Ollama
    if not check_ollama():
        print(
            "⚠️ Please install Ollama from https://ollama.ai and run this script again"
        )
        return False

    # Install dependencies
    install_dependencies()

    # Create configuration
    create_config()

    # Download model
    download_model()

    # Run tests
    print("\n" + "=" * 50)
    print("🧪 RUNNING VALIDATION TESTS")
    print("=" * 50)

    tests_passed = run_tests()

    # Final summary
    print("\n" + "=" * 50)
    print("📋 SETUP SUMMARY")
    print("=" * 50)

    if tests_passed:
        print("🎉 SETUP COMPLETED SUCCESSFULLY!")
        print("\n🚀 You can now use ParManus AI:")
        print("   python main.py")
        print(
            "\n📚 Check INSTALLATION_AND_USAGE_GUIDE.md for detailed usage instructions"
        )
        return True
    else:
        print("⚠️ Setup completed with warnings")
        print("   Some tests failed - check the output above")
        print("   ParManus AI should still work, but some features may be limited")
        return False


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
