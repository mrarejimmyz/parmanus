# 🤖 ParManus AI - Autonomous Computer Control Agent

[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/)
[![Windows](https://img.shields.io/badge/platform-Windows-lightgrey.svg)](https://www.microsoft.com/windows)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

> **Advanced AI Agent with Full Computer Control Capabilities**
> ParManus AI is a powerful autonomous agent that can control your entire computer system using advanced vision and automation capabilities powered by Llama 3.2 11B.

## 🌟 Features

### 🖥️ **Complete Computer Control**
- **Screen Capture & Analysis** - Take screenshots and analyze visual content
- **Precise Mouse Control** - Click, move, drag, scroll at any coordinates
- **Advanced Keyboard Control** - Type text, send keys, execute shortcuts
- **Application Management** - Launch, close, and control any application
- **Window Management** - Move, resize, minimize, maximize, focus windows
- **Computer Vision** - Find and interact with UI elements using image recognition
- **System Information** - Monitor system performance and hardware details

### 🤖 **AI-Powered Automation**
- **Natural Language Commands** - Control your computer using plain English
- **Visual Understanding** - AI can see and understand your screen content
- **Complex Workflows** - Execute multi-step automation sequences
- **Error Recovery** - Intelligent error handling and recovery mechanisms
- **Safety Features** - Built-in safeguards to prevent system damage

### 🔧 **Advanced Capabilities**
- **29 Computer Control Actions** - Comprehensive automation toolkit
- **Clipboard Operations** - Read and write clipboard content
- **Process Management** - List, monitor, and control running processes
- **Command Execution** - Execute safe system commands
- **Batch Operations** - Process multiple files and perform bulk actions

## 📋 Requirements

### System Requirements
- **Operating System**: Windows 10/11 (64-bit)
- **Python**: 3.11 or higher
- **RAM**: 8GB minimum (16GB recommended for optimal performance)
- **GPU**: NVIDIA GPU with CUDA support (optional but recommended)
- **Storage**: 5GB free space

### Software Dependencies
- **Ollama** (for local AI model hosting)
- **Visual Studio Code** (recommended IDE)
- **Git** (for version control)

## 🚀 Installation

### Step 1: Clone the Repository
```powershell
git clone https://github.com/yourusername/ParManusAI.git
cd ParManusAI
```

### Step 2: Set Up Python Environment
```powershell
# Create virtual environment
python -m venv venv

# Activate virtual environment
.\venv\Scripts\Activate.ps1

# Upgrade pip
python -m pip install --upgrade pip
```

### Step 3: Install Dependencies
```powershell
# Install all required packages
pip install -r requirements.txt

# Install additional computer control dependencies
pip install pyautogui pygetwindow pyperclip opencv-python screeninfo mss winshell
```

### Step 4: Install and Configure Ollama
```powershell
# Download and install Ollama from https://ollama.ai
# Or use the provided setup script
python setup_ollama.py

# Pull the required model
ollama pull llama3.2-vision:11b
```

### Step 5: Configure the System
```powershell
# Copy and edit configuration
cp config/config.toml.example config.toml

# Edit config.toml with your preferences
notepad config.toml
```

### Step 6: Verify Installation
```powershell
# Run the comprehensive test suite
python test_computer_control_actions_async.py

# Run action names validation
python test_action_names_validation.py

# Test integration
python test_manus_computer_control_integration.py
```

## ⚙️ Configuration

### Basic Configuration (`config.toml`)
```toml
[llm]
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
workspace_root = "f:\\parmanu\\ParManusAI"
log_level = "INFO"
```

### Advanced Configuration Options
- **Safety Mode**: Enables additional confirmations for destructive actions
- **Screenshot Quality**: Controls image capture resolution and compression
- **Mouse Speed**: Adjusts mouse movement speed (slow/normal/fast)
- **Error Recovery**: Enables automatic error detection and recovery
- **Timeout Settings**: Configures operation timeouts

## 🎯 Usage

### Interactive Mode
Start ParManus AI in interactive mode for natural conversation:

```powershell
python main.py
```

**Example Interactions:**
```
User: Take a screenshot and tell me what's on my screen
Agent: I'll take a screenshot and analyze what's visible...

User: Open calculator and calculate 25 * 47
Agent: I'll launch the calculator app and perform that calculation...

User: Find all PDF files in my Downloads folder and organize them
Agent: I'll search for PDF files and help organize them...
```

### Command Line Mode
Execute specific tasks directly:

```powershell
# Take a screenshot
python main.py --prompt "Take a screenshot of my desktop"

# Automate application tasks
python main.py --prompt "Open notepad and write a hello world message"

# System administration
python main.py --prompt "Show me system information and running processes"
```

### Specific Agent Modes
```powershell
# Use computer control agent specifically
python main.py --agent manus --prompt "Control my computer to organize files"

# Browser automation
python main.py --agent browser --prompt "Navigate to google.com and search for AI news"

# File operations
python main.py --agent file --prompt "Analyze the contents of my documents folder"
```

## 🔧 Computer Control Actions

### 📸 Screen Capture
```python
# Take full screenshot
computer_control(action="screenshot")

# Capture specific region
computer_control(action="screenshot_region", x=100, y=100, width=500, height=300)
```

### 🖱️ Mouse Control
```python
# Click at coordinates
computer_control(action="mouse_click", x=100, y=200, button="left")

# Move mouse
computer_control(action="mouse_move", x=500, y=300)

# Drag operation
computer_control(action="mouse_drag", x=100, y=100, parameters={"end_x": 200, "end_y": 200})

# Scroll
computer_control(action="mouse_scroll", scroll_direction="down", scroll_amount=3)

# Get current position
computer_control(action="get_mouse_position")
```

### ⌨️ Keyboard Control
```python
# Type text
computer_control(action="type_text", text="Hello World!")

# Send special keys
computer_control(action="send_keys", keys="enter")

# Key combinations
computer_control(action="key_combination", keys="ctrl+c")
```

### 📱 Application Management
```python
# Launch application
computer_control(action="launch_app", target="notepad")

# Close application
computer_control(action="close_app", target="notepad")

# List processes
computer_control(action="list_processes")

# Kill process
computer_control(action="kill_process", target="notepad.exe")
```

### 🪟 Window Management
```python
# List windows
computer_control(action="list_windows")

# Focus window
computer_control(action="focus_window", target="Calculator")

# Move window
computer_control(action="move_window", target="Notepad", x=100, y=100)

# Resize window
computer_control(action="resize_window", target="Notepad", width=800, height=600)

# Minimize/Maximize
computer_control(action="minimize_window", target="Notepad")
computer_control(action="maximize_window", target="Notepad")
```

### 📋 Clipboard Operations
```python
# Get clipboard content
computer_control(action="get_clipboard")

# Set clipboard content
computer_control(action="set_clipboard", text="Hello Clipboard!")
```

### 🔍 System Information
```python
# Get system info
computer_control(action="get_system_info")

# Get screen info
computer_control(action="get_screen_info")

# Execute safe commands
computer_control(action="execute_command", target="dir")
```

## 🛡️ Safety Features

### Built-in Safety Mechanisms
- **PyAutoGUI Failsafe**: Move mouse to top-left corner to stop execution
- **Command Filtering**: Only safe system commands are allowed
- **Process Protection**: Critical system processes are protected
- **Confirmation Prompts**: Destructive actions require confirmation
- **Timeout Protection**: All operations have configurable timeouts

### Security Considerations
- **Restricted Commands**: Only whitelisted system commands can be executed
- **Safe Mode**: Additional safety checks for potentially dangerous operations
- **Input Validation**: All user inputs are validated and sanitized
- **Error Handling**: Comprehensive error handling prevents system crashes

## 🧪 Testing

### Run All Tests
```powershell
# Complete test suite
python test_computer_control_actions_async.py

# Action validation
python test_action_names_validation.py

# Integration test
python test_manus_computer_control_integration.py
```

### Expected Test Results
```
🧪 Testing Computer Control Tool Actions
==================================================
✅ Take Screenshot: PASS
✅ Get Mouse Position: PASS
✅ Get System Info: PASS
✅ Get Screen Info: PASS
✅ List Processes: PASS
✅ Get Clipboard: PASS
✅ Set Clipboard: PASS
✅ List Windows: PASS

📈 SUCCESS RATE: 100.0%
```

## 🎯 Example Workflows

### 1. Automated File Organization
```python
# Natural language command
"Organize my Desktop by moving all images to a new folder called 'Images'"

# The AI will:
# 1. Take a screenshot to see the desktop
# 2. Identify image files
# 3. Create an 'Images' folder
# 4. Move all image files to the folder
```

### 2. Application Automation
```python
# Natural language command
"Open calculator, compute 15% of 250, then copy the result to clipboard"

# The AI will:
# 1. Launch calculator application
# 2. Perform the calculation
# 3. Copy result to clipboard
# 4. Confirm completion
```

### 3. System Monitoring
```python
# Natural language command
"Show me system performance and take a screenshot of task manager"

# The AI will:
# 1. Gather system information
# 2. Open task manager
# 3. Take a screenshot
# 4. Provide performance analysis
```

## 🔧 Troubleshooting

### Common Issues and Solutions

#### Issue: "Computer control tool not working"
**Solution:**
```powershell
# Verify dependencies
pip install pyautogui pygetwindow pyperclip opencv-python

# Check permissions (run as administrator if needed)
# Disable antivirus interference temporarily
```

#### Issue: "Ollama model not found"
**Solution:**
```powershell
# Check Ollama status
ollama list

# Pull the model if missing
ollama pull llama3.2-vision:11b

# Restart Ollama service
ollama serve
```

#### Issue: "PyAutoGUI failsafe triggered"
**Solution:**
```python
# Disable failsafe if needed (use with caution)
import pyautogui
pyautogui.FAILSAFE = False

# Or move mouse away from top-left corner
```

#### Issue: "Permission denied errors"
**Solution:**
```powershell
# Run PowerShell as Administrator
# Check Windows UAC settings
# Verify antivirus exclusions
```

### Debug Mode
Enable detailed logging for troubleshooting:
```powershell
# Set debug environment variable
$env:MANUS_DEBUG = "true"

# Run with verbose logging
python main.py --prompt "test command" --verbose
```

## 📚 API Reference

### Core Classes

#### `ComputerControlTool`
Primary class for computer automation with 29 available actions.

#### `Manus`
Main AI agent class with enhanced computer control capabilities.

#### `AutomationTool`
Advanced workflow automation and task scheduling.

### Available Actions Reference
See [`COMPUTER_CONTROL_ACTIONS_GUIDE.md`](COMPUTER_CONTROL_ACTIONS_GUIDE.md) for complete action reference.

## 🤝 Contributing

### Development Setup
```powershell
# Clone repository
git clone https://github.com/yourusername/ParManusAI.git
cd ParManusAI

# Install development dependencies
pip install -r requirements.txt
pip install pytest black flake8 mypy

# Run tests
pytest tests/

# Format code
black app/
```

### Adding New Features
1. Create feature branch: `git checkout -b feature/new-feature`
2. Implement changes with tests
3. Run test suite: `python test_computer_control_actions_async.py`
4. Submit pull request

### Reporting Issues
- Use GitHub Issues for bug reports
- Include system information and error logs
- Provide reproduction steps
- Test with latest version first

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- **Ollama Team** - For providing excellent local LLM hosting
- **PyAutoGUI** - For cross-platform GUI automation
- **Meta AI** - For the Llama 3.2 model
- **OpenCV** - For computer vision capabilities
- **Python Community** - For the amazing ecosystem

## 📞 Support

### Getting Help
- **Documentation**: Check this README and action guide
- **Issues**: GitHub Issues for bugs and feature requests
- **Discussions**: GitHub Discussions for questions
- **Email**: [your-email@example.com]

### Community
- **Discord**: [Your Discord Server]
- **Reddit**: [Your Subreddit]
- **Twitter**: [@YourTwitter]

---

**⚡ Ready to revolutionize your computer automation with AI? Get started now!**

```powershell
git clone https://github.com/yourusername/ParManusAI.git
cd ParManusAI
pip install -r requirements.txt
python main.py
```

> **🎯 ParManus AI - Your Autonomous Computer Control Agent**
> *Bringing the future of AI-powered automation to your fingertips*
