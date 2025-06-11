# 🤖 ParManus AI - Autonomous Computer Control Agent

[![GitHub stars](https://img.shields.io/github/stars/mrarejimmyz/parmanus?style=social)](https://github.com/mrarejimmyz/parmanus/stargazers)
[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/)
[![Windows](https://img.shields.io/badge/platform-Windows-lightgrey.svg)](https://www.microsoft.com/windows)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

> **ParManus AI: Advanced Autonomous Computer Control Agent**
>
> Enhanced fork with comprehensive computer automation capabilities powered by Llama 3.2 11B Vision. Take complete control of your computer using natural language commands.

🚀 **ParManus can now autonomously control your entire computer system!** 🖥️

## ⚡ Quick Start

```powershell
# Clone and setup in 3 commands
git clone https://github.com/yourusername/ParManusAI.git
cd ParManusAI
python setup_parmanus.py
```

**Then start using it:**
```powershell
python main.py --prompt "Take a screenshot and organize my desktop"
```

## 🌟 **NEW: Full Computer Control Capabilities**

### 🖥️ **What ParManus AI Can Do**
- **📸 Screen Analysis** - Take screenshots and understand visual content
- **🖱️ Mouse Control** - Click, drag, scroll anywhere on screen with pixel precision
- **⌨️ Keyboard Automation** - Type text, send keys, execute shortcuts
- **📱 App Management** - Launch, close, and control any application
- **🪟 Window Control** - Move, resize, minimize, maximize windows
- **👁️ Computer Vision** - Find and interact with UI elements automatically
- **📋 Clipboard Operations** - Read and write clipboard content
- **🔧 System Monitoring** - Get system info, process lists, performance data
- **⚡ Complex Workflows** - Execute multi-step automation sequences

### 🎯 **Example Commands**
```
"Take a screenshot and tell me what applications are running"
"Open calculator and compute 15% of 250"
"Find all PDF files in Downloads and organize them by date"
"Take control of my browser and search for AI news"
"Show me system performance and running processes"
"Automate creating a backup of my Documents folder"
```

## ✨ Enhanced Features

ParManus includes several key optimizations and revolutionary computer control capabilities:

### 🚀 **Core Optimizations**
- **Persistent Model Caching**: Eliminates 9-10 second loading time on subsequent runs
- **Enhanced Tool-Calling**: Fixed critical issues with tool execution
- **Improved Timeout Handling**: Returns partial results when possible
- **Resource Management**: Proper cleanup procedures and memory optimization
- **Interrupt Handling**: Graceful shutdown with proper resource cleanup

### 🤖 **Computer Control Features**
- **29 Automation Actions**: Complete computer control toolkit
- **Visual AI Integration**: Llama 3.2 Vision for screenshot analysis
- **Safety Mechanisms**: Built-in safeguards and error recovery
- **Natural Language Interface**: Control your computer with plain English
- **Cross-Application Support**: Works with any Windows application

## 📚 Complete Documentation

For detailed installation, configuration, and usage instructions, see:
- **[📖 Installation & Usage Guide](INSTALLATION_AND_USAGE_GUIDE.md)** - Complete setup and usage documentation
- **[🎯 Computer Control Actions Guide](COMPUTER_CONTROL_ACTIONS_GUIDE.md)** - All 29 available automation actions
- **[✅ Computer Control Completion Report](COMPUTER_CONTROL_COMPLETION_REPORT.md)** - Technical implementation details

<video src="https://private-user-images.githubusercontent.com/61239030/420168772-6dcfd0d2-9142-45d9-b74e-d10aa75073c6.mp4" controls="controls" muted="muted" class="d-block rounded-bottom-2 border-top width-fit" style="max-height:640px; min-height: 200px"></video>

## 🚀 Installation

### Method 1: Quick Setup (Recommended)
```powershell
# Clone repository
git clone https://github.com/yourusername/ParManusAI.git
cd ParManusAI

# Run automated setup script
python setup_parmanus.py
```

### Method 2: Manual Installation

#### Prerequisites
- Windows 10/11
- Python 3.11+
- [Ollama](https://ollama.ai) installed

#### Step-by-step Setup
```powershell
# 1. Create virtual environment
python -m venv venv
.\venv\Scripts\Activate.ps1

# 2. Install dependencies
pip install -r requirements.txt

# 3. Install computer control dependencies
pip install pyautogui pygetwindow pyperclip opencv-python screeninfo mss

# 4. Download AI model
ollama pull llama3.2-vision:11b

# 5. Run tests
python test_computer_control_actions_async.py
```

4. Install dependencies:
```bash
uv pip install -r requirements.txt
```

## ⚙️ Configuration

ParManus AI automatically creates a `config.toml` file during setup. You can customize settings:

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
```

## 🎯 Usage

### Interactive Mode
```powershell
python main.py
```

### Command Line Mode
```powershell
# Single command execution
python main.py --prompt "Take a screenshot and tell me what's on screen"

# Computer control tasks
python main.py --prompt "Open calculator and compute 25 * 47"

# File operations
python main.py --prompt "Organize my Downloads folder by file type"
```

### Specific Agent Modes
```powershell
# Computer control agent
python main.py --agent manus --prompt "Control my desktop applications"

# Browser automation
python main.py --agent browser --prompt "Navigate to google.com"

# File operations
python main.py --agent file --prompt "Analyze my documents"
```

### Advanced Features
```powershell
# MCP tool version
python run_mcp.py

# Multi-agent workflows
python run_flow.py
```

## 🧪 Testing

Verify your installation:
```powershell
# Test computer control actions
python test_computer_control_actions_async.py

# Validate action names
python test_action_names_validation.py

# Integration test
python test_manus_computer_control_integration.py
```

## 🛡️ Safety & Security

- **Built-in Safety**: PyAutoGUI failsafe and command filtering
- **Safe Mode**: Additional confirmations for destructive actions
- **Whitelisted Commands**: Only safe system commands allowed
- **Error Recovery**: Intelligent error handling and recovery

## 📖 Documentation

- **[Complete Installation Guide](INSTALLATION_AND_USAGE_GUIDE.md)** - Detailed setup instructions
- **[Computer Control Actions](COMPUTER_CONTROL_ACTIONS_GUIDE.md)** - All 29 automation actions
- **[Technical Report](COMPUTER_CONTROL_COMPLETION_REPORT.md)** - Implementation details

## 🤝 Contributing

We welcome contributions! To contribute:

1. **Fork the repository**
2. **Create feature branch**: `git checkout -b feature/amazing-feature`
3. **Run tests**: `python test_computer_control_actions_async.py`
4. **Submit pull request**

### Development Setup
```powershell
# Install development dependencies
pip install pytest black flake8 mypy

# Run tests
pytest tests/

# Format code
black app/
```

## Community Group

Join our networking group on Feishu and share your experience with other developers!

<div align="center" style="display: flex; gap: 20px;">
    <img src="assets/community_group.jpg" alt="ParManus Community Group" width="300" />
</div>

## Star History

[![Star History Chart](https://api.star-history.com/svg?repos=mrarejimmyz/parmanus&type=Date)](https://star-history.com/#mrarejimmyz/parmanus&Date)

## Sponsors

Thanks to [PPIO](https://ppinfra.com/user/register?invited_by=OCPKCN&utm_source=github_parmanus&utm_medium=github_readme&utm_campaign=link) for computing source support.

> PPIO: The most affordable and easily-integrated MaaS and GPU cloud solution.

## Acknowledgement

ParManus is maintained by [@mrarejimmyz](https://github.com/mrarejimmyz) with significant performance optimizations and tool-calling enhancements.

Thanks to all previous contributors from the original OpenManus project, including contributors from MetaGPT.

Additional thanks to [anthropic-computer-use](https://github.com/anthropics/anthropic-quickstarts/tree/main/computer-use-demo)
and [browser-use](https://github.com/browser-use/browser-use) for providing basic support for this project!

We are also grateful to [AAAJ](https://github.com/metauto-ai/agent-as-a-judge), [MetaGPT](https://github.com/geekan/MetaGPT), [OpenHands](https://github.com/All-Hands-AI/OpenHands) and [SWE-agent](https://github.com/SWE-agent/SWE-agent).

We also thank stepfun(阶跃星辰) for supporting our Hugging Face demo space.

## Cite

```bibtex
@misc{parmanus2025,
  author = {MrareJimmy, Jimmy Zhang and Xinbin Liang and Jinyu Xiang and Zhaoyang Yu and Jiayi Zhang and Sirui Hong and Sheng Fan and Xiao Tang},
  title = {ParManus: An optimized framework for building general AI agents},
  year = {2025},
  publisher = {GitHub},
  url = {https://github.com/mrarejimmyz/parmanus},
}
```
