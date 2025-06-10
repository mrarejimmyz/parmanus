# ParManus AI Agent - Complete System with Local GGUF Support

A comprehensive AI agent system with full tool calling capabilities, optimized for local GGUF models while supporting Ollama as a fallback.

## 🚀 Features

### **Core Capabilities**
- **Local GGUF Models**: Native support for your local Llama 3.2 models
- **Full Tool System**: Complete integration with all ParManus tools
- **Vision Support**: Multi-modal capabilities with llava models
- **Agent Routing**: Automatic agent selection (manus, code, browser, file, planner)
- **Memory System**: Session persistence and recovery
- **Hybrid Architecture**: Works with both local models and Ollama

### **Available Tools**
- **Browser Automation**: Web scraping, form filling, navigation
- **File Operations**: Read, write, edit files and documents
- **Code Execution**: Python script execution and debugging
- **Web Search**: Search engines and information retrieval
- **Planning Tools**: Task breakdown and organization
- **Terminal/Bash**: System command execution

### **Agent Types**
- **Manus**: General-purpose AI assistant with all tools
- **Code**: Programming and development specialist
- **Browser**: Web automation and scraping expert
- **File**: Document and data processing specialist
- **Planner**: Task planning and organization assistant

## 📋 Prerequisites

### **For Local Models (Recommended)**
1. **GGUF Models**: Place your models in the `models/` directory
   ```
   models/
   ├── Llama-3.2-11B-Vision-Instruct.Q4_K_M.gguf
   └── llava-1.6-mistral-7b-gguf/
       ├── ggml-model-q4_k.gguf
       └── mmproj-model-f16.gguf
   ```

2. **GPU Support**: CUDA-compatible GPU (RTX 3070 or better recommended)

### **For Ollama (Optional Fallback)**
1. Install Ollama:
   ```bash
   curl -fsSL https://ollama.ai/install.sh | sh
   ```

2. Pull models:
   ```bash
   ollama pull llama3.2
   ollama pull llama3.2-vision
   ```

## 🛠️ Installation

1. **Clone Repository**:
   ```bash
   git clone https://github.com/mrarejimmyz/ParManusAI.git
   cd ParManusAI
   ```

2. **Install Dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

3. **Setup Browser (for browser tools)**:
   ```bash
   playwright install chromium
   ```

4. **Configure (Optional)**:
   ```bash
   cp config/config.toml config/config.toml.local
   # Edit config/config.toml.local as needed
   ```

## 🎯 Usage

### **Quick Start**
```bash
# Basic usage with local models
python main.py --prompt "Hello, analyze this code for me"

# Interactive mode
python main.py

# Specific agent
python main.py --agent code --prompt "Write a Python web scraper"

# Browser automation
python main.py --agent browser --prompt "Search for Python tutorials"

# File operations
python main.py --agent file --prompt "Read and summarize document.txt"
```

### **Advanced Usage**
```bash
# Use Ollama instead of local models
python main.py --api-type ollama --prompt "Help me with this task"

# Custom workspace
python main.py --workspace ./my_project --prompt "Analyze this project"

# Limit agent steps
python main.py --max-steps 10 --prompt "Complex task"

# Simple mode (no tools)
python main.py --simple --prompt "Just chat"

# Custom config
python main.py --config config/my_config.toml --prompt "Task"
```

### **Interactive Commands**
Once in interactive mode, you can:
- Type any prompt for the AI to process
- Use `quit`, `exit`, `bye`, or `q` to exit
- The system automatically routes to appropriate agents

## ⚙️ Configuration

### **Default Configuration** (`config/config.toml`)
```toml
[llm]
api_type = "local"  # or "ollama"
model = "Llama-3.2-11B-Vision-Instruct"
model_path = "models/Llama-3.2-11B-Vision-Instruct.Q4_K_M.gguf"
max_tokens = 2048
temperature = 0.0
n_gpu_layers = -1  # Use all GPU layers
gpu_memory_limit = 7000  # MB for RTX 3070

[llm.vision]
enabled = true
model = "llava-v1.6-mistral-7b"
model_path = "models/llava-1.6-mistral-7b-gguf/ggml-model-q4_k.gguf"
clip_model_path = "models/llava-1.6-mistral-7b-gguf/mmproj-model-f16.gguf"

[browser]
headless = false
disable_security = true

[memory]
save_session = false
recover_last_session = false
```

### **Environment Variables**
```bash
export PARMANUS_WORKSPACE="./workspace"
export PARMANUS_MODEL_PATH="./models/your-model.gguf"
export PARMANUS_API_TYPE="local"
```

## 🔧 Tool System

### **Available Tools**
1. **BrowserUseTool**: Web automation and scraping
2. **StrReplaceEditor**: File editing and manipulation
3. **PythonExecute**: Code execution and debugging
4. **WebSearch**: Internet search capabilities
5. **Bash**: Terminal command execution
6. **PlanningTool**: Task breakdown and planning
7. **AskHuman**: Interactive user input

### **Tool Usage Examples**
```bash
# Browser automation
python main.py --prompt "Go to github.com and search for Python projects"

# File operations
python main.py --prompt "Create a Python script that reads CSV files"

# Code execution
python main.py --prompt "Write and run a script to analyze data.csv"

# Web search
python main.py --prompt "Search for the latest Python 3.12 features"

# Planning
python main.py --prompt "Create a plan to build a web application"
```

## 🎭 Agent Routing

The system automatically selects the best agent based on your prompt:

- **Keywords for Code Agent**: code, program, script, debug, function, python, javascript
- **Keywords for Browser Agent**: browse, web, scrape, website, url, browser
- **Keywords for File Agent**: file, save, read, write, data, edit
- **Keywords for Planner Agent**: plan, schedule, task, organize, steps
- **Default**: Manus (general-purpose with all tools)

## 🔍 Troubleshooting

### **Local Model Issues**
```bash
# Check if model exists
ls -la models/

# Test model loading
python -c "from llama_cpp import Llama; print('llama-cpp-python works')"

# Check GPU
nvidia-smi
```

### **Ollama Issues**
```bash
# Check Ollama status
ollama list

# Start Ollama server
ollama serve

# Test connection
curl http://localhost:11434/api/tags
```

### **Tool Issues**
```bash
# Install browser dependencies
playwright install

# Check Python environment
python -c "import playwright; print('Playwright available')"
```

### **Common Solutions**
1. **Model not found**: Check `model_path` in config
2. **GPU memory error**: Reduce `n_gpu_layers` or `gpu_memory_limit`
3. **Tool errors**: Install missing dependencies
4. **Permission errors**: Check file/directory permissions

## 📊 Performance

### **Optimizations**
- **GPU Acceleration**: Full CUDA support for local models
- **Memory Management**: Efficient context window handling
- **Tool Caching**: Reuse tool instances across calls
- **Async Operations**: Non-blocking tool execution

### **Benchmarks** (RTX 3070, 8GB VRAM)
- **Text Generation**: ~20-30 tokens/second
- **Tool Execution**: ~1-3 seconds per tool call
- **Memory Usage**: ~6-7GB GPU, ~2-4GB RAM
- **Startup Time**: ~10-15 seconds (model loading)

## 🔒 Privacy & Security

- **100% Local**: All AI processing on your hardware
- **No Data Transmission**: No external API calls for AI inference
- **Secure Tools**: Sandboxed execution environment
- **Session Privacy**: Local session storage only

## 🚀 Advanced Features

### **Custom Agents**
Create custom agents by extending the base classes:
```python
from app.agent.base import BaseAgent

class MyCustomAgent(BaseAgent):
    # Your custom implementation
    pass
```

### **Custom Tools**
Add new tools by implementing the BaseTool interface:
```python
from app.tool.base import BaseTool

class MyCustomTool(BaseTool):
    # Your custom tool implementation
    pass
```

### **MCP Integration**
Connect to Model Context Protocol servers for extended capabilities.

## 📈 Roadmap

- [ ] Additional model format support (ONNX, TensorRT)
- [ ] More specialized agents (data analysis, creative writing)
- [ ] Enhanced vision capabilities
- [ ] Plugin system for custom tools
- [ ] Web interface
- [ ] API server mode

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests if applicable
5. Submit a pull request

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🆘 Support

- **Issues**: Open an issue on GitHub
- **Discussions**: Use GitHub Discussions
- **Documentation**: Check the wiki for detailed guides

---

**Note**: This system is optimized for local GGUF models but provides Ollama fallback for maximum compatibility. All tools work with both backends.

