# ParManus AI - Troubleshooting Guide

## 🚨 Common Issues and Solutions

### Issue 1: "Failed to load model from file"
**Problem**: Local GGUF model file not found
**Solution**: 
```bash
# Option A: Use Ollama instead
python main.py --api-type ollama --prompt "test"

# Option B: Download GGUF models and place in models/ directory
# Option C: Update config to point to your actual model files
```

### Issue 2: "Connection error" with Ollama
**Problem**: Ollama server not running
**Solution**:
```bash
# Start Ollama server
ollama serve

# In another terminal, test
ollama list
ollama pull llama3.2
```

### Issue 3: "No module named 'llama_cpp'"
**Problem**: llama-cpp-python not installed
**Solution**:
```bash
pip install llama-cpp-python
# Or for GPU support:
pip install llama-cpp-python[cuda]
```

### Issue 4: "No module named 'openai'"
**Problem**: OpenAI package not installed (needed for Ollama)
**Solution**:
```bash
pip install openai httpx
```

## 🛠️ Quick Setup Commands

### For Ollama (Recommended)
```bash
# 1. Run the setup script
./setup_ollama.sh

# 2. Or manual setup:
ollama serve &
ollama pull llama3.2
python main.py --api-type ollama --prompt "test"
```

### For Local GGUF Models
```bash
# 1. Place your GGUF model in models/ directory
# 2. Update config.toml with correct path
# 3. Install llama-cpp-python
pip install llama-cpp-python
# 4. Test
python main.py --api-type local --prompt "test"
```

## 🔧 Configuration Files

### Use Ollama Configuration
```bash
cp config/config_ollama.toml config/config.toml
```

### Check Current Configuration
```bash
python -c "
import tomllib
with open('config/config.toml', 'rb') as f:
    config = tomllib.load(f)
print('API Type:', config['llm']['api_type'])
print('Model:', config['llm']['model'])
"
```

## 🧪 Testing Commands

### Test System Components
```bash
python test_system.py
```

### Test Specific Modes
```bash
# Test Ollama
python main.py --api-type ollama --prompt "Hello" --no-wait

# Test Local (if you have models)
python main.py --api-type local --prompt "Hello" --no-wait

# Test Simple Mode
python main.py --simple --prompt "Hello" --no-wait
```

## 📋 Dependency Check

### Check Required Packages
```bash
python -c "
packages = ['pydantic', 'tomllib', 'llama_cpp', 'openai', 'loguru']
for pkg in packages:
    try:
        __import__(pkg)
        print(f'✅ {pkg}')
    except ImportError:
        print(f'❌ {pkg} - install with: pip install {pkg}')
"
```

### Install All Dependencies
```bash
pip install -r requirements.txt
```

## 🎯 Working Examples

### Once Ollama is Running
```bash
# Basic usage
python main.py --prompt "review google.com"

# Specific agent
python main.py --agent browser --prompt "search for Python tutorials"

# Interactive mode
python main.py
```

### With Local Models (when available)
```bash
# Use local models
python main.py --api-type local --prompt "analyze this code"
```

## 🔍 Debug Information

### Check Ollama Status
```bash
curl http://localhost:11434/api/tags
ollama list
ps aux | grep ollama
```

### Check Model Files
```bash
find models/ -name "*.gguf" -ls
ls -la models/
```

### Check Configuration
```bash
cat config/config.toml
python test_system.py
```

