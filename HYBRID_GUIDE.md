# ParManus AI - Hybrid System Guide

## 🎯 **Hybrid Architecture Overview**

The hybrid system uses **two specialized models** for optimal performance:

### **🛠️ llama3.2 (Tools Model)**
- **Purpose**: Tool calling, function execution, agent operations
- **Capabilities**: Browser automation, file operations, code execution, web search
- **When Used**: Automatically when tools are needed

### **👁️ llama3.2-vision (Vision Model)**  
- **Purpose**: Image analysis, visual understanding
- **Capabilities**: Screenshot analysis, image description, visual reasoning
- **When Used**: Automatically when vision content is detected

## 🚀 **Setup Instructions**

### **Quick Setup**
```bash
./setup_hybrid.sh
```

### **Manual Setup**
```bash
# 1. Pull both models
ollama pull llama3.2
ollama pull llama3.2-vision

# 2. Use hybrid config
cp config/config_hybrid.toml config/config.toml

# 3. Test the system
python main.py --prompt "review google.com"
```

## 🤖 **How Model Routing Works**

### **Automatic Model Selection**
The system automatically chooses the right model based on:

1. **Tool Requests** → llama3.2
   - Any agent operation (browser, code, file, search)
   - Function calling requirements
   - Multi-step workflows

2. **Vision Content** → llama3.2-vision
   - Image analysis requests
   - Screenshot descriptions
   - Visual reasoning tasks

3. **General Chat** → llama3.2
   - Default for text-only conversations
   - Better tool integration

## 📋 **Usage Examples**

### **Tool-Based Tasks (llama3.2)**
```bash
# Browser automation
python main.py --prompt "review google.com"

# Code generation and execution
python main.py --agent code --prompt "write a web scraper"

# File operations
python main.py --agent file --prompt "analyze data.csv"

# Web search
python main.py --prompt "search for latest AI news"
```

### **Vision Tasks (llama3.2-vision)**
```bash
# Image analysis
python main.py --prompt "analyze this screenshot"

# Visual description
python main.py --prompt "describe what you see in the image"

# Visual reasoning
python main.py --prompt "what's wrong with this UI design?"
```

### **Mixed Tasks (Automatic Routing)**
```bash
# Will use tools model for browsing, vision model if screenshots needed
python main.py --prompt "browse to a website and describe what you see"

# Interactive mode - routes automatically
python main.py
```

## 🔧 **Configuration Details**

### **Hybrid Config Structure**
```toml
[llm]
api_type = "ollama"
model = "llama3.2"  # Primary tools model

[llm.vision]
enabled = true
model = "llama3.2-vision"  # Vision model
```

### **Model Routing Logic**
```python
# Tools needed? → llama3.2
if tools_requested:
    use_model = "llama3.2"

# Vision content? → llama3.2-vision  
elif has_images or vision_keywords:
    use_model = "llama3.2-vision"

# Default → llama3.2
else:
    use_model = "llama3.2"
```

## ✅ **Benefits of Hybrid Approach**

### **🛠️ Full Tool Support**
- ✅ Browser automation works perfectly
- ✅ File operations fully functional
- ✅ Code execution and debugging
- ✅ Web search and data retrieval
- ✅ All agent types available

### **👁️ Vision Capabilities**
- ✅ Image analysis and description
- ✅ Screenshot understanding
- ✅ Visual reasoning and feedback
- ✅ UI/UX analysis

### **🚀 Performance Optimized**
- ✅ Right model for the right task
- ✅ No tool limitations on vision model
- ✅ No vision limitations on tools model
- ✅ Seamless automatic switching

## 🔍 **Troubleshooting**

### **Model Not Found Errors**
```bash
# Ensure both models are pulled
ollama list
ollama pull llama3.2
ollama pull llama3.2-vision
```

### **Tool Calling Issues**
```bash
# Verify tools model is working
ollama run llama3.2 "Hello"
```

### **Vision Issues**
```bash
# Verify vision model is working
ollama run llama3.2-vision "Describe an image"
```

### **Check System Status**
```bash
# Test hybrid system
python main.py --prompt "What models are you using?" --no-wait
```

## 🎯 **Best Practices**

### **For Tool Tasks**
- Use specific agent types: `--agent browser`, `--agent code`
- Request specific actions: "browse", "execute", "search"
- The system will automatically use llama3.2

### **For Vision Tasks**
- Include vision keywords: "see", "image", "screenshot", "visual"
- Upload images or reference visual content
- The system will automatically use llama3.2-vision

### **For Mixed Workflows**
- Let the system route automatically
- Use interactive mode for complex tasks
- The system handles model switching seamlessly

## 📊 **Performance Comparison**

| Task Type | Single Model | Hybrid System |
|-----------|-------------|---------------|
| Tool Calling | ❌ Vision model fails | ✅ Perfect with llama3.2 |
| Image Analysis | ✅ Works | ✅ Better with llama3.2-vision |
| Browser Automation | ❌ No tools support | ✅ Full support |
| Code Execution | ❌ No tools support | ✅ Full support |
| Mixed Tasks | ❌ Limited | ✅ Seamless routing |

## 🚀 **Ready to Use**

Your hybrid system gives you:
- ✅ **Best tool performance** (llama3.2)
- ✅ **Best vision performance** (llama3.2-vision)  
- ✅ **Automatic routing** (no manual switching)
- ✅ **Full ParManus capabilities** (all agents and tools)

Just run `python main.py --prompt "review google.com"` and enjoy the best of both worlds!

