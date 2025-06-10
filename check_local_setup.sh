#!/bin/bash

# ParManus AI - Local Model Setup and Diagnostic Script

echo "🔍 ParManus AI Local Model Diagnostic"
echo "======================================"

# Check if we're in the right directory
if [ ! -f "main.py" ]; then
    echo "❌ Error: Please run this script from the ParManus AI directory"
    exit 1
fi

echo "📁 Current directory: $(pwd)"

# Check for model file
MODEL_FILE="models/Llama-3.2-11B-Vision-Instruct.Q4_K_M.gguf"
echo ""
echo "🔍 Checking for model file..."
if [ -f "$MODEL_FILE" ]; then
    echo "✅ Model found: $MODEL_FILE"
    echo "📊 Model size: $(du -h "$MODEL_FILE" | cut -f1)"
else
    echo "❌ Model NOT found: $MODEL_FILE"
    echo ""
    echo "📋 Please ensure your model file is located at:"
    echo "   $(pwd)/$MODEL_FILE"
    echo ""
    echo "💡 If your model is elsewhere, either:"
    echo "   1. Copy it: cp /path/to/your/model.gguf $MODEL_FILE"
    echo "   2. Create symlink: ln -s /path/to/your/model.gguf $MODEL_FILE"
    echo "   3. Update config.toml with the correct path"
fi

# Check models directory structure
echo ""
echo "📂 Models directory structure:"
if [ -d "models" ]; then
    find models/ -type f -name "*.gguf" -exec echo "✅ Found: {}" \; 2>/dev/null
    if [ $(find models/ -name "*.gguf" 2>/dev/null | wc -l) -eq 0 ]; then
        echo "⚠️  No .gguf files found in models/ directory"
    fi
else
    echo "❌ models/ directory does not exist"
    echo "📁 Creating models directory..."
    mkdir -p models
fi

# Check Python dependencies
echo ""
echo "🐍 Checking Python dependencies..."
python3 -c "
import sys
required = ['pydantic', 'llama_cpp', 'tomllib']
missing = []

for pkg in required:
    try:
        if pkg == 'tomllib':
            import tomllib
        elif pkg == 'llama_cpp':
            from llama_cpp import Llama
        else:
            __import__(pkg)
        print(f'✅ {pkg}')
    except ImportError:
        print(f'❌ {pkg}')
        missing.append(pkg)

if missing:
    print(f'\\n📦 Install missing packages:')
    if 'llama_cpp' in missing:
        print('   pip install llama-cpp-python')
    for pkg in missing:
        if pkg != 'llama_cpp':
            print(f'   pip install {pkg}')
"

# Check configuration
echo ""
echo "⚙️ Checking configuration..."
if [ -f "config/config.toml" ]; then
    echo "✅ Configuration file found"
    python3 -c "
import tomllib
try:
    with open('config/config.toml', 'rb') as f:
        config = tomllib.load(f)
    print(f'✅ API Type: {config[\"llm\"][\"api_type\"]}')
    print(f'✅ Model: {config[\"llm\"][\"model\"]}')
    print(f'✅ Model Path: {config[\"llm\"][\"model_path\"]}')
except Exception as e:
    print(f'❌ Config error: {e}')
"
else
    echo "❌ Configuration file not found"
fi

# Test basic functionality
echo ""
echo "🧪 Testing basic system..."
if python3 -c "import main" 2>/dev/null; then
    echo "✅ Main script imports successfully"
else
    echo "❌ Main script has import errors"
fi

echo ""
echo "🎯 Next Steps:"
echo "1. Ensure your model file is at: $MODEL_FILE"
echo "2. Install dependencies: pip install -r requirements.txt"
echo "3. Test the system: python main.py --prompt 'Hello' --no-wait"
echo ""
echo "📚 For more help, see TROUBLESHOOTING.md"

