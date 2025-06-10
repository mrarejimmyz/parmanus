#!/bin/bash

# ParManus AI - Quick Setup for Llama 3.2 Vision

echo "🚀 Setting up ParManus AI for Llama 3.2 Vision (Unified Model)"
echo "=============================================================="

# Check if we're in the right directory
if [ ! -f "main.py" ]; then
    echo "❌ Error: Please run this script from the ParManus AI directory"
    exit 1
fi

# Check for model file
MODEL_FILE="models/Llama-3.2-11B-Vision-Instruct.Q4_K_M.gguf"
echo "🔍 Checking for Llama 3.2 Vision model..."

if [ -f "$MODEL_FILE" ]; then
    echo "✅ Model found: $MODEL_FILE"
    echo "📊 Model size: $(du -h "$MODEL_FILE" | cut -f1)"
else
    echo "❌ Model NOT found: $MODEL_FILE"
    echo ""
    echo "📋 Please place your Llama 3.2 Vision model at:"
    echo "   $(pwd)/$MODEL_FILE"
    echo ""
    echo "💡 This single model handles both text and vision tasks!"
    exit 1
fi

# Install dependencies
echo ""
echo "📦 Installing dependencies..."
pip install -r requirements.txt

# Test the system
echo ""
echo "🧪 Testing the system..."
python main.py --prompt "Hello, test the unified Llama 3.2 Vision model" --no-wait

echo ""
echo "✅ Setup complete!"
echo ""
echo "🎯 Your Llama 3.2 Vision model can now handle:"
echo "   • Text conversations"
echo "   • Image analysis"
echo "   • Code generation"
echo "   • Web browsing tasks"
echo "   • File operations"
echo "   • All tool calling functionality"
echo ""
echo "🚀 Usage examples:"
echo "   python main.py --prompt 'review google.com'"
echo "   python main.py --agent code --prompt 'write a Python script'"
echo "   python main.py --agent browser --prompt 'search for AI news'"
echo "   python main.py  # Interactive mode"

