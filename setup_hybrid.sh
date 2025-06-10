#!/bin/bash

# ParManus AI - Hybrid Setup (Tools + Vision)

echo "🚀 Setting up ParManus AI with Hybrid Models"
echo "============================================="
echo "🛠️ llama3.2 for tool calling"
echo "👁️ llama3.2-vision for vision tasks"
echo ""

# Check if we're in the right directory
if [ ! -f "main.py" ]; then
    echo "❌ Error: Please run this script from the ParManus AI directory"
    exit 1
fi

# Check if Ollama is installed
if ! command -v ollama &> /dev/null; then
    echo "📦 Installing Ollama..."
    curl -fsSL https://ollama.ai/install.sh | sh
else
    echo "✅ Ollama already installed"
fi

# Start Ollama server in background
echo "🔄 Starting Ollama server..."
ollama serve &
OLLAMA_PID=$!

# Wait for server to start
echo "⏳ Waiting for Ollama server to start..."
sleep 5

# Pull both models
echo "📥 Pulling llama3.2 (tools model)..."
ollama pull llama3.2

echo "📥 Pulling llama3.2-vision (vision model)..."
ollama pull llama3.2-vision

# Set up hybrid configuration
echo "⚙️ Setting up hybrid configuration..."
cp config/config_hybrid.toml config/config.toml

# Install Python dependencies
echo "📦 Installing Python dependencies..."
pip install -r requirements.txt

# Test the hybrid system
echo ""
echo "🧪 Testing hybrid system..."
echo "Testing tools capability..."
python main.py --prompt "What tools do you have available?" --no-wait

echo ""
echo "Testing vision capability..."
python main.py --prompt "Describe what you can see in images" --no-wait

echo ""
echo "✅ Hybrid setup complete!"
echo ""
echo "🎯 Your ParManus AI now has:"
echo "   🛠️ Full tool calling (llama3.2)"
echo "   👁️ Vision capabilities (llama3.2-vision)"
echo "   🤖 Smart routing between models"
echo "   🌐 Browser automation"
echo "   📁 File operations"
echo "   💻 Code execution"
echo "   🔍 Web search"
echo ""
echo "🚀 Usage examples:"
echo "   # Tool-based tasks (uses llama3.2)"
echo "   python main.py --prompt 'review google.com'"
echo "   python main.py --agent code --prompt 'write a Python script'"
echo ""
echo "   # Vision tasks (uses llama3.2-vision)"
echo "   python main.py --prompt 'analyze this screenshot'"
echo "   python main.py --prompt 'describe what you see'"
echo ""
echo "   # Interactive mode"
echo "   python main.py"
echo ""
echo "🛑 To stop Ollama server:"
echo "   kill $OLLAMA_PID"
echo ""
echo "📋 Available models:"
ollama list

