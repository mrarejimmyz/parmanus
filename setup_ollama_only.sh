#!/bin/bash

# ParManus AI - Ollama Setup Script

echo "🚀 Setting up ParManus AI with Ollama (Simplified)"
echo "================================================="

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

# Pull required model
echo "📥 Pulling Llama 3.2 Vision model..."
ollama pull llama3.2-vision

# Install Python dependencies (simplified)
echo "📦 Installing Python dependencies..."
pip install -r requirements.txt

# Test the system
echo "🧪 Testing the system..."
python main.py --prompt "Hello, test the Ollama-only system" --no-wait

echo ""
echo "✅ Setup complete!"
echo ""
echo "🎯 ParManus AI is now running with:"
echo "   • Ollama backend only (no llama-cpp-python)"
echo "   • Llama 3.2 Vision model (unified text + vision)"
echo "   • Full tool calling support"
echo "   • All agent types available"
echo ""
echo "🚀 Usage examples:"
echo "   python main.py --prompt 'review google.com'"
echo "   python main.py --agent code --prompt 'write a Python script'"
echo "   python main.py --agent browser --prompt 'search for AI news'"
echo "   python main.py  # Interactive mode"
echo ""
echo "🛑 To stop Ollama server:"
echo "   kill $OLLAMA_PID"
echo ""
echo "📋 Available models:"
ollama list

