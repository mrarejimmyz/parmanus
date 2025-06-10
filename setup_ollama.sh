#!/bin/bash

# ParManus AI Setup Script for Ollama

echo "🚀 Setting up ParManus AI with Ollama..."

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

# Pull required models
echo "📥 Pulling Llama 3.2 models..."
ollama pull llama3.2
ollama pull llama3.2-vision

# Copy Ollama config
echo "⚙️ Setting up configuration..."
cp config/config_ollama.toml config/config.toml

# Install Python dependencies
echo "📦 Installing Python dependencies..."
pip install -r requirements.txt

# Test the system
echo "🧪 Testing the system..."
python main.py --prompt "Hello, this is a test" --no-wait

echo "✅ Setup complete!"
echo ""
echo "🎯 To use ParManus AI:"
echo "   python main.py --prompt 'Your prompt here'"
echo "   python main.py  # For interactive mode"
echo ""
echo "🛑 To stop Ollama server:"
echo "   kill $OLLAMA_PID"

