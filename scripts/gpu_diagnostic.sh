#!/bin/bash
# gpu_diagnostic.sh - Script to diagnose GPU acceleration for llama-cpp-python
# Created based on ParManus CUDA GPU Fix Implementation Guide

echo "🔍 GPU Diagnostic Tool for ParManus"
echo "==================================="

# Check CUDA installation
echo "Checking CUDA installation..."
if [ -d "${CUDA_HOME}" ]; then
  echo "✅ CUDA_HOME is set to: ${CUDA_HOME}"
else
  echo "❌ CUDA_HOME is not properly set"
fi

# Check NVIDIA driver
echo -e "\nChecking NVIDIA driver..."
if command -v nvidia-smi &> /dev/null; then
  echo "✅ NVIDIA driver is installed:"
  nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv
else
  echo "❌ NVIDIA driver not found or not accessible"
fi

# Check llama-cpp-python installation
echo -e "\nChecking llama-cpp-python installation..."
python -c "from llama_cpp import llama_cpp; has_cublas = hasattr(llama_cpp, '_LLAMA_CUBLAS'); \
print(f'CUDA support enabled: {has_cublas and llama_cpp._LLAMA_CUBLAS}')" || \
echo "❌ Failed to check llama-cpp-python CUDA support"

# Test model loading with GPU
echo -e "\nTesting model loading with GPU..."
if [ -f "/models/llama-jb.gguf" ]; then
  python -c "from llama_cpp import Llama; print('Loading model with GPU acceleration...'); \
  llm = Llama(model_path='/models/llama-jb.gguf', n_gpu_layers=-1, verbose=True); \
  print('✅ Model loaded successfully with GPU acceleration'); \
  output = llm('Hello, world!', max_tokens=5); \
  print(f'Model output: {output}')" || echo "❌ Failed to load model with GPU acceleration"
else
  echo "❌ Model file not found at /models/llama-jb.gguf"
fi

echo -e "\n🎯 GPU Diagnostic Complete"
