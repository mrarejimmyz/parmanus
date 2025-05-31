#!/bin/bash
# quick_cuda_fix.sh - Script to fix CUDA support for llama-cpp-python
# Created based on ParManus CUDA GPU Fix Implementation Guide

set -e
echo "🔍 Checking current llama-cpp-python installation..."
pip show llama-cpp-python

echo "🧹 Uninstalling current llama-cpp-python..."
pip uninstall -y llama-cpp-python

echo "🚀 Installing llama-cpp-python with CUDA support..."
echo "Strategy 1: Using pre-built CUDA wheels..."
pip install --no-cache-dir llama-cpp-python --extra-index-url https://abetlen.github.io/llama-cpp-python/whl/cu122 && \
    echo "✅ Successfully installed llama-cpp-python with CUDA support using pre-built wheels!" && \
    STRATEGY_1_SUCCESS=true || STRATEGY_1_SUCCESS=false

if [ "$STRATEGY_1_SUCCESS" != "true" ]; then
    echo "Strategy 1 failed. Trying Strategy 2: Source compilation with CUDA flags..."

    # Ensure CUDA environment variables are set
    export CUDA_HOME=${CUDA_HOME:-/usr/local/cuda}
    export PATH=$CUDA_HOME/bin:$PATH
    export LD_LIBRARY_PATH=$CUDA_HOME/lib64:$LD_LIBRARY_PATH

    # Install with CUDA support via source compilation
    CMAKE_ARGS="-DLLAMA_CUBLAS=on" FORCE_CMAKE=1 pip install --no-cache-dir llama-cpp-python && \
        echo "✅ Successfully installed llama-cpp-python with CUDA support via source compilation!" && \
        STRATEGY_2_SUCCESS=true || STRATEGY_2_SUCCESS=false

    if [ "$STRATEGY_2_SUCCESS" != "true" ]; then
        echo "Strategy 2 failed. Trying Strategy 3: cuBLAS compilation (fallback)..."

        # Try with explicit cuBLAS flags
        LLAMA_CUBLAS=1 pip install --no-cache-dir llama-cpp-python && \
            echo "✅ Successfully installed llama-cpp-python with CUDA support via cuBLAS compilation!" && \
            STRATEGY_3_SUCCESS=true || STRATEGY_3_SUCCESS=false

        if [ "$STRATEGY_3_SUCCESS" != "true" ]; then
            echo "❌ All installation strategies failed. Please check your CUDA installation and try again."
            exit 1
        fi
    fi
fi

echo "🔍 Verifying GPU acceleration..."
# Create a simple test script
cat > /tmp/test_gpu.py << 'EOF'
import ctypes
import os
from llama_cpp import Llama

# Try to load CUDA libraries to verify they're accessible
try:
    ctypes.CDLL("libcudart.so")
    print("✅ CUDA Runtime library loaded successfully")
except:
    try:
        # Try with full path if default loading fails
        cuda_path = os.environ.get('CUDA_HOME', '/usr/local/cuda')
        ctypes.CDLL(f"{cuda_path}/lib64/libcudart.so")
        print("✅ CUDA Runtime library loaded successfully (with full path)")
    except Exception as e:
        print(f"❌ Failed to load CUDA Runtime library: {e}")

# Check if model loading shows GPU-related messages
print("Loading a small model to test GPU acceleration...")
try:
    # Use a small model for quick testing
    model_path = "/models/llama-jb.gguf"
    if not os.path.exists(model_path):
        print(f"Model not found at {model_path}, using a dummy model path for testing")
        model_path = None

    if model_path:
        # Set n_gpu_layers to force GPU usage
        llm = Llama(model_path=model_path, n_gpu_layers=1, verbose=True)
        print("✅ Model loaded successfully with GPU acceleration enabled")

        # Run a simple inference to verify
        output = llm("Hello, world!", max_tokens=5)
        print("✅ Inference completed successfully")
    else:
        print("⚠️ Skipping model loading test due to missing model file")
except Exception as e:
    print(f"❌ Error during model loading or inference: {e}")

# Print summary of GPU support
print("\n🔍 GPU Support Summary:")
try:
    from llama_cpp import llama_cpp
    has_cublas = hasattr(llama_cpp, "_LLAMA_CUBLAS")
    if has_cublas and llama_cpp._LLAMA_CUBLAS:
        print("✅ llama_cpp was compiled with CUDA support")
    else:
        print("❌ llama_cpp was NOT compiled with CUDA support")
except Exception as e:
    print(f"❌ Could not determine CUDA compilation status: {e}")
EOF

# Run the test script
python /tmp/test_gpu.py

echo -e "\n📊 GPU Usage Check:"
if command -v nvidia-smi &> /dev/null; then
    nvidia-smi
else
    echo "⚠️ nvidia-smi not found. If you're running in a container, check GPU usage from the host."
fi

echo -e "\n🎯 Final Status:"
python -c "
from llama_cpp import llama_cpp
has_cublas = hasattr(llama_cpp, '_LLAMA_CUBLAS')
if has_cublas and llama_cpp._LLAMA_CUBLAS:
    print('✅ SUCCESS: llama-cpp-python is installed with CUDA support!')
    exit(0)
else:
    print('❌ FAILURE: llama-cpp-python is NOT installed with CUDA support.')
    exit(1)
"

if [ $? -eq 0 ]; then
    echo -e "\n🎉 GPU acceleration is now enabled for llama-cpp-python!"
    echo "You should see significant performance improvements during inference."
    echo "To verify during runtime, check for CUDA/GPU messages in the application logs."
else
    echo -e "\n❌ GPU acceleration setup failed."
    echo "Please check the error messages above and try again."
    echo "You may need to check your CUDA installation or try the manual installation steps."
fi
