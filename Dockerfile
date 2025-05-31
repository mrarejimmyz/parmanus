FROM nvidia/cuda:12.2.2-devel-ubuntu22.04
# Environment setup
ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHON_VERSION=3.11
# Install system packages in stages to isolate failures
RUN apt-get update --fix-missing && apt-get install -y --no-install-recommends \
    software-properties-common \
    curl \
    wget \
    ca-certificates \
    && apt-get clean
# Add deadsnakes PPA for Python 3.11
RUN add-apt-repository ppa:deadsnakes/ppa -y && apt-get update
# Install Python 3.11 and core dependencies
RUN apt-get install -y --no-install-recommends \
    python3.11 \
    python3.11-venv \
    python3.11-distutils \
    python3.11-dev \
    python3-pip \
    git \
    && apt-get clean
# Install browser and GUI dependencies
RUN apt-get install -y --no-install-recommends \
    # OpenGL and graphics
    libgl1 \
    # Core browser dependencies
    libglib2.0-0 \
    libnss3 \
    libnspr4 \
    libdbus-1-3 \
    libatk1.0-0 \
    libatk-bridge2.0-0 \
    libdrm2 \
    libxkbcommon0 \
    libgtk-3-0 \
    libatspi2.0-0 \
    # X11 libraries
    libx11-6 \
    libxcomposite1 \
    libxdamage1 \
    libxext6 \
    libxfixes3 \
    libxrandr2 \
    libxrender1 \
    libxss1 \
    libxtst6 \
    # Additional browser support
    fonts-liberation \
    libappindicator3-1 \
    libasound2 \
    xdg-utils \
    # Virtual display for headless operation
    xvfb \
    x11vnc \
    fluxbox \
    wmctrl \
    # Build dependencies for llama-cpp-python
    build-essential \
    cmake \
    gcc \
    g++ \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*
# Set up Python symlinks and create virtual environment
RUN ln -sf /usr/bin/python3.11 /usr/bin/python && \
    ln -sf /usr/bin/python3.11 /usr/bin/python3 && \
    python3.11 -m pip install --upgrade pip && \
    python3.11 -m venv /opt/venv
# Activate virtual environment for all subsequent commands
ENV PATH="/opt/venv/bin:$PATH"
# Set working directory
WORKDIR /app/ParManus
# Set CUDA environment variables properly
ENV CUDA_HOME="/usr/local/cuda"
ENV CUDACXX="/usr/local/cuda/bin/nvcc"
ENV PATH="${CUDA_HOME}/bin:${PATH}"
ENV LD_LIBRARY_PATH="${CUDA_HOME}/lib64:${LD_LIBRARY_PATH}"
# Install PyTorch with CUDA 12.2 support (matching your CUDA version)
RUN pip install --no-cache-dir torch==2.4.1 torchvision==0.19.1 torchaudio==2.4.1 --index-url https://download.pytorch.org/whl/cu121

# Install CUDA-enabled llama-cpp-python using pre-built wheels
# This avoids compilation issues while still providing GPU support
RUN pip install --no-cache-dir llama-cpp-python --extra-index-url https://abetlen.github.io/llama-cpp-python/whl/cu122

# Copy requirements and install remaining Python dependencies
COPY requirements.txt .
RUN grep -v "llama-cpp-python" requirements.txt > requirements_no_llama.txt && \
    pip install --no-cache-dir -r requirements_no_llama.txt
# Install browser-use and Playwright browsers
RUN pip install --no-cache-dir "browser-use[memory]" && \
    playwright install chromium --with-deps
# Create models directory
RUN mkdir -p /models

# Download Llama model with verification
RUN curl -L --retry 3 --retry-delay 5 \
    "https://huggingface.co/grimjim/Llama-3.1-8B-Instruct-abliterated_via_adapter-GGUF/resolve/main/Llama-3.1-8B-Instruct-abliterated_via_adapter.Q5_K_M.gguf" \
    -o /models/llama-jb.gguf && \
    python -c "import struct; f=open('/models/llama-jb.gguf','rb'); magic=f.read(4); f.close(); assert magic==b'GGUF', f'Invalid magic in llama-jb.gguf: {magic}'; print('Verified llama-jb.gguf is a valid GGUF file')"

# Create a dummy vision model file that will be replaced at runtime
# This allows the Docker build to complete without requiring the vision model
RUN echo "# This is a placeholder file. The actual model will be downloaded at runtime." > /models/qwen-vl-7b-awq.gguf

# Create default config for local models
RUN mkdir -p /app/ParManus/config && \
    echo '# Global LLM configuration\n\
    [llm]\n\
    model = "llama-jb"\n\
    model_path = "/models/llama-jb.gguf"\n\
    max_tokens = 2048\n\
    temperature = 0.0\n\
    \n\
    [llm.vision]\n\
    model = "qwen-vl-7b"\n\
    model_path = "/models/qwen-vl-7b-awq.gguf"\n\
    max_tokens = 2048\n\
    temperature = 0.0' > /app/ParManus/config/config.toml

# Copy rest of the code
COPY . .

# Create model download script
RUN echo '#!/bin/bash\n\
echo "Downloading Qwen-VL model..."\n\
curl -L --retry 5 --retry-delay 10 \
  "https://huggingface.co/TheBloke/Qwen-VL-Chat-GGUF/resolve/main/qwen-vl-chat.Q5_K_M.gguf" \
  -o /models/qwen-vl-7b-awq.gguf.tmp\n\
\n\
# Verify the downloaded file\n\
python -c "import struct; f=open(\"/models/qwen-vl-7b-awq.gguf.tmp\",\"rb\"); magic=f.read(4); f.close(); assert magic==b\"GGUF\", f\"Invalid magic: {magic}\"; print(\"Verified model is a valid GGUF file\")"\n\
\n\
if [ $? -eq 0 ]; then\n\
  mv /models/qwen-vl-7b-awq.gguf.tmp /models/qwen-vl-7b-awq.gguf\n\
  echo "Model downloaded and verified successfully."\n\
else\n\
  echo "Failed to download a valid model. Please download manually."\n\
  exit 1\n\
fi' > /app/ParManus/download_vision_model.sh && \
    chmod +x /app/ParManus/download_vision_model.sh

# Create startup script for Xvfb (virtual display) with lock file cleanup and model download
RUN echo '#!/bin/bash\n\
# Clean up any existing X lock files\n\
rm -f /tmp/.X*-lock\n\
rm -f /tmp/.X11-unix/X*\n\
\n\
# Download vision model if needed\n\
if [ ! -s /models/qwen-vl-7b-awq.gguf ] || [ "$(head -c 4 /models/qwen-vl-7b-awq.gguf)" != "GGUF" ]; then\n\
  echo "Vision model not found or invalid. Downloading..."\n\
  /app/ParManus/download_vision_model.sh\n\
  if [ $? -ne 0 ]; then\n\
    echo "WARNING: Vision model download failed. Vision features will not work."\n\
    echo "Please download the model manually and mount it to /models/qwen-vl-7b-awq.gguf"\n\
  fi\n\
fi\n\
\n\
# Start Xvfb\n\
Xvfb :99 -screen 0 1920x1080x24 &\n\
exec "$@"' > /entrypoint.sh && \
    chmod +x /entrypoint.sh

# Set environment variables for headless browser operation
ENV DISPLAY=:99
ENV PLAYWRIGHT_BROWSERS_PATH=/ms-playwright

# Set default command to use main.py (which now imports the patch module)
ENTRYPOINT ["/entrypoint.sh"]
CMD ["python", "main.py"]
