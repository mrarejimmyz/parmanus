FROM nvidia/cuda:12.2.2-base-ubuntu22.04

# Environment setup
ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHON_VERSION=3.12

# Fix repository connectivity issues by using different mirrors and retries
RUN sed -i 's|http://archive.ubuntu.com|http://mirror.kakao.com|g' /etc/apt/sources.list && \
    sed -i 's|http://security.ubuntu.com|http://mirror.kakao.com|g' /etc/apt/sources.list

# Install system packages in stages to isolate failures
RUN apt-get update --fix-missing && apt-get install -y --no-install-recommends \
    software-properties-common \
    curl \
    wget \
    ca-certificates \
    git \
    && apt-get clean

# Add deadsnakes PPA for Python 3.12
RUN add-apt-repository ppa:deadsnakes/ppa -y && apt-get update

# Install Python 3.12 and core dependencies
RUN apt-get install -y --no-install-recommends \
    python3.12 \
    python3.12-venv \
    python3.12-distutils \
    python3.12-dev \
    python3-pip \
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
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Set up Python symlinks and create virtual environment
RUN ln -sf /usr/bin/python3.12 /usr/bin/python && \
    ln -sf /usr/bin/python3.12 /usr/bin/python3 && \
    python3.12 -m pip install --upgrade pip && \
    python3.12 -m venv /opt/venv

# Activate virtual environment for all subsequent commands
ENV PATH="/opt/venv/bin:$PATH"

# Install PyTorch with CUDA 12.1 (compatible with Python 3.12)
RUN pip install --no-cache-dir torch==2.3.1 torchvision==0.18.1 --index-url https://download.pytorch.org/whl/cu121

# Install Ollama for local model inference
RUN curl -fsSL https://ollama.com/install.sh | sh

# Set working directory
WORKDIR /app/OpenManus

# Copy requirements and install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Install browser-use and Playwright browsers
RUN pip install --no-cache-dir "browser-use[memory]" && \
    playwright install chromium --with-deps

# Create models directory
RUN mkdir -p /models

# Create default config for Ollama
RUN mkdir -p /app/OpenManus/config && \
    echo '# Global LLM configuration\n\
[llm]\n\
api_type = "ollama"\n\
model = "llama3.2"\n\
base_url = "http://localhost:11434/v1"\n\
api_key = "ollama"\n\
max_tokens = 4096\n\
temperature = 0.0\n\
\n\
[llm.vision]\n\
api_type = "ollama"\n\
model = "llama3.2-vision"\n\
base_url = "http://localhost:11434/v1"\n\
api_key = "ollama"\n\
max_tokens = 4096\n\
temperature = 0.0' > /app/OpenManus/config/config.toml

# Copy rest of the code
COPY . .

# Create startup script to pull models and start services
RUN echo '#!/bin/bash\n\
# Start Ollama in the background\n\
ollama serve &\n\
\n\
# Wait for Ollama to start\n\
echo "Waiting for Ollama to start..."\n\
sleep 5\n\
\n\
# Pull models if they don\'t exist\n\
if ! ollama list | grep -q "llama3.2"; then\n\
    echo "Pulling llama3.2 model..."\n\
    ollama pull llama3.2\n\
fi\n\
\n\
if ! ollama list | grep -q "llama3.2-vision"; then\n\
    echo "Pulling llama3.2-vision model..."\n\
    ollama pull llama3.2-vision\n\
fi\n\
\n\
# Start Xvfb for headless browser\n\
Xvfb :99 -screen 0 1920x1080x24 &\n\
\n\
# Execute the command passed to the script\n\
exec "$@"\n' > /entrypoint.sh && \
    chmod +x /entrypoint.sh

# Set environment variables for headless browser operation
ENV DISPLAY=:99
ENV PLAYWRIGHT_BROWSERS_PATH=/ms-playwright

# Set default command
ENTRYPOINT ["/entrypoint.sh"]
CMD ["python", "main.py"]
