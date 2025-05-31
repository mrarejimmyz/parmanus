# Docker GPU Configuration Guide for ParManus

This document provides guidance on ensuring proper GPU support in the ParManus Docker container.

## Docker Configuration

The Docker configuration has been updated to use CUDA-enabled pre-built wheels for llama-cpp-python, which should provide GPU acceleration without build issues.

### Key Components

1. **Base Image**: `nvidia/cuda:12.2.2-devel-ubuntu22.04`
   - Includes CUDA development tools and runtime libraries

2. **llama-cpp-python Installation**:
   ```dockerfile
   RUN pip install --no-cache-dir llama-cpp-python --extra-index-url https://abetlen.github.io/llama-cpp-python/whl/cu122
   ```
   - Uses pre-built wheels with CUDA 12.2 support
   - Avoids compilation issues while maintaining GPU acceleration

3. **Docker Compose Configuration**:
   ```yaml
   environment:
     - NVIDIA_VISIBLE_DEVICES=all
     - NVIDIA_DRIVER_CAPABILITIES=compute,utility
   deploy:
     resources:
       reservations:
         devices:
           - driver: nvidia
             count: all
             capabilities: [ gpu ]
   ```
   - Properly configures GPU passthrough to the container

## Verification Steps

To verify GPU support is working correctly:

1. Build and start the container:
   ```bash
   docker-compose up --build -d
   ```

2. Check GPU visibility inside the container:
   ```bash
   docker exec -it parmanus nvidia-smi
   ```
   - Should show your GPU(s) and CUDA version

3. Verify llama-cpp-python CUDA support:
   ```bash
   docker exec -it parmanus python -c "from llama_cpp import Llama; print(f'CUDA available: {Llama.get_cuda_device_count() > 0}')"
   ```
   - Should output "CUDA available: True"

## Host Requirements

For GPU support to work:

1. Host machine must have NVIDIA drivers installed
2. Docker must have the NVIDIA Container Toolkit installed:
   ```bash
   # Install NVIDIA Container Toolkit
   distribution=$(. /etc/os-release;echo $ID$VERSION_ID)
   curl -s -L https://nvidia.github.io/nvidia-docker/gpgkey | sudo apt-key add -
   curl -s -L https://nvidia.github.io/nvidia-docker/$distribution/nvidia-docker.list | sudo tee /etc/apt/sources.list.d/nvidia-docker.list
   sudo apt-get update && sudo apt-get install -y nvidia-container-toolkit
   sudo systemctl restart docker
   ```

3. Test with a simple container:
   ```bash
   docker run --gpus all nvidia/cuda:12.2.2-base-ubuntu22.04 nvidia-smi
   ```

## Troubleshooting

If GPU support is not working:

1. Ensure host NVIDIA drivers match or exceed the CUDA version in the container
2. Verify the NVIDIA Container Toolkit is properly installed
3. Check Docker daemon configuration includes the NVIDIA runtime
4. Ensure the user has permissions to access the GPU devices
