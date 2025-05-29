# Model Caching Configuration

This document explains the model caching configuration implemented in the Dockerfile and docker-compose.yml.

## Overview

The configuration has been updated to efficiently handle model downloads and caching:

1. The Dockerfile creates an entrypoint script that conditionally downloads models only if they don't already exist
2. The Docker Compose file sets up a named volume for persistent model storage
3. Models are downloaded once and cached for future container runs

## Models

The following models are used:

1. **Llama-3.1-8B-Instruct**: Downloaded from HuggingFace (grimjim/Llama-3.1-8B-Instruct-abliterated_via_adapter-GGUF)
   - Format: GGUF (Q5_K_M quantization)
   - Stored as: `/models/llama-jb.gguf`

2. **Qwen-VL-7B-AWQ**: Downloaded from HuggingFace (Qwen/Qwen-VL-7B-AWQ)
   - Format: GGUF (AWQ quantization)
   - Stored as: `/models/qwen-vl-7b-awq.gguf`

## Usage

To build and run the container with model caching:

```bash
docker-compose up -d
```

The first run will download the models if they don't exist. Subsequent runs will reuse the cached models from the named volume.

## Configuration Details

- Models are stored in a named volume `model-data` mounted at `/models`
- The entrypoint script checks if models exist before downloading
- NVIDIA GPU support is configured for optimal performance
