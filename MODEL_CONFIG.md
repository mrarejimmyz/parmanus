# OpenManus Local Model Configuration

This document explains the local model configuration implemented in the Dockerfile and docker-compose.yml.

## Overview

The configuration has been updated to use local models via Ollama without requiring API keys:

1. The Dockerfile installs Ollama for local model inference
2. A startup script automatically pulls required models if they don't exist
3. The Docker Compose file sets up a named volume for persistent model storage
4. Models are downloaded once and cached for future container runs

## Models

The following models are used through Ollama:

1. **llama3.2**: Used as the main LLM model
   - Automatically pulled during container startup
   - Used for text generation and reasoning

2. **llama3.2-vision**: Used as the vision model
   - Automatically pulled during container startup
   - Used for image understanding and multimodal tasks

## Usage

To build and run the container with local models:

```bash
docker-compose up -d
```

The first run will download the models if they don't exist. Subsequent runs will reuse the cached models from the named volume.

## Configuration Details

- Models are stored in a named volume `ollama-models` mounted at `/root/.ollama`
- The entrypoint script starts Ollama and pulls models if needed
- NVIDIA GPU support is configured for optimal performance
- No API keys are required for model usage

## Customization

To use different models, you can modify the config.toml file and the entrypoint.sh script to specify different Ollama models.
