# Robust Model Acquisition Guide for ParManus

This document explains the robust model acquisition strategy implemented in ParManus to handle potential issues with downloading large model files from Hugging Face.

## Model Acquisition Strategy

The system now uses a two-phase approach for model acquisition:

### 1. Build-time Acquisition (Llama Model)
- The Llama 3.1 8B model is downloaded during Docker build
- The file is verified to be a valid GGUF file by checking the magic bytes
- If verification fails, the build fails, providing immediate feedback

### 2. Runtime Acquisition (Vision Model)
- The Qwen-VL vision model is downloaded at container startup
- This approach avoids build failures due to temporary Hugging Face issues
- The downloaded file is verified before use
- If verification fails, a clear error message is provided

## How It Works

1. During Docker build:
   - A placeholder file is created for the vision model
   - A download script is created
   - The entrypoint script is configured to check and download the model

2. At container startup:
   - The entrypoint script checks if the vision model exists and is valid
   - If not, it attempts to download and verify the model
   - The container proceeds with or without vision capabilities based on download success

## Manual Model Download (If Needed)

If automatic download fails, you can manually download the model and mount it to the container:

1. Download the model file:
   ```bash
   curl -L "https://huggingface.co/TheBloke/Qwen-VL-Chat-GGUF/resolve/main/qwen-vl-chat.Q5_K_M.gguf" -o qwen-vl-chat.Q5_K_M.gguf
   ```

2. Verify it's a valid GGUF file:
   ```bash
   python -c "import struct; f=open('qwen-vl-chat.Q5_K_M.gguf','rb'); magic=f.read(4); f.close(); print(f'Magic bytes: {magic}')"
   # Should output: Magic bytes: b'GGUF'
   ```

3. Mount the file when running the container:
   ```bash
   docker run -v /path/to/qwen-vl-chat.Q5_K_M.gguf:/models/qwen-vl-7b-awq.gguf parmanus
   ```

4. Or update your docker-compose.yml:
   ```yaml
   volumes:
     - /path/to/qwen-vl-chat.Q5_K_M.gguf:/models/qwen-vl-7b-awq.gguf
     - model-data:/models
   ```

## Alternative Vision Models

If the recommended model continues to have issues, you can try these alternatives:

1. Smaller Qwen-VL models:
   - https://huggingface.co/TheBloke/Qwen-VL-Chat-GGUF/resolve/main/qwen-vl-chat.Q4_K_M.gguf

2. Other vision-language models:
   - LLaVA: https://huggingface.co/mys/ggml_llava-v1.5-7b/resolve/main/ggml-model-q4_k.gguf
   - CogVLM: https://huggingface.co/TheBloke/cogvlm-chat-hf-GGUF/resolve/main/cogvlm-chat-hf.Q4_K_M.gguf

Remember to update the config.toml file if using an alternative model.
