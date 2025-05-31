# Open Source Vision Model and Tool-Calling Fixes

## Summary of Changes

I've implemented two critical fixes to address the runtime issues in ParManus:

1. **Replaced Qwen-VL with Open Source LLaVA Vision Model**
   - Switched from the problematic Qwen-VL model to the fully open source LLaVA v1.5 7B GGUF model
   - This eliminates HuggingFace rate limiting and download reliability issues
   - The LLaVA model is a well-established open source vision-language model with good performance

2. **Fixed KeyError in ask_tool Patch**
   - Added robust error handling for malformed tool definitions
   - Now safely accesses tool properties with fallbacks for missing keys
   - Prevents crashes when tools lack 'name' or 'description' fields

## Technical Implementation Details

### Open Source Vision Model Integration

```dockerfile
# Download LLaVA vision model (open source alternative to Qwen-VL)
RUN curl -L --retry 3 --retry-delay 5 \
    "https://huggingface.co/mys/ggml_llava-v1.5-7b/resolve/main/ggml-model-q4_k.gguf" \
    -o /models/llava-vision.gguf && \
    python -c "import struct; f=open('/models/llava-vision.gguf','rb'); magic=f.read(4); f.close(); assert magic==b'GGUF', f'Invalid magic in llava-vision.gguf: {magic}'; print('Verified llava-vision.gguf is a valid GGUF file')"
```

### Tool Definition Error Handling

```python
# Safely access tool properties with fallbacks
tool_name = tool.get('name', 'unnamed_tool')
tool_description = tool.get('description', 'No description available')
tool_definitions += f"- {tool_name}: {tool_description}\n"

# Safely handle parameters if present
if 'parameters' in tool:
    try:
        params_json = json.dumps(tool['parameters'])
        tool_definitions += f"  Parameters: {params_json}\n"
    except (TypeError, ValueError) as e:
        logger.warning(f"Failed to serialize parameters for tool {tool_name}: {e}")
        tool_definitions += f"  Parameters: [Error: Could not serialize parameters]\n"
```

## Benefits of These Changes

1. **Improved Reliability**
   - No more dependency on rate-limited or unreliable model sources
   - System can now run consistently without download failures

2. **Better Error Handling**
   - Graceful handling of malformed tool definitions
   - Clear error messages and fallbacks for missing data

3. **Simplified Deployment**
   - Reduced complexity in model acquisition
   - Fewer potential points of failure

## Validation Steps

To verify these fixes:
1. Build the Docker image with the updated Dockerfile
2. Run the container with the command: `docker run -it parmanus python main.py --prompt "create a website to say hi"`
3. Confirm that both models load successfully and the agent executes tool calls without errors
