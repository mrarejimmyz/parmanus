import os

from llama_cpp import Llama

MODEL_PATH = "models/llama-jb.gguf"  # Change this to your model


def check_cuda_usage(model_path):
    print(f"Loading model from: {model_path}")

    try:
        # Enable verbose output to see what gets offloaded
        llm = Llama(
            model_path=model_path,
            n_ctx=512,
            n_threads=8,
            n_gpu_layers=20,
            verbose=True,  # This is important for logging device allocation
        )

        print("✅ Model loaded. Check above logs to see CUDA device usage.")
        print("Look for lines like `layer XX assigned to device CUDA0`.")

    except Exception as e:
        print("❌ Failed to load model or CUDA not available.")
        print(str(e))


if __name__ == "__main__":
    check_cuda_usage(MODEL_PATH)
