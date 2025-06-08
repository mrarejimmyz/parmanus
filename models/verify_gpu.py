import os
import sys

from llama_cpp import Llama
from llama_cpp.llama_chat_format import Llava16ChatHandler

# --- Configuration for your models (matching your config.toml) ---
# Main LLM model (llama-jb)
LLM_MODEL_NAME = "llama-jb"
LLM_MODEL_PATH = os.path.join("models", "llama-jb.gguf")

# LLaVA Vision model (main LLM component for LLaVA)
LLAVA_MODEL_NAME = "llava-v1.6-mistral-7b"
LLAVA_MODEL_PATH = os.path.join("models", "llava-model.gguf")

# CLIP Vision Projector model (mmproj component for LLaVA)
CLIP_MODEL_PATH = os.path.join("models", "mmproj-model.gguf")

# Common Llama.cpp parameters (matching your config.toml where applicable)
N_GPU_LAYERS = (
    -1
)  # Offload all layers to GPU. Set to 0 to force CPU for testing if needed.
N_CTX = 8192  # Context window size for LLaVA (from your config)
TEMPERATURE_LLM = 0.0  # For llama-jb (from your config)
TEMPERATURE_LLAVA = 0.1  # For LLaVA (from your config)
LOGITS_ALL = True  # Essential for LLaVA (from your config)


def print_status(message, color="white"):
    """Helper function to print messages with color."""
    colors = {
        "green": "\033[92m",
        "red": "\033[91m",
        "yellow": "\033[93m",
        "cyan": "\033[96m",
        "white": "\033[0m",
    }
    # Check if we are in a terminal that supports colors, otherwise strip them
    if sys.stdout.isatty():
        print(f"{colors.get(color, colors['white'])}{message}{colors['white']}")
    else:
        print(message)


def verify_gpu_and_models():
    print_status(
        "--- Verifying llama-cpp-python GPU Support and Model Loading ---", "cyan"
    )

    # 1. We assume GPU support is compiled if n_gpu_layers is set and no errors.
    # The true test will be whether models load successfully with n_gpu_layers=-1.
    print_status(
        "\n1. Proceeding assuming llama-cpp-python was compiled with CUDA support.",
        "yellow",
    )
    print_status(
        "   When n_gpu_layers=-1, llama-cpp-python will attempt to offload all layers to GPU.",
        "white",
    )

    # 2. Verify existence of model files
    print_status("\n2. Checking for model files...", "yellow")
    all_files_exist = True

    # Check LLM model
    if os.path.exists(LLM_MODEL_PATH):
        print_status(f"   ✅ {LLM_MODEL_PATH} found.", "green")
    else:
        print_status(f"   ❌ {LLM_MODEL_PATH} NOT found. Please download it.", "red")
        all_files_exist = False

    # Check LLaVA LLM model
    if os.path.exists(LLAVA_MODEL_PATH):
        print_status(f"   ✅ {LLAVA_MODEL_PATH} found.", "green")
    else:
        print_status(
            f"   ❌ {LLAVA_MODEL_PATH} NOT found. Please download the LLaVA LLM GGUF (e.g., llava-v1.x-model.gguf).",
            "red",
        )
        all_files_exist = False

    # Check CLIP vision projector model
    if os.path.exists(CLIP_MODEL_PATH):
        print_status(f"   ✅ {CLIP_MODEL_PATH} found.", "green")
    else:
        print_status(
            f"   ❌ {CLIP_MODEL_PATH} NOT found. Please download the CLIP projector GGUF (the 'mmproj' file).",
            "red",
        )
        print_status(
            f"      (e.g., llava-v1.x-model-mmproj.gguf from HuggingFace, renamed to {os.path.basename(CLIP_MODEL_PATH)})",
            "red",
        )
        all_files_exist = False

    if not all_files_exist:
        print_status(
            "\nSome model files are missing. Please download them into the 'models' folder.",
            "red",
        )
        return False

    # 3. Attempt to load the text-only LLM (llama-jb)
    print_status(
        f"\n3. Attempting to load {LLM_MODEL_NAME} ({LLM_MODEL_PATH})...", "yellow"
    )
    llm_text_only = None  # Initialize to None
    try:
        # For a text-only model, we don't need chat_handler or logits_all
        llm_text_only = Llama(
            model_path=LLM_MODEL_PATH,
            n_gpu_layers=N_GPU_LAYERS,
            n_ctx=4096,  # Common context size for general LLMs
            temperature=TEMPERATURE_LLM,
            verbose=True,  # Set to True for more loading details from llama-cpp-python
        )
        print_status(
            f"   ✅ {LLM_MODEL_NAME} loaded successfully as a text-only model!", "green"
        )

        # Try a quick inference to ensure it works
        test_prompt = "Hello, what is your purpose?"
        print_status(f"   -> Testing inference with '{test_prompt}'...", "white")
        response = llm_text_only(test_prompt, max_tokens=64, stop=["\n"], echo=False)
        print_status(
            f"   -> Response snippet: {response['choices'][0]['text'].strip()[:100]}...",
            "white",
        )

    except Exception as e:
        print_status(f"   ❌ Failed to load {LLM_MODEL_NAME}: {e}", "red")
        print_status(
            "      Possible reasons: Corrupted file, incompatible GGUF version, or insufficient VRAM/RAM.",
            "red",
        )
        print_status(
            "      If you see 'CUDA' related errors here (e.g., related to GPU memory), your llama-cpp-python compilation might be incorrect or you're out of VRAM.",
            "red",
        )
        return False
    finally:
        # Explicitly delete the model instance to free memory before loading the next
        if llm_text_only:
            del llm_text_only

    # 4. Attempt to load the LLaVA multimodal model
    print_status(
        f"\n4. Attempting to load {LLAVA_MODEL_NAME} ({LLAVA_MODEL_PATH}) with vision...",
        "yellow",
    )
    llm_llava = None  # Initialize to None
    try:
        # Use the correct CLIP_MODEL_PATH for the mmproj file
        llava_chat_handler = Llava16ChatHandler(clip_model_path=CLIP_MODEL_PATH)
        llm_llava = Llama(
            model_path=LLAVA_MODEL_PATH,
            chat_handler=llava_chat_handler,
            n_gpu_layers=N_GPU_LAYERS,
            n_ctx=N_CTX,
            logits_all=LOGITS_ALL,
            temperature=TEMPERATURE_LLAVA,
            verbose=True,  # Set to True to see CLIP_CUDA backend info during loading
        )
        print_status(
            f"   ✅ {LLAVA_MODEL_NAME} loaded successfully as a multimodal model!",
            "green",
        )

    except Exception as e:
        print_status(
            f"   ❌ Failed to load {LLAVA_MODEL_NAME} or its vision component: {e}",
            "red",
        )
        print_status(
            "      Possible reasons: Corrupted GGUF files, incompatible mmproj/LLM pair, or insufficient VRAM/RAM.",
            "red",
        )
        print_status(
            f"      Ensure the CLIP_MODEL_PATH ({CLIP_MODEL_PATH}) is correct (it should be the 'mmproj' file) and compatible with the LLaVA LLM model ({LLAVA_MODEL_PATH}).",
            "red",
        )
        print_status(
            "      If you see 'CUDA' related errors here (e.g., related to GPU memory), your llama-cpp-python compilation might be incorrect or you're out of VRAM.",
            "red",
        )
        return False
    finally:
        # Explicitly delete the model instance to free memory
        if llm_llava:
            del llm_llava

    print_status("\n--- Verification Complete! ---", "cyan")
    print_status(
        "If all models loaded successfully without 'Failed to load' errors, your setup is likely correct for GPU offloading (as n_gpu_layers=-1 was used during loading).",
        "green",
    )
    print_status(
        "If you encounter out-of-memory errors later, you may need to reduce n_gpu_layers or use a smaller model.",
        "yellow",
    )
    return True


if __name__ == "__main__":
    verify_gpu_and_models()
