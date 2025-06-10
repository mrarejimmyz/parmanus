import argparse
import asyncio
import os
import sys
import time

# Import agent router and voice modules
from app.agent.router import AgentRouter

# Import configuration
from app.config import get_config
from app.gpu_manager import get_gpu_manager

# Import compatibility patch BEFORE LLM classes
from app.llm_compatibility_patch import (
    ModelCompatibilityError,
    patch_llm_for_compatibility,
)

# Apply compatibility patches
patch_llm_for_compatibility()

from app.llm_factory import create_llm_async

# Import LLM classes AFTER patches are applied
from app.llm_optimized import LLMOptimized

# Import the main patch module FIRST to ensure all patches are applied
from app.main_patch import logger
from app.manus import Manus  # Add Manus import

# Import memory system
from app.memory import Memory
from app.voice.stt import SpeechToText
from app.voice.tts import TextToSpeech


async def main():
    # Initialize configuration and GPU manager
    config = get_config()
    gpu_manager = get_gpu_manager()

    # Configure GPU memory limits
    gpu_manager.configure_memory_limits(text_limit=6.0, vision_limit=0.0)

    try:
        # Initialize LLM using factory (supports both Ollama and llama.cpp)
        llm = await create_llm_async(settings=config.llm)

        # For llama.cpp backend, set GPU manager
        if hasattr(llm, "gpu_manager"):
            llm.gpu_manager = gpu_manager

        # Initialize router with the created LLM (it will handle agent creation internally)
        router = AgentRouter(llm=llm)

        # Preload model if it's llama.cpp backend
        model = None
        if hasattr(llm, "_preload_text_model"):
            model = await llm._preload_text_model()
            if not model:
                raise RuntimeError("Failed to load language model")

        # Main interaction loop
        while True:
            user_input = input("Enter your request (or 'quit' to exit): ")
            if user_input.lower() == "quit":
                break

            try:
                result = await router.route(user_input)
                print(result)
            except Exception as e:
                logger.error(f"Error processing request: {e}")

    except ModelCompatibilityError as e:
        logger.error(f"Model compatibility error: {e}")
        print("\n🚨 Your Llama 3.2 Vision model requires newer software support.")
        print("🔧 Please follow the suggestions above to use your model.")
        print(
            "💡 For immediate use, consider installing Ollama and using ollama backend."
        )
    except Exception as e:
        logger.error(f"Application error: {e}")
    finally:
        await gpu_manager.cleanup()


if __name__ == "__main__":
    asyncio.run(main())
