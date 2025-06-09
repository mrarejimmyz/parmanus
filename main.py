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
        # Initialize and load LLM with the same GPU manager instance
        llm = LLMOptimized(settings=config.llm, gpu_manager=gpu_manager)

        # Create Manus instance with the initialized LLM and GPU manager
        manus = Manus(llm=llm, gpu_manager=gpu_manager)

        model = await llm._preload_text_model()
        if not model:
            raise RuntimeError("Failed to load language model")

        # Initialize router with the llm from Manus
        router = AgentRouter(llm=manus.llm)

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

    except Exception as e:
        logger.error(f"Application error: {e}")
    finally:
        await gpu_manager.cleanup()


if __name__ == "__main__":
    asyncio.run(main())
