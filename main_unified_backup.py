import argparse
import asyncio
import os
import sys
import time

# Import Manus agent
from app.agent.manus import Manus

# Import agent router
from app.agent.router import AgentRouter

# Import configuration
from app.config import get_config
from app.gpu_manager import get_gpu_manager

# Import LLM factory and classes
from app.llm_factory import create_llm_async
from app.llm_optimized import LLMOptimized

# Import logger
from app.logger import logger

# Import memory system
from app.memory import Memory


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

    except Exception as e:
        logger.error(f"Application error: {e}")
        print(f"\n🚨 Error: {e}")
        print("💡 Please check your configuration and try again.")
    finally:
        await gpu_manager.cleanup()


if __name__ == "__main__":
    asyncio.run(main())
