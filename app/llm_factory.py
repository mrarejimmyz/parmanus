"""
LLM Factory for ParManus
Handles the creation of appropriate LLM backends based on configuration
"""

from typing import Union

from app.config import LLMSettings
from app.logger import logger


def create_llm(settings: LLMSettings) -> Union["OllamaLLM", "LLMOptimized"]:
    """
    Create the appropriate LLM instance based on the backend configuration.

    Args:
        settings: LLM configuration settings

    Returns:
        LLM instance (either OllamaLLM or LLMOptimized)
    """
    backend = getattr(settings, "backend", "llamacpp").lower()

    if backend == "ollama":
        try:
            from app.llm_ollama import OllamaLLM

            # Check if Ollama is available
            ollama_llm = OllamaLLM(settings)
            if ollama_llm.check_model_availability():
                logger.info(f"Using Ollama backend with model: {settings.model}")
                return ollama_llm
            else:
                logger.warning(
                    f"Ollama model {settings.model} not available, falling back to llama.cpp"
                )
                # Fall through to llama.cpp
        except Exception as e:
            logger.warning(
                f"Failed to initialize Ollama backend: {e}, falling back to llama.cpp"
            )
            # Fall through to llama.cpp

    # Default to llama.cpp backend
    try:
        from app.llm_optimized import LLMOptimized

        logger.info(f"Using llama.cpp backend with model: {settings.model}")
        return LLMOptimized(settings)
    except Exception as e:
        logger.error(f"Failed to initialize llama.cpp backend: {e}")
        raise


async def create_llm_async(settings: LLMSettings) -> Union["OllamaLLM", "LLMOptimized"]:
    """
    Async version of create_llm that can pull Ollama models if needed.

    Args:
        settings: LLM configuration settings

    Returns:
        LLM instance (either OllamaLLM or LLMOptimized)
    """
    backend = getattr(settings, "backend", "llamacpp").lower()

    if backend == "ollama":
        try:
            from app.llm_ollama import OllamaLLM

            # Create Ollama instance
            ollama_llm = OllamaLLM(settings)

            # Check if model is available, if not try to pull it
            if not ollama_llm.check_model_availability():
                logger.info(f"Model {settings.model} not found, attempting to pull...")
                success = await ollama_llm.pull_model_if_needed()
                if not success:
                    logger.warning(
                        "Failed to pull Ollama model, falling back to llama.cpp"
                    )
                    # Fall through to llama.cpp
                else:
                    logger.info(
                        f"Successfully pulled and using Ollama backend with model: {settings.model}"
                    )
                    return ollama_llm
            else:
                logger.info(f"Using Ollama backend with model: {settings.model}")
                return ollama_llm

        except Exception as e:
            logger.warning(
                f"Failed to initialize Ollama backend: {e}, falling back to llama.cpp"
            )
            # Fall through to llama.cpp

    # Default to llama.cpp backend
    try:
        from app.llm_optimized import LLMOptimized

        logger.info(f"Using llama.cpp backend with model: {settings.model}")
        return LLMOptimized(settings)
    except Exception as e:
        logger.error(f"Failed to initialize llama.cpp backend: {e}")
        raise
