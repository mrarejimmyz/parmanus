"""
Compatibility patch for handling unsupported model architectures like mllama.
This module provides fallback options and error handling for unsupported models.
"""

import json
import logging
import os
from pathlib import Path
from typing import Any, Dict, Optional

logger = logging.getLogger(__name__)


class ModelCompatibilityError(Exception):
    """Raised when a model architecture is not supported."""

    pass


def suggest_alternatives_for_mllama(model_path: str) -> Dict[str, Any]:
    """
    Suggest alternatives when mllama architecture is detected.
    """
    suggestions = {
        "architecture": "mllama",
        "status": "unsupported",
        "alternatives": [
            {
                "option": "ollama",
                "description": "Use Ollama backend which supports Llama 3.2 Vision",
                "setup": [
                    "1. Install Ollama from https://ollama.ai",
                    "2. Run: ollama pull llama3.2-vision:11b",
                    "3. Update config to use ollama backend",
                ],
            },
            {
                "option": "convert",
                "description": "Convert to supported format",
                "setup": [
                    "1. Use llama.cpp converter with newer version",
                    "2. Convert to standard llama format if possible",
                    "3. Use text-only capabilities",
                ],
            },
            {
                "option": "upgrade",
                "description": "Upgrade llama-cpp-python",
                "setup": [
                    "1. Check for newer versions with mllama support",
                    "2. Build from source if needed",
                    "3. Use development branch",
                ],
            },
        ],
        "workaround": "Use text-only model for now",
    }

    return suggestions


def handle_mllama_fallback(settings, gpu_manager=None):
    """
    Handle fallback for mllama models by suggesting alternatives.
    """
    error_message = """
    ╔══════════════════════════════════════════════════════════════════════════════════╗
    ║                          MODEL ARCHITECTURE NOT SUPPORTED                        ║
    ╠══════════════════════════════════════════════════════════════════════════════════╣
    ║                                                                                  ║
    ║  Your Llama 3.2 Vision model uses the 'mllama' architecture which is not       ║
    ║  currently supported by llama-cpp-python 0.3.9.                                ║
    ║                                                                                  ║
    ║  RECOMMENDED SOLUTIONS:                                                          ║
    ║                                                                                  ║
    ║  1. 🚀 USE OLLAMA (RECOMMENDED):                                                ║
    ║     • Install Ollama: https://ollama.ai                                        ║
    ║     • Run: ollama pull llama3.2-vision:11b                                     ║
    ║     • Update config.toml to use ollama backend                                 ║
    ║                                                                                  ║
    ║  2. 🔄 WAIT FOR UPDATE:                                                         ║
    ║     • Wait for llama-cpp-python to support mllama                              ║
    ║     • Check for updates regularly                                               ║
    ║                                                                                  ║
    ║  3. 📝 USE TEXT-ONLY:                                                          ║
    ║     • Use a text-only Llama 3.2 model                                         ║
    ║     • Vision features will be disabled                                          ║
    ║                                                                                  ║
    ╚══════════════════════════════════════════════════════════════════════════════════╝
    """

    print(error_message)
    logger.error(
        "mllama architecture not supported by current llama-cpp-python version"
    )

    # Provide user with specific next steps
    alternatives = suggest_alternatives_for_mllama(settings.model_path)

    return None, alternatives


def patch_llm_for_compatibility():
    """
    Apply compatibility patches to the LLM classes.
    """
    try:
        from app.llm_optimized import LLMOptimized

        # Store original methods
        original_load_text_model = LLMOptimized._load_text_model
        original_load_cpu_fallback = LLMOptimized._load_cpu_fallback

        def patched_load_text_model(self):
            """Patched version that handles unsupported architectures."""
            try:
                return original_load_text_model(self)
            except Exception as e:
                error_str = str(e).lower()
                if "unknown model architecture" in error_str and "mllama" in error_str:
                    # Handle mllama specifically
                    model, alternatives = handle_mllama_fallback(
                        self.settings, self.gpu_manager
                    )
                    raise ModelCompatibilityError(
                        f"Unsupported model architecture 'mllama'. "
                        f"Please use Ollama or wait for llama-cpp-python update. "
                        f"Alternatives: {alternatives}"
                    )
                else:
                    # Re-raise other errors
                    raise

        def patched_load_cpu_fallback(self):
            """Patched CPU fallback that also handles unsupported architectures."""
            try:
                return original_load_cpu_fallback(self)
            except Exception as e:
                error_str = str(e).lower()
                if "unknown model architecture" in error_str and "mllama" in error_str:
                    # Handle mllama specifically
                    model, alternatives = handle_mllama_fallback(
                        self.settings, self.gpu_manager
                    )
                    raise ModelCompatibilityError(
                        f"Unsupported model architecture 'mllama'. "
                        f"Please use Ollama or wait for llama-cpp-python update. "
                        f"Alternatives: {alternatives}"
                    )
                else:
                    # Re-raise other errors
                    raise

        # Apply patches
        LLMOptimized._load_text_model = patched_load_text_model
        LLMOptimized._load_cpu_fallback = patched_load_cpu_fallback
        logger.info("Applied compatibility patch for unsupported model architectures")

    except ImportError:
        logger.warning("Could not apply compatibility patch - LLMOptimized not found")
