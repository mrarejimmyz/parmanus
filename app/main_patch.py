"""
Main patch module to ensure consistent patching across all entry points.
This module should be imported by all entry points before any LLM instantiation.
"""

import logging
import os
import sys
import types
from typing import Optional

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def initialize_patches(retries: int = 3) -> bool:
    """Initialize all required patches with retries."""
    for attempt in range(retries):
        try:
            # Import required classes
            from app.llm import LLM
            from app.llm_optimized import LLMOptimized

            # Import and apply optimized patches if available
            try:
                from app.llm_tool_patch_optimized import (
                    _format_tool_call,
                    _parse_tool_calls,
                    ask_tool,
                )

                logger.info("Using optimized LLM tool patch")
            except ImportError:
                # Fall back to standard patch
                from app.llm_tool_patch import (
                    _format_tool_call,
                    _parse_tool_calls,
                    ask_tool,
                )

                logger.info("Using standard LLM tool patch")

            # Properly bind methods to both LLM classes
            for cls in [LLM, LLMOptimized]:
                # Required methods - bind as class methods
                cls.ask_tool = ask_tool
                cls._parse_tool_calls = _parse_tool_calls
                cls._format_tool_call = _format_tool_call

                logger.info(f"Successfully patched {cls.__name__} with tool methods")

            # Import and register cleanup handlers
            from app.cleanup import cleanup_handler, signal_handler

            logger.info("Cleanup handlers registered")

            return True

        except ImportError as e:
            logger.warning(f"Import error on attempt {attempt + 1}: {e}")
            if attempt < retries - 1:
                continue
            logger.error("Failed to import required modules")
            if attempt == retries - 1:
                raise

        except Exception as e:
            logger.error(f"Error applying patches on attempt {attempt + 1}: {e}")
            if attempt < retries - 1:
                continue
            raise

    return False


# Initialize patches
if not initialize_patches():
    logger.error("Failed to initialize required patches")
    sys.exit(1)
