import atexit
import signal
import sys

from app.llm import LLM

# Import the tool patch
from app.llm_tool_patch import _parse_tool_calls, ask_tool
from app.logger import logger

# Patch the LLM class with the missing methods
LLM.ask_tool = ask_tool
LLM._parse_tool_calls = _parse_tool_calls

# Global model cache reference for cleanup
MODEL_CACHE = {}


def signal_handler(sig, frame):
    """Handle interrupt signals gracefully."""
    logger.warning("Interrupt signal received. Cleaning up resources...")

    # Clean up model resources
    try:
        LLM.cleanup_all_models()
    except Exception as e:
        logger.error(f"Error during model cleanup: {e}")

    # Clean up any other resources
    logger.info("Cleanup complete. Exiting.")
    sys.exit(0)


def cleanup_handler():
    """Handle cleanup on normal exit."""
    logger.info("Application exiting. Cleaning up resources...")

    # Clean up model resources
    try:
        LLM.cleanup_all_models()
    except Exception as e:
        logger.error(f"Error during model cleanup: {e}")

    logger.info("Cleanup complete.")


# Register signal handlers
signal.signal(signal.SIGINT, signal_handler)
signal.signal(signal.SIGTERM, signal_handler)

# Register exit handler
atexit.register(cleanup_handler)

# Log initialization
logger.info("Interrupt handlers and cleanup procedures initialized.")
