import importlib.util
import os
import sys
from pathlib import Path

# Add the main directory to Python path
root_dir = Path(__file__).parent.absolute()
sys.path.insert(0, str(root_dir))

# Import cleanup handlers first
from app.cleanup import cleanup_handler, signal_handler

# Import and apply patches
from app.llm_tool_patch_improved import patch_llm_class

patch_llm_class()  # Apply patches before importing other modules

# Import main components
from app.agent.manus import Manus
from app.logger import logger


async def main():
    """Main entry point with improved error handling."""
    try:
        # Initialize agent
        agent = Manus()
        await agent.run()
    except KeyboardInterrupt:
        logger.info("Received interrupt signal, cleaning up...")
        cleanup_handler()
    except Exception as e:
        logger.error(f"Error in main: {e}", exc_info=True)
        cleanup_handler()
        sys.exit(1)


if __name__ == "__main__":
    import asyncio

    asyncio.run(main())
