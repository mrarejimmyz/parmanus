"""Apply fixes for tool calling issues."""

import os
import shutil
import sys
from pathlib import Path


# Backup original files
def backup_file(filepath):
    """Create a backup of a file with timestamp."""
    if os.path.exists(filepath):
        timestamp = Path(filepath).stat().st_mtime
        backup_path = f"{filepath}.backup_{int(timestamp)}"
        shutil.copy2(filepath, backup_path)
        print(f"Created backup: {backup_path}")


# Apply patches
def apply_patches():
    """Apply all necessary patches."""

    # Get root directory
    root_dir = os.path.dirname(os.path.abspath(__file__))

    # Files to patch
    files = [
        "app/llm_tool_patch.py",
        "app/llm_tool_patch_optimized.py",
        "main_optimized.py",
    ]

    # Create backups
    for file in files:
        filepath = os.path.join(root_dir, file)
        backup_file(filepath)

    # Copy new implementation
    patch_fix_path = os.path.join(root_dir, "app/llm_tool_patch_fix.py")
    for file in files[:2]:  # Only patch the tool patch files
        filepath = os.path.join(root_dir, file)
        shutil.copy2(patch_fix_path, filepath)
        print(f"Applied patch to: {filepath}")

    # Update main_optimized.py
    main_optimized_path = os.path.join(root_dir, "main_optimized.py")
    with open(main_optimized_path, "r") as f:
        content = f.read()

    # Add proper imports and initialization
    updated_content = '''import os
import sys
import importlib.util
from pathlib import Path

# Add the main directory to Python path
root_dir = Path(__file__).parent.absolute()
sys.path.insert(0, str(root_dir))

# Import cleanup handlers first
from app.cleanup import signal_handler, cleanup_handler

# Import and apply patches
from app.llm_tool_patch_fix import patch_llm_class
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
'''

    with open(main_optimized_path, "w") as f:
        f.write(updated_content)
    print(f"Updated: {main_optimized_path}")


if __name__ == "__main__":
    apply_patches()
