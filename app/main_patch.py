"""
Main patch module to ensure consistent patching across all entry points.
This module should be imported by all entry points before any LLM instantiation.
"""

import sys
import os
import logging
import types

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Import and apply patches
try:
    # Import the tool patch functions and patch function
    from app.llm_tool_patch import patch_llm_class
    
    # Apply the patch to ensure methods are properly bound
    patch_llm_class()
    logger.info("LLM tool methods patched successfully")
        
    # Import cleanup handlers to ensure they're registered
    from app.cleanup import signal_handler, cleanup_handler
    logger.info("Cleanup handlers registered")
    
except Exception as e:
    logger.error(f"Error applying patches: {e}")
    raise
