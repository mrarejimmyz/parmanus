"""
Main patch module to ensure consistent patching across all entry points.
This module should be imported by all entry points before any LLM instantiation.
"""

import sys
import os
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Import and apply patches
try:
    # Import the tool patch functions
    from app.llm_tool_patch import ask_tool, _parse_tool_calls
    from app.llm import LLM
    
    # Ensure the LLM class has the necessary methods
    if not hasattr(LLM, 'ask_tool'):
        LLM.ask_tool = ask_tool
        LLM._parse_tool_calls = _parse_tool_calls
        logger.info("LLM tool methods patched successfully")
    else:
        logger.info("LLM tool methods already patched")
        
    # Import cleanup handlers to ensure they're registered
    from app.cleanup import signal_handler, cleanup_handler
    logger.info("Cleanup handlers registered")
    
except Exception as e:
    logger.error(f"Error applying patches: {e}")
    raise
