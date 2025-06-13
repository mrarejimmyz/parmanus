
"""
Complete ParManus AI Agent System with Full Tool Integration
Optimized for local GGUF models while maintaining all functionality.
"""

import argparse
import asyncio
import json
import os
import sys
import time
import traceback
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

# Core imports
from pydantic import BaseModel, Field
import tomllib

# Conditional imports
try:
    from loguru import logger
except ImportError:
    import logging
    logger = logging.getLogger(__name__)
    logging.basicConfig(level=logging.INFO)

# Import ParManus components
try:
    from app.llm_hybrid import create_llm_with_tools
    from app.agent.manus import Manus
    from app.agent.code import CodeAgent
    from app.agent.browser import BrowserAgent
    from app.config import Config as ParManusConfig
    from app.memory import Memory
    from app.schema import Message, AgentState
    PARMANUS_AVAILABLE = True
except ImportError as e:
    logger.warning(f"ParManus components not fully available: {e}")
    PARMANUS_AVAILABLE = False

# Import refactored components
from app.config import Config, load_config
from app.agent_wrappers import create_agent
from app.agent_router import route_agent
from app.main_utils import initialize_system, display_startup_info, process_prompt


async def main():
    """Main entry point with full functionality."""
    parser = argparse.ArgumentParser(description="ParManus AI Agent - Complete System")
    parser.add_argument("--config", type=str, help="Path to configuration file")
    parser.add_argument("--prompt", type=str, help="Input prompt")
    parser.add_argument("--no-wait", action="store_true", help="Exit immediately if no prompt")
    parser.add_argument("--agent", type=str, help="Specify agent (manus, code, browser, file, planner)")
    parser.add_argument("--api-type", type=str, choices=["local", "ollama"], help="Override API type")
    parser.add_argument("--simple", action="store_true", help="Use simple agents only")
    parser.add_argument("--workspace", type=str, help="Workspace directory")
    parser.add_argument("--max-steps", type=int, help="Maximum agent steps")
    args = parser.parse_args()
    
    try:
        config, llm, memory = await initialize_system(args)
        display_startup_info(config, args, PARMANUS_AVAILABLE)

        processed_cmd_prompt = False
        while True:
            prompt = None
            if not processed_cmd_prompt and args.prompt:
                prompt = args.prompt
                processed_cmd_prompt = True
                logger.info(f"📝 Processing: {prompt[:100]}...")
            else:
                if sys.stdin.isatty():
                    try:
                        prompt = input("\n💬 Enter your prompt (or \'quit\' to exit): ")
                        if prompt.lower() in ["quit", "exit", "bye", "q"]:
                            break
                    except (EOFError, KeyboardInterrupt):
                        if args.no_wait:
                            break
                        continue
                else:
                    if args.no_wait and processed_cmd_prompt:
                        break
                    time.sleep(5)
                    continue
            
            await process_prompt(prompt, args, llm, config, memory, PARMANUS_AVAILABLE)

    except Exception as e:
        logger.error(f"An unhandled error occurred: {e}")
        logger.error(traceback.format_exc())
    finally:
        if 'memory' in locals() and memory:
            memory.save_session()


if __name__ == "__main__":
    asyncio.run(main())


