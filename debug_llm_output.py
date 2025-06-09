#!/usr/bin/env python3
"""Debug script to capture and analyze LLM output causing tool call failures."""

import asyncio
import json
import logging
import os
import sys
from pathlib import Path

# Add the main directory to Python path
root_dir = Path(__file__).parent.absolute()
sys.path.insert(0, str(root_dir))

# Configure logging for detailed output
logging.basicConfig(
    level=logging.DEBUG,
    format="%(asctime)s | %(levelname)s | %(name)s:%(funcName)s:%(lineno)d - %(message)s",
)

from app.cleanup import cleanup_handler, signal_handler
from app.config import config
from app.llm_optimized import LLMOptimized
from app.logger import logger
from app.schema import Message, ToolChoice


async def debug_llm_output():
    """Debug LLM output to identify parsing issues."""

    try:
        # Initialize LLM with debug settings
        llm = LLMOptimized(config.llm)
        logger.info(f"Initialized LLM with model: {llm.model}")

        # Simple test messages
        messages = [
            Message.user_message(
                "What tools do you have available? Please use one of them to help me."
            )
        ]

        # Add some basic tools for testing
        tools = [
            {
                "name": "python_execute",
                "description": "Execute Python code",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "code": {
                            "type": "string",
                            "description": "Python code to execute",
                        }
                    },
                    "required": ["code"],
                },
            },
            {
                "name": "browser_use",
                "description": "Use browser to navigate websites",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "action": {
                            "type": "string",
                            "description": "Browser action to perform",
                        }
                    },
                    "required": ["action"],
                },
            },
        ]

        logger.info("About to call ask_tool with debug logging...")

        # Temporarily patch the parsing function to capture raw output
        original_parse = llm._parse_tool_calls

        def debug_parse_tool_calls(text):
            logger.critical(f"=== RAW LLM OUTPUT ===")
            logger.critical(f"Text: {repr(text)}")
            logger.critical(f"Length: {len(text)}")
            logger.critical(f"First 500 chars: {text[:500]}")
            logger.critical(f"Last 500 chars: {text[-500:]}")
            logger.critical(f"========================")

            # Try original parsing
            try:
                result = original_parse(text)
                logger.critical(f"=== PARSE RESULT ===")
                logger.critical(f"Tool calls: {result}")
                logger.critical(f"Count: {len(result)}")
                logger.critical(f"===================")
                return result
            except Exception as e:
                logger.critical(f"=== PARSE ERROR ===")
                logger.critical(f"Error: {e}")
                logger.critical(f"Error type: {type(e)}")
                logger.critical(f"==================")
                raise

        llm._parse_tool_calls = debug_parse_tool_calls

        # Call with debug output
        result = await llm.ask_tool(
            messages=messages,
            tools=tools,
            tool_choice=ToolChoice.AUTO,
            temp=0.1,
            timeout=30,
        )

        logger.info(f"=== FINAL RESULT ===")
        logger.info(f"Result: {json.dumps(result, indent=2)}")
        logger.info(f"==================")

    except Exception as e:
        logger.error(f"Debug test failed: {e}", exc_info=True)

        # Also print to stdout for visibility
        print(f"\n=== ERROR DETAILS ===")
        print(f"Error: {e}")
        print(f"Error type: {type(e)}")
        print(f"====================\n")

    finally:
        # Cleanup
        if "llm" in locals():
            try:
                llm.cleanup_models()
            except:
                pass


if __name__ == "__main__":
    asyncio.run(debug_llm_output())
