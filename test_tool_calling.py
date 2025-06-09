#!/usr/bin/env python3
"""Test actual tool calling functionality."""

import asyncio
import os
import sys

# Add the project root to Python path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Now import required modules
from app.config import get_config
from app.llm_optimized import LLMOptimized

# Import the main patch module FIRST
from app.main_patch import logger
from app.tool import Bash, BrowserUseTool, StrReplaceEditor, Terminate, ToolCollection


async def test_tool_calling():
    print("Setting up LLM and tools...")

    try:
        # Create an LLM instance with proper config
        config = get_config()
        llm = LLMOptimized(config.llm)
        print("✅ LLMOptimized instance created successfully")

        # Create a tool collection like the application does
        tool_collection = ToolCollection(
            BrowserUseTool(), Bash(), StrReplaceEditor(), Terminate()
        )
        tools = tool_collection.to_params()
        print(f"✅ Available tools: {[tool['function']['name'] for tool in tools]}")

        # Test a simple prompt that should trigger tool use
        test_prompt = "Create a simple HTML file called hello.html with basic structure"

        print(f"Testing ask_tool with: {test_prompt}")

        # This is where the error should no longer happen
        result = await llm.ask_tool(
            messages=[{"role": "user", "content": test_prompt}],
            tools=tools,
            max_retries=1,
            timeout=30,
        )

        print("✅ Tool call completed successfully!")
        print(f"Result keys: {list(result.keys())}")
        print(f"Content length: {len(result.get('content', ''))}")
        print(f"Tool calls: {len(result.get('tool_calls', []))}")

        return True

    except Exception as e:
        print(f"❌ Error occurred: {e}")
        print(f"Error type: {type(e)}")
        import traceback

        traceback.print_exc()
        return False


if __name__ == "__main__":
    success = asyncio.run(test_tool_calling())
    if success:
        print("\n🎉 Tool calling test PASSED!")
    else:
        print("\n💥 Tool calling test FAILED!")
