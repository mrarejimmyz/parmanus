#!/usr/bin/env python3
"""Debug tool format issue."""

import os
import sys

# Add the project root to Python path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Import the main patch module FIRST
from app.main_patch import logger

# Now import required modules
from app.tool import Bash, BrowserUseTool, StrReplaceEditor, Terminate, ToolCollection


def debug_tool_format():
    print("Debugging tool format...")

    # Create a tool collection like the application does
    tool_collection = ToolCollection(
        BrowserUseTool(), Bash(), StrReplaceEditor(), Terminate()
    )
    tools = tool_collection.to_params()

    print(f"Number of tools: {len(tools)}")
    for i, tool in enumerate(tools):
        print(f"\nTool {i}:")
        print(f"Type: {type(tool)}")
        print(f"Keys: {list(tool.keys())}")
        if isinstance(tool, dict):
            for key, value in tool.items():
                print(f"  {key}: {type(value)} = {str(value)[:100]}...")
        print()


if __name__ == "__main__":
    debug_tool_format()
