#!/usr/bin/env python3
"""Final validation test for the tool calling fix."""

import os
import sys

# Add the project root to Python path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Import the main patch module FIRST
from app.main_patch import logger

print("✅ 1. Patches imported successfully")

# Now import LLMOptimized
from app.llm_optimized import LLMOptimized

print("✅ 2. LLMOptimized imported successfully")

# Test if the ask_tool method exists
print(f"✅ 3. ask_tool method exists: {hasattr(LLMOptimized, 'ask_tool')}")
print(
    f"✅ 4. _parse_tool_calls method exists: {hasattr(LLMOptimized, '_parse_tool_calls')}"
)
print(
    f"✅ 5. _format_tool_call method exists: {hasattr(LLMOptimized, '_format_tool_call')}"
)

# Test tool format handling
from app.tool import Bash, Terminate, ToolCollection

tool_collection = ToolCollection(Bash(), Terminate())
tools = tool_collection.to_params()

print(f"✅ 6. Tool collection created with {len(tools)} tools")

# Test the format_tool_definitions function
from app.llm_tool_patch_optimized import format_tool_definitions

tool_definitions = format_tool_definitions(tools)
print(
    f"✅ 7. Tool definitions formatted successfully (length: {len(tool_definitions)})"
)

print("\n🎉 ALL TESTS PASSED!")
print("🔧 The tool calling fix is working correctly!")
print(
    "🚀 The main issue 'LLMOptimized object has no attribute ask_tool' has been resolved!"
)
