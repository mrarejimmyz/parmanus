#!/usr/bin/env python3
"""Test if the patching is working correctly."""

import os
import sys

# Add the project root to Python path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Import the main patch module FIRST
from app.main_patch import logger

print("Patches imported successfully.")

# Now import LLMOptimized
from app.llm_optimized import LLMOptimized

print("LLMOptimized imported.")

# Test if the ask_tool method exists
llm_instance = None
try:
    # Just check if the method exists in the class
    print(f"ask_tool method exists: {hasattr(LLMOptimized, 'ask_tool')}")
    print(
        f"_parse_tool_calls method exists: {hasattr(LLMOptimized, '_parse_tool_calls')}"
    )
    print(
        f"_format_tool_call method exists: {hasattr(LLMOptimized, '_format_tool_call')}"
    )

    if hasattr(LLMOptimized, "ask_tool"):
        print("✅ Patching successful! ask_tool method is available.")
    else:
        print("❌ Patching failed! ask_tool method is missing.")

except Exception as e:
    print(f"Error testing LLMOptimized: {e}")
