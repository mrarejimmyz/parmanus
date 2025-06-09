#!/usr/bin/env python3
"""
Simple tool calling test
"""

import asyncio
import os
import sys

# Add the app directory to Python path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "app"))

# Import and apply patches first
from app.llm_tool_patch_fix_clean import patch_llm_class

patch_llm_class()

from app.llm import LLM
from app.schema import Message


async def test_tool_calling():
    """Test tool calling with simple tools"""

    print("🔧 Testing tool calling functionality...")

    # Initialize LLM
    llm = LLM()

    # Simple test tool
    test_tools = [
        {
            "type": "function",
            "function": {
                "name": "get_weather",
                "description": "Get the current weather for a city",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "city": {"type": "string", "description": "The city name"}
                    },
                    "required": ["city"],
                },
            },
        }
    ]

    # Test messages
    messages = [Message.user_message("What's the weather like in New York?")]

    try:
        print("📝 Calling ask_tool with REQUIRED tool choice...")
        result = await llm.ask_tool(
            messages=messages, tools=test_tools, tool_choice="required", timeout=60
        )

        print(f"✅ Response received!")
        print(f"📊 Content: {result.get('content', 'No content')}")
        print(f"🛠️ Tool calls: {len(result.get('tool_calls', []))}")

        if result.get("tool_calls"):
            for i, tool_call in enumerate(result["tool_calls"]):
                print(f"  Tool call {i+1}: {tool_call}")
        else:
            print("❌ No tool calls found!")

        return True

    except Exception as e:
        print(f"❌ Error during tool calling: {e}")
        import traceback

        traceback.print_exc()
        return False


if __name__ == "__main__":
    asyncio.run(test_tool_calling())
