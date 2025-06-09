#!/usr/bin/env python3
"""
Test tool calling functionality with detailed debugging
"""

import asyncio
import logging
import os
import sys

# Add the app directory to Python path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "app"))

# Import and apply patches first
from app.llm_tool_patch_fix import patch_llm_class

patch_llm_class()

from app.llm import LLM
from app.schema import Message

# Set up logging to see what's happening
logging.basicConfig(
    level=logging.DEBUG, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


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

        # Print raw response for debugging
        print(f"🔍 Raw response keys: {list(result.keys())}")
        print(f"📈 Usage: {result.get('usage', {})}")

    except Exception as e:
        print(f"❌ Error during tool calling: {e}")
        import traceback

        traceback.print_exc()


async def test_simple_completion():
    """Test basic completion for comparison"""

    print("\n🧪 Testing basic completion...")

    llm = LLM()    try:
        # Use ask method for basic completion
        messages = [{"role": "user", "content": "Say hello and describe yourself in one sentence."}]
        response = await llm.ask(
            messages=messages,
            timeout=30
        )

        print(f"✅ Basic completion: {response}")

    except Exception as e:
        print(f"❌ Error during basic completion: {e}")


async def main():
    """Run all tests"""
    await test_simple_completion()
    await test_tool_calling()


if __name__ == "__main__":
    asyncio.run(main())
