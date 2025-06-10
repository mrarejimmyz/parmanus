#!/usr/bin/env python3
"""
Test script for Ollama integration with ParManus
"""

import asyncio
import sys
from pathlib import Path

# Add the main directory to Python path
root_dir = Path(__file__).parent.absolute()
sys.path.insert(0, str(root_dir))

from app.config import get_config
from app.llm_factory import create_llm_async
from app.logger import logger
from app.schema import Message


async def test_ollama_integration():
    """Test the Ollama integration."""
    print("🔧 Testing Ollama integration with ParManus...")

    try:
        # Get configuration
        config = get_config()
        print(f"Using backend: {getattr(config.llm, 'backend', 'llamacpp')}")
        print(f"Model: {config.llm.model}")

        # Create LLM instance using factory
        print("Creating LLM instance...")
        llm = await create_llm_async(config.llm)
        print(f"✅ Successfully created LLM instance: {type(llm).__name__}")

        # Test basic text generation
        print("\n📝 Testing basic text generation...")
        messages = [Message.user_message("Hello! Can you introduce yourself?")]

        response = await llm.ask(messages)
        print(f"✅ Response: {response[:200]}...")

        # Test vision capabilities if it's Ollama
        if hasattr(llm, "vision_enabled") and llm.vision_enabled:
            print("\n👁️ Testing vision capabilities...")
            vision_message = Message.user_message(
                "Describe what capabilities you have for analyzing images."
            )
            vision_response = await llm.ask([vision_message])
            print(f"✅ Vision response: {vision_response[:200]}...")

        # Test tool calling
        print("\n🔧 Testing tool calling capabilities...")
        test_tools = [
            {
                "type": "function",
                "function": {
                    "name": "get_weather",
                    "description": "Get current weather for a city",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "city": {"type": "string", "description": "City name"}
                        },
                        "required": ["city"],
                    },
                },
            }
        ]

        tool_messages = [
            Message.user_message(
                "What's the weather like in New York? Use the get_weather function."
            )
        ]

        if hasattr(llm, "ask_tool"):
            tool_response = await llm.ask_tool(tool_messages, tools=test_tools)
            print(f"✅ Tool response: {tool_response}")
        else:
            print("⚠️ Tool calling not implemented for this backend")

        print("\n🎉 All tests completed successfully!")
        return True

    except Exception as e:
        logger.error(f"Test failed: {e}")
        print(f"❌ Test failed: {e}")
        return False


if __name__ == "__main__":
    success = asyncio.run(test_ollama_integration())
    sys.exit(0 if success else 1)
