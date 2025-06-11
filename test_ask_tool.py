#!/usr/bin/env python3
"""
Simple test to isolate the 'name' error in tool calling
"""
import asyncio
import os
import sys

# Add the project root to the Python path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from app.config import config
from app.gpu_manager import get_gpu_manager
from app.llm_factory import create_llm_async
from app.logger import logger
from app.schema import Message


async def test_ask_tool_simple():
    """Test the ask_tool method with a simple tool definition."""
    try:
        # Initialize LLM using the factory (which will select Ollama)
        llm = await create_llm_async(config.llm)

        # Simple tool definition
        tools = [
            {
                "name": "terminate",
                "description": "Use this to end the conversation",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "reason": {
                            "type": "string",
                            "description": "Reason for termination",
                        }
                    },
                    "required": ["reason"],
                },
            }
        ]

        # Simple message
        messages = [Message.user_message("Hello, please terminate this conversation.")]

        print("🔧 Testing ask_tool method...")

        try:
            response = await llm.ask_tool(
                messages=messages, tools=tools, temp=0.1, timeout=30
            )

            print(f"✅ ask_tool succeeded!")
            print(f"Response: {response}")

        except Exception as e:
            print(f"❌ ask_tool failed: {e}")
            logger.error(f"ask_tool error details: {e}", exc_info=True)  # Cleanup
        if hasattr(llm, "cleanup_models"):
            llm.cleanup_models()
        elif hasattr(llm, "cleanup"):
            await llm.cleanup()

    except Exception as e:
        print(f"❌ Test setup failed: {e}")
        logger.error(f"Test setup error: {e}", exc_info=True)


if __name__ == "__main__":
    print("🚀 Testing ask_tool method...")
    asyncio.run(test_ask_tool_simple())
    print("🏁 Test completed!")
