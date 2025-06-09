#!/usr/bin/env python3
"""Test completion generation after fixing stop tokens."""

import asyncio
import os
import sys

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from app.config import config
from app.llm_optimized import LLMOptimized
from app.main_patch import logger
from app.schema import Message


async def test_completion_fix():
    """Test that completion generation works properly after stop token fix."""

    print("=== Testing Completion Generation Fix ===")

    try:
        # Initialize LLM
        llm = LLMOptimized(config.llm)

        # Test 1: Basic completion without tools
        print("\n1. Testing basic completion generation...")

        messages = [
            Message(role="user", content="Hello! Please introduce yourself briefly.")
        ]

        response = await llm.ask(messages, timeout=30)
        print(f"Response length: {len(response)} characters")
        print(f"Response preview: {response[:200]}...")

        if not response or len(response.strip()) < 10:
            print("❌ FAILED: Empty or very short response")
            return False
        else:
            print("✅ SUCCESS: Generated proper response")

        # Test 2: Tool calling with proper completion
        print("\n2. Testing tool calling completion...")

        # Define a simple tool
        tools = [
            {
                "type": "function",
                "function": {
                    "name": "test_tool",
                    "description": "A simple test tool",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "message": {
                                "type": "string",
                                "description": "A test message",
                            }
                        },
                        "required": ["message"],
                    },
                },
            }
        ]

        tool_messages = [
            Message(
                role="user",
                content="Please use the test_tool to send a greeting message.",
            )
        ]

        # Test ask_tool method
        if hasattr(llm, "ask_tool"):
            try:
                tool_response = await llm.ask_tool(
                    messages=tool_messages, tools=tools, timeout=30
                )

                completion_text = tool_response.get("content", "")
                tool_calls = tool_response.get("tool_calls", [])

                print(
                    f"Tool response completion length: {len(completion_text)} characters"
                )
                print(f"Tool response preview: {completion_text[:200]}...")
                print(f"Tool calls found: {len(tool_calls)}")

                if not completion_text or len(completion_text.strip()) < 5:
                    print("❌ FAILED: Empty or very short tool completion")
                    return False
                else:
                    print("✅ SUCCESS: Generated proper tool completion")

                    # Check if tool calls were parsed
                    if tool_calls:
                        print(f"✅ SUCCESS: Found {len(tool_calls)} tool calls")
                        for i, tc in enumerate(tool_calls):
                            print(
                                f"  Tool call {i+1}: {tc.get('function', {}).get('name', 'unknown')}"
                            )
                    else:
                        print(
                            "⚠️ INFO: No tool calls parsed (may be normal depending on response)"
                        )

            except Exception as e:
                print(f"❌ FAILED: ask_tool method error: {e}")
                return False
        else:
            print("❌ FAILED: ask_tool method not found")
            return False

        print("\n=== All tests passed! Completion generation is working properly ===")
        return True

    except Exception as e:
        print(f"❌ FAILED: Test error: {e}")
        import traceback

        traceback.print_exc()
        return False


if __name__ == "__main__":
    success = asyncio.run(test_completion_fix())
    sys.exit(0 if success else 1)
