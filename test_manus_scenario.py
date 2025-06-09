#!/usr/bin/env python3
"""Debug script to test the actual Manus agent scenario."""

import asyncio
import json
import os
import sys
from pathlib import Path

# Add the main directory to Python path
root_dir = Path(__file__).parent.absolute()
sys.path.insert(0, str(root_dir))

print("Testing Manus agent scenario...")
sys.stdout.flush()

try:
    from app.config import config
    from app.llm_optimized import LLMOptimized
    from app.schema import Message, ToolChoice
    from app.tool.ask_human import AskHuman
    from app.tool.browser_use_tool import BrowserUseTool
    from app.tool.python_execute import PythonExecute
    from app.tool.str_replace_editor import StrReplaceEditor
    from app.tool.terminate import Terminate

    print("Successfully imported required modules")
    sys.stdout.flush()

    async def test_manus_scenario():
        """Test the same scenario that Manus agent encounters."""

        # Initialize LLM like Manus does
        llm = LLMOptimized(config.llm)
        print(f"Initialized LLM: {llm.model}")
        sys.stdout.flush()

        # Get tools like Manus does
        tools = []
        try:
            tools.append(PythonExecute().model_dump())
            print("Added PythonExecute tool")
        except Exception as e:
            print(f"Failed to add PythonExecute: {e}")

        try:
            tools.append(BrowserUseTool().model_dump())
            print("Added BrowserUseTool")
        except Exception as e:
            print(f"Failed to add BrowserUseTool: {e}")

        print(f"Total tools available: {len(tools)}")
        for tool in tools:
            print(f"  - {tool.get('name', 'unknown')}")
        sys.stdout.flush()

        # Test message like from Manus
        messages = [Message.user_message("build me a webpage")]

        print("About to call ask_tool with Manus-like setup...")
        sys.stdout.flush()

        try:
            result = await llm.ask_tool(
                messages=messages,
                tools=tools,
                tool_choice=ToolChoice.AUTO,
                temp=0.0,
                timeout=30,
                max_retries=2,
            )

            print("SUCCESS!")
            print(f"Content: {result.get('content', '')[:200]}...")
            print(f"Tool calls: {len(result.get('tool_calls', []))}")
            for i, tc in enumerate(result.get("tool_calls", [])):
                print(
                    f"  Tool call {i}: {tc.get('function', {}).get('name', 'unknown')}"
                )

        except Exception as e:
            print(f"ERROR: {e}")
            print(f"Error type: {type(e).__name__}")

            # Check if this is our specific error
            error_str = str(e)
            if "'name'" in error_str:
                print("*** This is the 'name' error! ***")
            elif "missing name" in error_str:
                print("*** This is the 'missing name' error! ***")

            import traceback

            traceback.print_exc()

        finally:
            # Cleanup
            try:
                llm.cleanup_models()
            except:
                pass

    # Run the test
    asyncio.run(test_manus_scenario())

except Exception as e:
    print(f"Failed to run test: {e}")
    import traceback

    traceback.print_exc()

print("Test completed.")
sys.stdout.flush()
