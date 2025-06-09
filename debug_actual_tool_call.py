#!/usr/bin/env python3
"""Debug the actual tool call that's failing."""

import asyncio
import json
import sys
from pathlib import Path

# Add the main directory to Python path
root_dir = Path(__file__).parent.absolute()
sys.path.insert(0, str(root_dir))

try:
    from app.config import config
    from app.llm_optimized import LLMOptimized
    from app.tool import (
        Bash,
        BrowserUseTool,
        StrReplaceEditor,
        Terminate,
        ToolCollection,
    )

    async def debug_ask_tool():
        print("Setting up LLM and tools...")

        # Create an LLM instance with proper config
        llm = LLMOptimized(config.llm)

        # Create a tool collection like the application does
        tool_collection = ToolCollection(
            BrowserUseTool(), Bash(), StrReplaceEditor(), Terminate()
        )
        tools = tool_collection.to_params()

        print(f"Available tools: {[tool['function']['name'] for tool in tools]}")

        # Test a simple prompt that should trigger tool use
        test_prompt = "Create a web page with basic HTML structure"

        try:
            # This is where the error happens
            print(f"Testing ask_tool with: {test_prompt}")
            result = await llm.ask_tool(
                messages=[{"role": "user", "content": test_prompt}],
                tools=tools,
                max_retries=1,
            )
            print(f"Success: {result}")
        except Exception as e:
            print(f"Error caught: {e}")
            print(f"Error type: {type(e)}")
            import traceback

            traceback.print_exc()

            # Let's also test the basic ask method
            print("\nTesting basic ask method:")
            try:
                basic_result = await llm.ask(test_prompt)
                print(f"Basic ask result: {basic_result[:200]}...")
            except Exception as basic_e:
                print(f"Basic ask error: {basic_e}")
                import traceback

                traceback.print_exc()

    # Run the debug
    asyncio.run(debug_ask_tool())

except ImportError as e:
    print(f"Import error: {e}")
    import traceback

    traceback.print_exc()
