#!/usr/bin/env python3
"""Simple debug to test LLM tool parsing."""

import json
import os
import sys
from pathlib import Path

# Add the main directory to Python path
root_dir = Path(__file__).parent.absolute()
sys.path.insert(0, str(root_dir))

print("Starting simple debug...")
sys.stdout.flush()

try:
    from app.llm_tool_patch import _parse_tool_calls

    print("Successfully imported _parse_tool_calls")

    # Test with some sample LLM outputs that might be causing issues
    test_outputs = [
        # Empty or minimal output
        "",
        "I need to help you.",
        "Let me use a tool to help.",
        # Malformed JSON
        "{ incomplete",
        '{"name": "test"}',
        '{"name": "test", "arguments": {}}',
        # Tool calls without proper structure
        "python_execute(code='print(hello)')",
        "Call python_execute with code='print(hello)'",
        # Well-formed examples
        '{"name": "python_execute", "arguments": {"code": "print(\\"hello\\")"}}',
        """<function_calls>
<function name="python_execute">
<parameter name="code">print("hello")</parameter>
</function>
</function_calls>""",
    ]

    for i, output in enumerate(test_outputs):
        print(f"\n=== Test {i+1}: {repr(output[:50])} ===")
        try:
            result = _parse_tool_calls(output)
            print(f"Success: {len(result)} tool calls parsed")
            if result:
                print(f"Tool calls: {json.dumps(result, indent=2)}")
        except Exception as e:
            print(f"Error: {e}")
            print(f"Error type: {type(e).__name__}")

            # Check if this is our specific 'name' error
            if "'name'" in str(e):
                print("*** This is the 'name' error we're looking for! ***")

    print("\nDebug completed.")

except Exception as e:
    print(f"Failed to import or run debug: {e}")
    import traceback

    traceback.print_exc()
