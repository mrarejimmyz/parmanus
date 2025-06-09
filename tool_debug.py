"""Debug script to analyze tool call data and find missing name error."""

import json
import logging
import re
import time
from typing import Any, Dict, List

# Configure logging
logging.basicConfig(
    level=logging.DEBUG,
    format="%(asctime)s - %(levelname)s - %(message)s",
    filename="tool_debug.log",
)
logger = logging.getLogger("tool_debugger")

# Sample data to test with
sample_texts = [
    '{"name": "browser_open", "arguments": {"url": "https://example.com"}}',
    '{"function": {"name": "browser_search", "arguments": "test query"}}',
    'I need to call `browser_navigate` with {url="https://example.com"}',
    '```json\n{"name": "browser_click", "arguments": {"selector": ".button"}}\n```',
    '<function_calls><function name="browser_screenshot"></function></function_calls>',
]


def _format_tool_call(data: Dict[str, Any]) -> Dict[str, Any]:
    """Format a tool call into the expected structure."""
    try:
        logger.debug(f"Input tool call data: {data}")

        if not isinstance(data, dict):
            logger.error(f"Tool call data must be a dictionary, got {type(data)}")
            raise ValueError(f"Tool call data must be a dictionary, got {type(data)}")

        if not data:
            logger.error("Tool call data cannot be empty")
            raise ValueError("Tool call data cannot be empty")

        # Extract tool name from either direct name field or function name
        name = None
        if "function" in data and isinstance(data["function"], dict):
            name = data["function"].get("name")
            logger.debug(f"Found name in function: {name}")
        elif "name" in data:
            name = data["name"]
            logger.debug(f"Found direct name: {name}")
        else:
            logger.error(f"No name field found in data: {data}")
            raise ValueError(f"No name field found in data: {data}")

        if not isinstance(name, str) or not name.strip():
            logger.error(f"Tool call missing valid name, got {name}")
            raise ValueError(f"Tool call missing valid name, got {name}")

        # Get arguments from either direct arguments field or function arguments
        args = {}
        if "function" in data and isinstance(data["function"], dict):
            args = data["function"].get("arguments", {})
            logger.debug(f"Found args in function: {args}")
        elif "arguments" in data:
            args = data["arguments"]
            logger.debug(f"Found direct args: {args}")
        else:
            logger.debug("No arguments found, using empty dict")

        # Normalize args to a string
        if isinstance(args, (dict, list)):
            try:
                args_str = json.dumps(args)
                logger.debug(f"Converted args to string: {args_str}")
            except Exception as e:
                logger.warning(
                    f"Failed to JSON encode arguments: {e}, using str() instead"
                )
                args_str = str(args)
        elif not isinstance(args, str):
            args_str = str(args)
            logger.debug(f"Converted non-string args to string: {args_str}")
        else:
            args_str = args
            logger.debug(f"Args already a string: {args_str}")

        result = {
            "id": data.get("id", f"call_{time.time_ns()}"),
            "type": "function",
            "function": {"name": name.strip(), "arguments": args_str},
        }
        logger.debug(f"Formatted result: {result}")
        return result
    except Exception as e:
        logger.error(f"Failed to format tool call: {e}", exc_info=True)
        raise ValueError(f"Tool call formatting failed: {str(e)}")


def _parse_tool_calls(text: str) -> List[Dict[str, Any]]:
    """Parse tool calls from text with improved error handling."""
    tool_calls = []
    logger.debug(f"Parsing tool calls from: {text[:100]}...")

    try:
        # First try to parse as pure JSON
        try:
            data = json.loads(text)
            logger.debug(f"Successfully parsed as JSON: {data}")
            if isinstance(data, dict):
                if "name" in data:
                    # Convert to proper structure with function wrapper
                    try:
                        logger.debug("Found name field in dict, attempting to format")
                        formatted = _format_tool_call(data)
                        tool_calls.append(formatted)
                        return tool_calls
                    except Exception as e:
                        logger.error(f"Error formatting tool call: {e}")
                elif (
                    "function" in data
                    and isinstance(data["function"], dict)
                    and "name" in data["function"]
                ):
                    # Already in correct format
                    try:
                        logger.debug(
                            "Found function field with name in dict, attempting to format"
                        )
                        formatted = _format_tool_call(data)
                        tool_calls.append(formatted)
                        return tool_calls
                    except Exception as e:
                        logger.error(f"Error formatting tool call: {e}")
                else:
                    logger.warning(f"Dict has no name or function.name: {data}")
            elif isinstance(data, list):
                for item in data:
                    if isinstance(item, dict):
                        if "name" in item or (
                            "function" in item
                            and isinstance(item["function"], dict)
                            and "name" in item["function"]
                        ):
                            try:
                                logger.debug(f"Processing list item: {item}")
                                formatted = _format_tool_call(item)
                                tool_calls.append(formatted)
                            except Exception as e:
                                logger.error(
                                    f"Error formatting tool call from list: {e}"
                                )
                if tool_calls:
                    return tool_calls
        except json.JSONDecodeError:
            logger.debug("Failed to parse as pure JSON, trying other patterns")
            pass

        # Other parsing logic...
        # (Redacted for brevity)

        return tool_calls

    except Exception as e:
        logger.error(f"Error parsing tool calls: {str(e)}", exc_info=True)
        return []


# Main function to test the tool call parsing
def main():
    logger.info("Starting tool call debug")

    # Also configure console logging for immediate feedback
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.DEBUG)
    logger.addHandler(console_handler)

    print("===== TESTING TOOL CALL PARSING =====")

    # Try to import direct from app modules
    try:
        import sys

        sys.path.insert(0, ".")
        from app.llm_tool_patch import _format_tool_call as app_format
        from app.llm_tool_patch import _parse_tool_calls as app_parse

        print("Successfully imported functions directly from app module")

        # Compare implementations
        print("\nCOMPARING IMPLEMENTATIONS:")
        for func_name, local_func, app_func in [
            ("_parse_tool_calls", _parse_tool_calls, app_parse),
            ("_format_tool_call", _format_tool_call, app_format),
        ]:
            print(f"\nFunction: {func_name}")
            print(f"Local implementation: {local_func}")
            print(f"App implementation: {app_func}")

            # Check if they're the same
            if local_func.__code__.co_code == app_func.__code__.co_code:
                print("IMPLEMENTATIONS MATCH")
            else:
                print("IMPLEMENTATIONS DIFFER")
    except Exception as e:
        print(f"Import from app failed: {e}")

    print("\n===== TESTING WITH SAMPLE INPUTS =====")
    for i, text in enumerate(sample_texts):
        print(f"\nSample {i+1}:")
        print(f"Input: {text[:100]}...")
        try:
            tool_calls = _parse_tool_calls(text)
            print(f"Result: {json.dumps(tool_calls, indent=2)}")

            # If we got results, test formatting each one
            if tool_calls:
                print("\nFormatting results:")
                for j, call in enumerate(tool_calls):
                    try:
                        formatted = _format_tool_call(call)
                        print(
                            f"Formatted call {j+1}: {json.dumps(formatted, indent=2)}"
                        )
                    except Exception as e:
                        print(f"Formatting error: {e}")
        except Exception as e:
            print(f"Parsing error: {e}")
        logger.info("---")

    print("\n===== TESTING WITH PROBLEMATIC INPUT =====")
    # Test with some malformed inputs that might be causing the issue
    problematic_inputs = [
        "I will help you build a webpage",
        '{"arguments": {"query": "test"}}',  # Missing name
        '{"function": {"arguments": {"query": "test"}}}',  # Missing name in function
    ]

    for i, text in enumerate(problematic_inputs):
        print(f"\nProblematic input {i+1}:")
        print(f"Input: {text}")
        try:
            tool_calls = _parse_tool_calls(text)
            print(f"Result: {json.dumps(tool_calls, indent=2)}")
        except Exception as e:
            print(f"Parsing error: {e}")

    logger.info("Testing complete")


if __name__ == "__main__":
    main()
