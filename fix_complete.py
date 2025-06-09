"""Complete fix for the tool call issue in ParManusAI."""

import os
import shutil
import time
from pathlib import Path

# Get root directory
root_dir = os.path.dirname(os.path.abspath(__file__))

# Files to patch
files_to_patch = [
    os.path.join(root_dir, "app/llm_tool_patch.py"),
    os.path.join(root_dir, "app/llm_tool_patch_optimized.py"),
]

# Create backups
for file_path in files_to_patch:
    timestamp = int(time.time())
    backup_path = f"{file_path}.backup_{timestamp}"
    shutil.copy2(file_path, backup_path)
    print(f"Created backup: {backup_path}")

# Complete fixed version of the file with proper debugging
fixed_content = '''"""Fixes for LLM tool patching issues."""

import asyncio
import json
import logging
import re
import time
import types
from typing import Any, Dict, List, Optional, Union

from app.exceptions import ModelTimeoutError, TokenLimitExceeded
from app.logger import logger
from app.schema import Message, ToolChoice


def calculate_adaptive_timeout(
    messages: List[Any], tools: Optional[List[Dict[str, Any]]] = None
) -> int:
    """Calculate adaptive timeout based on input complexity."""
    base_timeout = 30
    message_factor = len(messages) * 2
    tool_factor = len(tools or []) * 1.5
    return min(int(base_timeout + message_factor + tool_factor), 120)


def format_tool_definitions(tools: Optional[List[Dict[str, Any]]]) -> str:
    """Format tool definitions for the prompt."""
    if not tools:
        return ""

    tool_str = "\\nAvailable tools:\\n"
    for tool in tools:
        tool_str += f"\\n- {tool['name']}: {tool['description']}"
        if "parameters" in tool:
            tool_str += "\\n  Parameters:"
            for param_name, param_info in (
                tool["parameters"].get("properties", {}).items()
            ):
                required = param_name in tool["parameters"].get("required", [])
                tool_str += f"\\n    - {param_name} ({'required' if required else 'optional'}): {param_info.get('description', '')}"
    return tool_str


def get_tool_instructions(tool_choice: Union[str, ToolChoice]) -> str:
    """Get tool instructions based on tool choice."""
    if tool_choice == ToolChoice.AUTO:
        return (
            "\\nYou can use tools if needed, but don't force their use if not necessary."
        )
    elif tool_choice == ToolChoice.REQUIRED:
        return "\\nYou must use one of the provided tools to complete this task."
    elif tool_choice == ToolChoice.NONE:
        return "\\nDo not use any tools for this task."
    return ""


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
                logger.warning(f"Failed to JSON encode arguments: {e}, using str() instead")
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
    """Parse tool calls from text with improved error handling and debugging."""
    tool_calls = []
    logger.debug(f"Parsing tool calls from: {text[:100]}...")

    try:
        # First try to parse as pure JSON
        try:
            data = json.loads(text)
            logger.debug(f"Successfully parsed as JSON: {type(data)}")
            if isinstance(data, dict):
                if "name" in data:
                    logger.debug("Found name field in dict, formatting as tool call")
                    tool_calls.append(_format_tool_call(data))
                    return tool_calls
                elif "function" in data and isinstance(data["function"], dict) and "name" in data["function"]:
                    logger.debug("Found function field with name in dict, formatting as tool call")
                    tool_calls.append(_format_tool_call(data))
                    return tool_calls
                else:
                    logger.warning(f"Dict has neither name nor function.name: {data}")
            elif isinstance(data, list):
                for item in data:
                    if isinstance(item, dict):
                        if "name" in item:
                            logger.debug(f"Processing list item with name: {item['name']}")
                            tool_calls.append(_format_tool_call(item))
                        elif "function" in item and isinstance(item["function"], dict) and "name" in item["function"]:
                            logger.debug(f"Processing list item with function.name: {item['function']['name']}")
                            tool_calls.append(_format_tool_call(item))
                        else:
                            logger.warning(f"List item has neither name nor function.name: {item}")
                if tool_calls:
                    return tool_calls
        except json.JSONDecodeError as e:
            logger.debug(f"Failed to parse as pure JSON, trying other patterns. Error: {e}")

        # Pattern 1: Natural language tool calls
        pattern = r"(?:Called|Using|Execute|Run|Invoke|Call)\\s+`([^`]+)`\\s+with(?:\\s+arguments?)?:?\\s*{([^}]+)}"
        matches = re.finditer(pattern, text, re.IGNORECASE)

        for match in matches:
            try:
                name = match.group(1).strip()
                args_str = match.group(2).strip()
                logger.debug(f"Found natural language tool call: name={name}, args={args_str}")

                try:
                    # Try to parse as JSON
                    args = json.loads("{" + args_str + "}")
                except json.JSONDecodeError:
                    # Fallback to simple key=value parsing
                    args = {}
                    for pair in args_str.split(","):
                        if "=" in pair:
                            k, v = pair.split("=", 1)
                            args[k.strip()] = v.strip().strip("\\"'")

                tool_data = {
                    "name": name,
                    "arguments": args
                }
                logger.debug(f"Parsed natural language into: {tool_data}")
                tool_calls.append(_format_tool_call(tool_data))
            except Exception as e:
                logger.error(f"Error processing natural language tool call: {e}")
                continue

        # Pattern 2: JSON blocks
        pattern = r"```(?:json)?\\s*([\\s\\S]*?)\\s*```"
        matches = re.finditer(pattern, text)

        for match in matches:
            try:
                json_block = match.group(1)
                logger.debug(f"Found JSON block: {json_block[:50]}...")
                data = json.loads(json_block)

                if isinstance(data, dict):
                    if "name" in data:
                        logger.debug(f"Found name in JSON block: {data['name']}")
                        tool_calls.append(_format_tool_call(data))
                    elif "function" in data and isinstance(data["function"], dict) and "name" in data["function"]:
                        logger.debug(f"Found function.name in JSON block: {data['function']['name']}")
                        tool_calls.append(_format_tool_call(data))
                    else:
                        logger.warning(f"JSON block dict has neither name nor function.name: {data}")
                elif isinstance(data, list):
                    for item in data:
                        if isinstance(item, dict):
                            if "name" in item or ("function" in item and isinstance(item["function"], dict) and "name" in item["function"]):
                                logger.debug(f"Processing JSON block list item")
                                tool_calls.append(_format_tool_call(item))
                            else:
                                logger.warning(f"JSON block list item has neither name nor function.name: {item}")
            except json.JSONDecodeError as e:
                logger.warning(f"Failed to parse JSON block: {e}")
                continue
            except Exception as e:
                logger.error(f"Error processing JSON block: {e}")
                continue

        # Pattern 3: XML-style function calls
        xml_pattern = r'<(?:antml:)?function_calls>\\s*<(?:antml:)?(?:invoke|function)\\s+name="([^"]+)">(.*?)</(?:antml:)?(?:invoke|function)>\\s*</(?:antml:)?function_calls>'
        xml_matches = re.finditer(xml_pattern, text, re.DOTALL)

        for xml_match in xml_matches:
            try:
                tool_name = xml_match.group(1).strip()
                params_text = xml_match.group(2).strip()
                logger.debug(f"Found XML-style call: name={tool_name}, params={params_text[:50]}...")

                tool_args = {}

                # Parse individual parameters
                param_pattern = r'<(?:antml:)?parameter\\s+name="([^"]+)">(.*?)</(?:antml:)?parameter>'
                param_matches = re.finditer(param_pattern, params_text, re.DOTALL)

                for param_match in param_matches:
                    param_name = param_match.group(1)
                    param_value = param_match.group(2).strip()
                    logger.debug(f"Found param: {param_name}={param_value[:30]}...")

                    # Try to parse as JSON if it looks like a JSON value
                    if param_value.startswith("{") or param_value.startswith("["):
                        try:
                            tool_args[param_name] = json.loads(param_value)
                        except json.JSONDecodeError:
                            tool_args[param_name] = param_value
                    else:
                        tool_args[param_name] = param_value

                tool_data = {
                    "name": tool_name,
                    "arguments": tool_args
                }
                logger.debug(f"Parsed XML-style into: {tool_data}")
                tool_calls.append(_format_tool_call(tool_data))
            except Exception as e:
                logger.error(f"Error processing XML-style tool call: {e}")
                continue

        logger.debug(f"Finished parsing, found {len(tool_calls)} tool calls")
        return tool_calls
    except Exception as e:
        logger.error(f"Error in _parse_tool_calls: {str(e)}", exc_info=True)
        return []


async def ask_tool(
    self,
    messages: List[Union[Message, Dict[str, Any]]],
    system_msgs: Optional[List[Union[Message, Dict[str, Any]]]] = None,
    tools: Optional[List[Dict[str, Any]]] = None,
    tool_choice: Union[str, ToolChoice] = ToolChoice.AUTO,
    temp: float = 0.0,
    timeout: Optional[int] = None,
    max_retries: int = 2,
    **kwargs,
) -> Dict[str, Any]:
    """Enhanced ask_tool method with improved error handling and timeouts."""

    if timeout is None:
        timeout = calculate_adaptive_timeout(messages, tools)

    start_time = time.time()
    last_exception = None
    result = None

    for attempt in range(max_retries + 1):
        try:
            if attempt > 0:
                logger.warning(
                    f"Retrying tool call (attempt {attempt + 1}/{max_retries + 1})"
                )

            # Format messages
            formatted_messages = []
            if system_msgs:
                formatted_messages.extend(
                    [
                        msg if isinstance(msg, dict) else msg.to_dict()
                        for msg in system_msgs
                    ]
                )
            formatted_messages.extend(
                [msg if isinstance(msg, dict) else msg.to_dict() for msg in messages]
            )

            # Prepare the prompt
            prompt = self._format_prompt_for_llama(formatted_messages)
            tool_definitions = format_tool_definitions(tools) if tools else ""
            tool_instructions = get_tool_instructions(tool_choice)
            enhanced_prompt = f"{prompt}\\n\\n{tool_definitions}{tool_instructions}"

            # Run model with timeout
            completion = await asyncio.wait_for(
                asyncio.get_event_loop().run_in_executor(
                    self._executor,
                    lambda: self.text_model.create_completion(
                        prompt=enhanced_prompt,
                        max_tokens=min(self.max_tokens, self.MAX_ALLOWED_OUTPUT_TOKENS),
                        temperature=temp,
                        stop=["<|user|>", "<|system|>"],
                        **kwargs,
                    ),
                ),
                timeout=timeout,
            )

            # Extract and validate completion text
            completion_text = completion.get("choices", [{}])[0].get("text", "").strip()
            if not completion_text:
                raise ValueError("Empty completion text received")

            # Parse tool calls with detailed error handling
            try:
                logger.debug(f"About to parse tool calls from: {completion_text[:100]}...")
                tool_calls = self._parse_tool_calls(completion_text)
                logger.debug(f"Parsed {len(tool_calls)} tool calls")

                # Verify tool calls are valid
                for i, tc in enumerate(tool_calls):
                    try:
                        if not tc.get("function", {}).get("name"):
                            logger.error(f"Tool call {i} missing name: {tc}")
                            raise ValueError(f"Tool call {i} missing name")
                    except Exception as e:
                        logger.error(f"Error validating tool call {i}: {e}")
                        raise
            except Exception as e:
                logger.error(f"Error parsing tool calls: {str(e)}", exc_info=True)
                # Return empty tool calls instead of failing
                tool_calls = []

            # Build successful response
            result = {
                "content": completion_text,
                "tool_calls": tool_calls,
                "usage": {
                    "prompt_tokens": self.count_tokens(enhanced_prompt),
                    "completion_tokens": self.count_tokens(completion_text),
                    "total_tokens": self.count_tokens(enhanced_prompt)
                    + self.count_tokens(completion_text),
                },
                "elapsed_time": time.time() - start_time,
                "attempts": attempt + 1,
            }
            return result

        except asyncio.TimeoutError as e:
            last_exception = e
            if attempt < max_retries:
                timeout = min(timeout * 1.5, 180)  # Cap at 3 minutes
                continue
            logger.error(f"Tool call timed out after {max_retries + 1} attempts")
            break

        except Exception as e:
            last_exception = e
            if attempt < max_retries and not isinstance(e, TokenLimitExceeded):
                continue
            logger.error(f"Tool call failed: {str(e)}")
            break

    # If we got a partial result, return it with error indication
    if result:
        result["error"] = str(last_exception)
        return result

    # Otherwise raise the exception
    raise last_exception or RuntimeError("Tool call failed with no error details")


def patch_llm_class():
    """Apply LLM patches with improved error handling."""
    from app.llm import LLM
    from app.llm_optimized import LLMOptimized

    try:
        # Create method wrappers that maintain proper self reference
        def ask_tool_wrapper(self, *args, **kwargs):
            return ask_tool(self, *args, **kwargs)

        def parse_tool_calls_wrapper(self, text):
            return _parse_tool_calls(text)

        def format_tool_call_wrapper(self, data):
            return _format_tool_call(data)

        # Patch both LLM and LLMOptimized classes
        for cls in [LLM, LLMOptimized]:
            cls.ask_tool = ask_tool_wrapper
            cls._parse_tool_calls = parse_tool_calls_wrapper
            cls._format_tool_call = format_tool_call_wrapper
            logger.info(f"Successfully patched {cls.__name__} with tool methods")
    except Exception as e:
        logger.error(f"Failed to patch LLM classes: {e}")
        raise


# Apply patches when module is imported
patch_llm_class()
'''

# Apply the fix to both files
for file_path in files_to_patch:
    with open(file_path, "w") as f:
        f.write(fixed_content)
    print(f"Applied comprehensive fix to: {file_path}")

print(
    "\nFixes successfully applied. Now run the application to test if the issue is resolved."
)
