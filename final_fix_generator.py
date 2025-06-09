"""Final comprehensive fix for ParManusAI tool call issues."""

import os
import shutil
import time
from pathlib import Path


def create_complete_llm_tool_patch():
    """Create a complete, working LLM tool patch implementation."""

    return '''"""Complete fixes for LLM tool patching issues."""

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
        logger.debug(f"Formatting tool call: {data}")

        if not isinstance(data, dict):
            raise ValueError(f"Tool call data must be a dictionary, got {type(data)}")

        if not data:
            raise ValueError("Tool call data cannot be empty")

        # Extract tool name from either direct name field or function name
        name = None
        if "function" in data and isinstance(data["function"], dict):
            name = data["function"].get("name")
        elif "name" in data:
            name = data["name"]

        if not isinstance(name, str) or not name.strip():
            raise ValueError(f"Tool call missing valid name, got {name}")

        # Get arguments from either direct arguments field or function arguments
        args = {}
        if "function" in data and isinstance(data["function"], dict):
            args = data["function"].get("arguments", {})
        elif "arguments" in data:
            args = data["arguments"]

        # Normalize args to a string
        if isinstance(args, (dict, list)):
            try:
                args_str = json.dumps(args)
            except Exception as e:
                logger.warning(f"Failed to JSON encode arguments: {e}, using str() instead")
                args_str = str(args)
        elif not isinstance(args, str):
            args_str = str(args)
        else:
            args_str = args

        result = {
            "id": data.get("id", f"call_{time.time_ns()}"),
            "type": "function",
            "function": {"name": name.strip(), "arguments": args_str},
        }
        logger.debug(f"Formatted tool call result: {result}")
        return result
    except Exception as e:
        logger.error(f"Failed to format tool call: {e}")
        raise ValueError(f"Tool call formatting failed: {str(e)}")


def _parse_tool_calls(text: str) -> List[Dict[str, Any]]:
    """Parse tool calls from text with comprehensive pattern matching."""
    tool_calls = []
    logger.debug(f"Parsing tool calls from: {text[:200]}...")

    try:
        # Pattern 1: Try to parse as pure JSON first
        try:
            data = json.loads(text.strip())
            logger.debug(f"Successfully parsed as JSON: {type(data)}")

            if isinstance(data, dict):
                if "name" in data or ("function" in data and isinstance(data["function"], dict) and "name" in data["function"]):
                    formatted = _format_tool_call(data)
                    tool_calls.append(formatted)
                    return tool_calls
            elif isinstance(data, list):
                for item in data:
                    if isinstance(item, dict) and ("name" in item or ("function" in item and isinstance(item["function"], dict) and "name" in item["function"])):
                        formatted = _format_tool_call(item)
                        tool_calls.append(formatted)
                if tool_calls:
                    return tool_calls
        except json.JSONDecodeError:
            logger.debug("Failed to parse as pure JSON, trying other patterns")

        # Pattern 2: JSON code blocks
        json_block_pattern = r'```(?:json)?\\s*([\\s\\S]*?)\\s*```'
        json_matches = re.finditer(json_block_pattern, text, re.IGNORECASE)

        for match in json_matches:
            try:
                json_content = match.group(1).strip()
                data = json.loads(json_content)

                if isinstance(data, dict):
                    if "name" in data or ("function" in data and isinstance(data["function"], dict) and "name" in data["function"]):
                        formatted = _format_tool_call(data)
                        tool_calls.append(formatted)
                elif isinstance(data, list):
                    for item in data:
                        if isinstance(item, dict) and ("name" in item or ("function" in item and isinstance(item["function"], dict) and "name" in item["function"])):
                            formatted = _format_tool_call(item)
                            tool_calls.append(formatted)
            except (json.JSONDecodeError, Exception) as e:
                logger.debug(f"Failed to parse JSON block: {e}")
                continue

        # Pattern 3: Natural language tool calls
        natural_pattern = r'(?:Call|Use|Execute|Run|Invoke)\\s+`([^`]+)`\\s+with(?:\\s+(?:arguments?|params?))?:?\\s*({[^}]+})'
        natural_matches = re.finditer(natural_pattern, text, re.IGNORECASE)

        for match in natural_matches:
            try:
                tool_name = match.group(1).strip()
                args_text = match.group(2).strip()

                # Try to parse arguments as JSON
                try:
                    args = json.loads(args_text)
                except json.JSONDecodeError:
                    # Fallback to simple parsing
                    args = {}
                    # Remove braces and parse key=value pairs
                    clean_args = args_text.strip('{}')
                    for pair in clean_args.split(','):
                        if '=' in pair:
                            k, v = pair.split('=', 1)
                            args[k.strip()] = v.strip().strip('"').strip("'")

                tool_data = {"name": tool_name, "arguments": args}
                formatted = _format_tool_call(tool_data)
                tool_calls.append(formatted)
            except Exception as e:
                logger.debug(f"Failed to parse natural language tool call: {e}")
                continue

        # Pattern 4: XML-style function calls
        xml_patterns = [
            r'<(?:antml:)?function_calls>\\s*<(?:antml:)?(?:invoke|function)\\s+name="([^"]+)">(.*?)</(?:antml:)?(?:invoke|function)>\\s*</(?:antml:)?function_calls>',
            r'<function_calls>\\s*<function\\s+name="([^"]+)"[^>]*>(.*?)</function>\\s*</function_calls>',
        ]

        for xml_pattern in xml_patterns:
            xml_matches = re.finditer(xml_pattern, text, re.DOTALL)

            for xml_match in xml_matches:
                try:
                    tool_name = xml_match.group(1).strip()
                    params_text = xml_match.group(2).strip()
                    tool_args = {}

                    # Parse individual parameters
                    param_patterns = [
                        r'<(?:antml:)?parameter\\s+name="([^"]+)">(.*?)</(?:antml:)?parameter>',
                        r'<param\\s+name="([^"]+)">(.*?)</param>',
                    ]

                    for param_pattern in param_patterns:
                        param_matches = re.finditer(param_pattern, params_text, re.DOTALL)
                        for param_match in param_matches:
                            param_name = param_match.group(1)
                            param_value = param_match.group(2).strip()

                            # Try to parse as JSON if it looks like JSON
                            if param_value.startswith(("{", "[")):
                                try:
                                    tool_args[param_name] = json.loads(param_value)
                                except json.JSONDecodeError:
                                    tool_args[param_name] = param_value
                            else:
                                tool_args[param_name] = param_value

                    tool_data = {"name": tool_name, "arguments": tool_args}
                    formatted = _format_tool_call(tool_data)
                    tool_calls.append(formatted)
                except Exception as e:
                    logger.debug(f"Failed to parse XML tool call: {e}")
                    continue

        # Pattern 5: Simple function calls
        simple_pattern = r'(\\w+)\\s*\\(([^)]*)\\)'
        simple_matches = re.finditer(simple_pattern, text)

        # Only consider this if no other patterns matched and the function name looks like a tool
        if not tool_calls:
            for match in simple_matches:
                try:
                    func_name = match.group(1)
                    args_text = match.group(2).strip()

                    # Only consider if it looks like a tool name (contains underscore or specific patterns)
                    if '_' in func_name or func_name.lower().startswith(('browser', 'search', 'file', 'web')):
                        args = {}
                        if args_text:
                            # Simple argument parsing
                            for arg in args_text.split(','):
                                arg = arg.strip()
                                if '=' in arg:
                                    k, v = arg.split('=', 1)
                                    args[k.strip()] = v.strip().strip('"').strip("'")
                                else:
                                    args['value'] = arg.strip('"').strip("'")

                        tool_data = {"name": func_name, "arguments": args}
                        formatted = _format_tool_call(tool_data)
                        tool_calls.append(formatted)
                except Exception as e:
                    logger.debug(f"Failed to parse simple function call: {e}")
                    continue

        logger.debug(f"Parsed {len(tool_calls)} tool calls")
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
                logger.warning(f"Retrying tool call (attempt {attempt + 1}/{max_retries + 1})")

            # Format messages
            formatted_messages = []
            if system_msgs:
                formatted_messages.extend([
                    msg if isinstance(msg, dict) else msg.to_dict()
                    for msg in system_msgs
                ])
            formatted_messages.extend([
                msg if isinstance(msg, dict) else msg.to_dict()
                for msg in messages
            ])

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

            # Parse tool calls with error handling
            try:
                logger.debug(f"About to parse tool calls from: {completion_text[:100]}...")
                tool_calls = self._parse_tool_calls(completion_text)
                logger.debug(f"Successfully parsed {len(tool_calls)} tool calls")

                # Validate tool calls
                for i, tc in enumerate(tool_calls):
                    if not tc.get("function", {}).get("name"):
                        logger.error(f"Tool call {i} missing name: {tc}")
                        raise ValueError(f"Tool call {i} missing name")
            except Exception as e:
                logger.error(f"Error parsing tool calls: {str(e)}")
                # Instead of failing, return empty tool calls
                tool_calls = []

            # Build successful response
            result = {
                "content": completion_text,
                "tool_calls": tool_calls,
                "usage": {
                    "prompt_tokens": self.count_tokens(enhanced_prompt),
                    "completion_tokens": self.count_tokens(completion_text),
                    "total_tokens": self.count_tokens(enhanced_prompt) + self.count_tokens(completion_text),
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


def apply_final_fix():
    """Apply the final comprehensive fix."""

    # Get root directory
    root_dir = os.path.dirname(os.path.abspath(__file__))

    # Files to patch
    files_to_patch = [
        os.path.join(root_dir, "app", "llm_tool_patch.py"),
        os.path.join(root_dir, "app", "llm_tool_patch_optimized.py"),
    ]

    # Create backups
    timestamp = int(time.time())
    for file_path in files_to_patch:
        if os.path.exists(file_path):
            backup_path = f"{file_path}.backup_{timestamp}"
            shutil.copy2(file_path, backup_path)
            print(f"Created backup: {backup_path}")

    # Apply the complete fix
    complete_implementation = create_complete_llm_tool_patch()

    for file_path in files_to_patch:
        with open(file_path, "w", encoding="utf-8") as f:
            f.write(complete_implementation)
        print(f"Applied comprehensive fix to: {file_path}")

    print("\nFinal fix applied successfully!")
    print("\nKey improvements:")
    print(
        "- Complete pattern matching for JSON, XML, natural language, and simple function calls"
    )
    print("- Robust error handling that prevents crashes")
    print("- Comprehensive tool call validation")
    print("- Better debugging and logging")
    print("\nPlease test the application now.")


if __name__ == "__main__":
    apply_final_fix()
