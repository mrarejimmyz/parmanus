"""Fixes for LLM tool patching issues."""

import asyncio
import inspect
import json
import logging
import re
import time
import types
from typing import Any, Callable, Dict, List, Optional, Union

from app.exceptions import ModelTimeoutError, TokenLimitExceeded
from app.llm import LLM
from app.llm_optimized import LLMOptimized
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

    tool_str = "\nAvailable tools:\n"
    for tool in tools:
        tool_str += f"\n- {tool['name']}: {tool['description']}"
        if "parameters" in tool:
            tool_str += "\n  Parameters:"
            for param_name, param_info in (
                tool["parameters"].get("properties", {}).items()
            ):
                required = param_name in tool["parameters"].get("required", [])
                tool_str += f"\n    - {param_name} ({'required' if required else 'optional'}): {param_info.get('description', '')}"
    return tool_str


def get_tool_instructions(tool_choice: Union[str, ToolChoice]) -> str:
    """Get tool instructions based on tool choice."""
    if tool_choice == ToolChoice.AUTO:
        return (
            "\nYou can use tools if needed, but don't force their use if not necessary."
        )
    elif tool_choice == ToolChoice.REQUIRED:
        return "\nYou must use one of the provided tools to complete this task."
    elif tool_choice == ToolChoice.NONE:
        return "\nDo not use any tools for this task."
    return ""


def _format_tool_call(data: Dict[str, Any]) -> Dict[str, Any]:
    """Format a tool call into the expected structure."""
    if not data:
        raise ValueError("Tool call data cannot be empty")

    # Extract tool name, with better error handling
    name = None
    if isinstance(data, dict):
        if "name" in data:
            name = data["name"]
        elif "function" in data and isinstance(data["function"], dict):
            name = data["function"].get("name")
        elif "function" in data and isinstance(data["function"], str):
            name = data["function"]

    if not name:
        raise ValueError("Tool call missing required name field")

    # Get arguments, with better type handling
    args = {}
    if "arguments" in data:
        args = data["arguments"]
    elif "function" in data and isinstance(data["function"], dict):
        args = data["function"].get("arguments", {})

    # Ensure args is properly formatted
    if isinstance(args, (dict, list)):
        args = json.dumps(args)
    elif not isinstance(args, str):
        args = str(args)

    # Create normalized tool call structure
    return {
        "id": data.get("id", f"call_{time.time_ns()}"),
        "type": "function",
        "function": {
            "name": str(name).strip(),  # Ensure name is string and cleaned
            "arguments": args,
        },
    }


# Helper functions for tool call handling
def _validate_tool_call(tool_call: Dict[str, Any]) -> bool:
    """Validate a tool call has all required fields and proper structure."""
    if not isinstance(tool_call, dict):
        logger.warning("Tool call must be a dictionary")
        return False

    try:
        # Check for both direct and wrapped function formats
        name = None
        if "function" in tool_call:
            if isinstance(tool_call["function"], dict):
                name = tool_call["function"].get("name")
            elif isinstance(tool_call["function"], str):
                name = tool_call["function"]
        elif "name" in tool_call:
            name = tool_call["name"]

        if not name:
            logger.warning("Tool call missing required name field")
            return False

        # For wrapped function calls, validate structure
        if "function" in tool_call:
            function = tool_call["function"]
            if isinstance(function, dict):
                # Validate arguments if present
                if "arguments" in function:
                    if not isinstance(function["arguments"], (str, dict)):
                        logger.warning("Tool call arguments must be string or dict")
                        return False

        return True

    except Exception as e:
        logger.warning(f"Tool call validation failed: {str(e)}")
        return False


def _normalize_tool_call(data: Dict[str, Any]) -> Dict[str, Any]:
    """Normalize different tool call formats into standard structure."""
    try:
        # If already in correct format and valid, return as is
        if _validate_tool_call(data):
            if "function" in data and isinstance(data["function"], dict):
                return data

        # Extract name with enhanced error handling
        name = None
        if isinstance(data, dict):
            if "name" in data:
                name = data["name"]
            elif "function" in data and isinstance(data["function"], dict):
                name = data["function"].get("name")
            elif "function" in data and isinstance(data["function"], str):
                name = data["function"]

        if not name:
            raise ValueError("Tool call missing required name field")

        # Handle arguments with proper type checking
        args = {}
        if isinstance(data, dict):
            if "arguments" in data:
                args = data["arguments"]
            elif "function" in data and isinstance(data["function"], dict):
                args = data["function"].get("arguments", {})

        # Ensure arguments are properly formatted
        if isinstance(args, (dict, list)):
            args = json.dumps(args)
        elif not isinstance(args, str):
            args = str(args)

        # Create properly formatted tool call
        normalized_call = {
            "id": data.get("id", f"call_{time.time_ns()}"),
            "type": "function",
            "function": {"name": str(name).strip(), "arguments": args},
        }

        if not _validate_tool_call(normalized_call):
            raise ValueError("Normalized tool call failed validation")

        return normalized_call

    except Exception as e:
        logger.error(f"Error normalizing tool call: {str(e)}", exc_info=True)
        raise


def _parse_tool_calls(text: str) -> List[Dict[str, Any]]:
    """Parse tool calls from text with improved error handling and XML support."""
    tool_calls = []

    if not text:
        return []

    logger.debug(f"Parsing tool calls from raw text length: {len(text)}")

    try:
        # First try to parse as pure JSON
        try:
            data = json.loads(text)
            if isinstance(data, dict):
                try:
                    tool_call = _normalize_tool_call(data)
                    if _validate_tool_call(tool_call):
                        tool_calls.append(tool_call)
                        return tool_calls
                except Exception as e:
                    logger.warning(f"Failed to normalize JSON tool call: {str(e)}")
            elif isinstance(data, list):
                for item in data:
                    try:
                        if isinstance(item, dict):
                            tool_call = _normalize_tool_call(item)
                            if _validate_tool_call(tool_call):
                                tool_calls.append(tool_call)
                    except Exception as e:
                        logger.warning(
                            f"Failed to normalize list item tool call: {str(e)}"
                        )
                if tool_calls:
                    return tool_calls
        except json.JSONDecodeError:
            logger.debug("Text is not valid JSON, trying XML patterns")

        # XML Pattern matching with proper namespace handling
        xml_patterns = [
            # Standard XML format
            r'<(?:antml:)?function_calls>\s*<(?:antml:)?invoke\s+name="([^"]+)">(.*?)</(?:antml:)?invoke>\s*</(?:antml:)?function_calls>',
            # Alternative format with function tag
            r'<(?:antml:)?function_calls>\s*<(?:antml:)?function\s+name="([^"]+)">(.*?)</(?:antml:)?function>\s*</(?:antml:)?function_calls>',
            # Simple invoke pattern (fallback)
            r'<(?:antml:)?invoke\s+name="([^"]+)">(.*?)</(?:antml:)?invoke>',
        ]

        for xml_pattern in xml_patterns:
            xml_matches = list(re.finditer(xml_pattern, text, re.DOTALL))
            logger.debug(f"Found {len(xml_matches)} potential XML-style tool calls")

            for xml_match in xml_matches:
                try:
                    tool_name = xml_match.group(1).strip()
                    params_text = xml_match.group(2).strip()
                    tool_args = {}

                    # Parse individual parameters with namespace handling
                    param_pattern = r'<(?:antml:)?parameter\s+name="([^"]+)">(.*?)</(?:antml:)?parameter>'
                    param_matches = list(
                        re.finditer(param_pattern, params_text, re.DOTALL)
                    )

                    if param_matches:
                        for param_match in param_matches:
                            try:
                                param_name = param_match.group(1)
                                param_value = param_match.group(2).strip()

                                # Try to parse as JSON if it looks like JSON
                                if param_value.startswith(
                                    "{"
                                ) or param_value.startswith("["):
                                    try:
                                        tool_args[param_name] = json.loads(param_value)
                                    except json.JSONDecodeError:
                                        tool_args[param_name] = param_value
                                else:
                                    tool_args[param_name] = param_value
                            except Exception as e:
                                logger.warning(
                                    f"Parameter parsing error for {param_name}: {str(e)}"
                                )
                                continue

                        # Create and validate the tool call with the parsed parameters
                        tool_call = _normalize_tool_call(
                            {"name": tool_name, "arguments": tool_args}
                        )

                        if _validate_tool_call(tool_call):
                            tool_calls.append(tool_call)
                            logger.debug(
                                f"Successfully parsed XML tool call: {tool_name}"
                            )

                except Exception as e:
                    logger.warning(
                        f"Error parsing XML tool call: {str(e)}", exc_info=True
                    )
                    continue

            # If we found valid XML tool calls, return them
            if tool_calls:
                return tool_calls

        # Fallback to natural language pattern
        nl_pattern = r"(?:Execute|Run|Invoke|Using|Call)\s+[`']([^`']+)[`']\s+with(?:\s+arguments?)?:?\s*({[^}]+}|\([^)]+\))"
        for match in re.finditer(nl_pattern, text, re.IGNORECASE | re.DOTALL):
            try:
                tool_name = match.group(1).strip()
                args_str = match.group(2).strip()

                # Handle args parsing
                tool_args = {}
                try:
                    # Remove outer parentheses if present
                    if args_str.startswith("(") and args_str.endswith(")"):
                        args_str = args_str[1:-1]
                    tool_args = json.loads(args_str)
                except json.JSONDecodeError:
                    # Fallback to key=value parsing
                    for pair in args_str.split(","):
                        if "=" in pair:
                            k, v = pair.split("=", 1)
                            key = k.strip().strip("\"'")
                            value = v.strip().strip("\"'")
                            tool_args[key] = value

                # Create and validate tool call
                tool_call = _normalize_tool_call(
                    {"name": tool_name, "arguments": tool_args}
                )

                if _validate_tool_call(tool_call):
                    tool_calls.append(tool_call)
                    logger.debug(
                        f"Successfully parsed natural language tool call: {tool_name}"
                    )

            except Exception as e:
                logger.warning(f"Error parsing natural language tool call: {str(e)}")
                continue

        return tool_calls

    except Exception as e:
        logger.error(f"Error in tool call parsing: {str(e)}", exc_info=True)
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
            enhanced_prompt = f"{prompt}\n\n{tool_definitions}{tool_instructions}"

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
                raise ValueError(
                    "Empty completion text received"
                )  # Parse and validate tool calls
            current_tool_calls = self._parse_tool_calls(completion_text)
            validated_tool_calls = []

            # Get available tools based on tool_choice
            available_tools = self._get_available_tools(tool_choice)

            # Validate each tool call and ensure proper structure
            if current_tool_calls:
                for tool_call in current_tool_calls:
                    try:
                        if not isinstance(tool_call, dict):
                            logger.warning(f"Invalid tool call format: {tool_call}")
                            continue

                        if not _validate_tool_call(tool_call):
                            logger.warning(
                                f"Tool call failed structure validation: {tool_call}"
                            )
                            continue

                        # If tools are provided, validate tool exists and parameters match
                        if available_tools:
                            tool_name = tool_call["function"]["name"]
                            if not any(t["name"] == tool_name for t in available_tools):
                                logger.warning(f"Unknown tool requested: {tool_name}")
                                continue

                            # Validate tool parameters
                            if not self._validate_tool_parameters(
                                tool_call, available_tools
                            ):
                                logger.warning(
                                    f"Invalid tool parameters for {tool_name}"
                                )
                                continue

                        validated_tool_calls.append(tool_call)
                    except Exception as e:
                        logger.warning(f"Error validating tool call: {str(e)}")
                        continue

            result = {
                "content": completion_text,
                "tool_calls": validated_tool_calls,
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

        except TokenLimitExceeded as e:
            # Don't retry on token limit errors
            raise e

        except Exception as e:
            last_exception = e
            if attempt < max_retries:
                continue
            logger.error(f"Tool call failed: {str(e)}")
            break

    # If we got a partial result, return it with error indication
    if result:
        result["error"] = str(last_exception)
        return result

    # Otherwise raise the exception
    raise last_exception or RuntimeError("Tool call failed with no error details")


def _get_available_tools(
    self, tool_choice: Union[str, ToolChoice] = ToolChoice.AUTO
) -> List[Dict[str, Any]]:
    """Get list of available tools based on tool choice."""
    try:
        if tool_choice == ToolChoice.NONE:
            return []

        all_tools = self.get_tools()
        if tool_choice == ToolChoice.AUTO:
            return all_tools

        if tool_choice == ToolChoice.REQUIRED and all_tools:
            return all_tools

        logger.warning(f"No tools available for tool_choice: {tool_choice}")
        return []
    except Exception as e:
        logger.error(f"Error getting available tools: {str(e)}")
        return []


def _validate_tool_parameters(
    self, tool_call: Dict[str, Any], tools: List[Dict[str, Any]]
) -> bool:
    """Validate tool call parameters match tool definition."""
    try:
        tool_name = tool_call["function"]["name"]
        tool_def = next((t for t in tools if t["name"] == tool_name), None)
        if not tool_def:
            return False

        # Get arguments as dict
        try:
            args = tool_call["function"]["arguments"]
            if isinstance(args, str):
                args = json.loads(args)
        except json.JSONDecodeError:
            return False

        # Check required parameters
        required_params = tool_def.get("parameters", {}).get("required", [])
        if not all(param in args for param in required_params):
            return False

        # Validate parameter types if schema available
        properties = tool_def.get("parameters", {}).get("properties", {})
        for param, value in args.items():
            if param in properties:
                param_type = properties[param].get("type")
                if param_type == "string" and not isinstance(value, str):
                    return False
                elif param_type == "number" and not isinstance(value, (int, float)):
                    return False
                elif param_type == "array" and not isinstance(value, list):
                    return False
                elif param_type == "object" and not isinstance(value, dict):
                    return False

        return True
    except Exception as e:
        logger.warning(f"Error validating tool parameters: {str(e)}")
        return False


def patch_llm_class():
    """Apply optimized LLM patches with improved error handling."""
    try:
        # Create method wrappers that maintain proper self reference
        def ask_tool_wrapper(self, *args, **kwargs):
            return ask_tool(self, *args, **kwargs)

        def parse_tool_calls_wrapper(self, text):
            return _parse_tool_calls(text)

        def validate_tool_call_wrapper(self, tool_call):
            return _validate_tool_call(tool_call)

        def validate_tool_parameters_wrapper(self, tool_call, tools):
            return _validate_tool_parameters(self, tool_call, tools)

        # Patch both LLM and LLMOptimized classes with optimized methods
        for cls in [LLM, LLMOptimized]:
            cls.ask_tool = ask_tool_wrapper
            cls._parse_tool_calls = parse_tool_calls_wrapper
            cls._validate_tool_call = validate_tool_call_wrapper
            cls._validate_tool_parameters = validate_tool_parameters_wrapper
            logger.info(
                f"Successfully patched {cls.__name__} with optimized tool methods"
            )

    except Exception as e:
        logger.error(f"Failed to patch LLM classes: {e}")
        raise


# Apply patches when module is imported
patch_llm_class()


def ask_tool(tool_call: Dict[str, Any], tools: Dict[str, Callable], **kwargs) -> str:
    """Process and execute a tool call with proper error handling."""
    try:
        # First normalize and validate the tool call
        tool_call = _normalize_tool_call(tool_call)
        if not _validate_tool_call(tool_call):
            raise ValueError("Invalid tool call format")

        # Extract function name and arguments
        name = tool_call["function"]["name"]
        args_str = tool_call["function"]["arguments"]

        # Get the tool function
        tool_fn = tools.get(name)
        if not tool_fn:
            raise ValueError(f"Unknown tool: {name}")

        # Parse arguments safely
        try:
            if args_str and args_str.strip():
                args = json.loads(args_str)
            else:
                args = {}
        except json.JSONDecodeError:
            logger.warning(f"Failed to parse arguments as JSON: {args_str}")
            args = {"input": args_str}  # Fallback to treating as single input

        # Execute tool with proper error handling
        try:
            if isinstance(args, dict):
                result = tool_fn(**args, **kwargs)
            else:
                result = tool_fn(args, **kwargs)

            # Handle different result types
            if result is None:
                return "Tool executed successfully with no output."
            elif isinstance(result, (str, int, float, bool)):
                return str(result)
            else:
                return json.dumps(result)

        except TypeError as te:
            # Handle argument mismatch errors
            logger.error(f"Tool execution type error: {str(te)}")
            sig = inspect.signature(tool_fn)
            expected_args = list(sig.parameters.keys())
            raise ValueError(
                f"Invalid arguments for tool {name}. Expected: {expected_args}"
            )

    except Exception as e:
        error_msg = f"Tool execution failed: {str(e)}"
        logger.error(error_msg, exc_info=True)
        return error_msg  # Return error message for the agent to handle


def wrap_tools(tools: Dict[str, Callable]) -> Dict[str, Callable]:
    """Wrap tool functions with proper argument handling and validation."""
    wrapped_tools = {}

    for name, fn in tools.items():

        def wrapped_fn(*args, **kwargs):
            try:
                # Handle both positional and keyword arguments
                if args and not kwargs:
                    if len(args) == 1 and isinstance(args[0], dict):
                        return fn(**args[0])
                    return fn(*args)
                return fn(**kwargs)
            except Exception as e:
                logger.error(f"Error in {name}: {str(e)}", exc_info=True)
                raise

        # Preserve function metadata
        wrapped_fn.__name__ = fn.__name__
        wrapped_fn.__doc__ = fn.__doc__
        wrapped_fn.__annotations__ = fn.__annotations__

        wrapped_tools[name] = wrapped_fn

    return wrapped_tools
