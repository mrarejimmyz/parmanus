"""Fixes for LLM tool patching issues."""

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
    """Format tool definitions for the prompt with JSON function call format."""
    if not tools:
        return ""

    tool_str = "\nAvailable tools:\n"
    for tool in tools:
        # Handle both direct format and OpenAI function call format
        if "function" in tool:
            func_data = tool["function"]
            name = func_data.get("name", "unknown")
            description = func_data.get("description", "")
            parameters = func_data.get("parameters", {})
        else:
            name = tool.get("name", "unknown")
            description = tool.get("description", "")
            parameters = tool.get("parameters", {})

        tool_str += f"\n- {name}: {description}"
        if parameters and "properties" in parameters:
            tool_str += "\n  Parameters:"
            for param_name, param_info in parameters.get("properties", {}).items():
                required = param_name in parameters.get("required", [])
                tool_str += f"\n    - {param_name} ({'required' if required else 'optional'}): {param_info.get('description', '')}"

    tool_str += """\n
To use tools, respond with a JSON object in this format:
{
  "id": "call_123456789",
  "type": "function",
  "function": {
    "name": "tool_name",
    "arguments": "{\"param1\": \"value1\", \"param2\": \"value2\"}"
  }
}"""
    return tool_str


def get_tool_instructions(tool_choice: Union[str, ToolChoice]) -> str:
    """Get tool instructions based on tool choice."""
    if tool_choice == ToolChoice.AUTO:
        return (
            "\nYou can use tools if needed, but you can also respond directly if no tools are required. "
            "If you use a tool, respond with the JSON function call format. "
            "If you respond normally, provide a helpful answer to the user's request."
        )
    elif tool_choice == ToolChoice.REQUIRED:
        return (
            "\nYou MUST use one of the provided tools to complete this task. "
            "Respond with the JSON function call object."
        )
    elif tool_choice == ToolChoice.NONE:
        return (
            "\nDo not use any tools for this task. Respond normally to help the user."
        )
    return ""


def _parse_tool_calls(text: str) -> List[Dict[str, Any]]:
    """Parse tool calls from text with improved error handling."""
    tool_calls = []

    try:
        logger.debug(f"Raw LLM output to parse: {text}")

        # First try to parse as pure JSON
        try:
            data = json.loads(text)
            logger.debug(f"Successfully parsed as JSON: {data}")
            if isinstance(data, dict):
                if "name" in data:
                    # Convert to proper structure with function wrapper
                    tool_calls.append(
                        {
                            "id": f"call_{time.time_ns()}",
                            "type": "function",
                            "function": {
                                "name": data["name"],
                                "arguments": (
                                    json.dumps(data.get("arguments", {}))
                                    if isinstance(data.get("arguments"), dict)
                                    else str(data.get("arguments", ""))
                                ),
                            },
                        }
                    )
                    return tool_calls
                elif (
                    "function" in data
                    and isinstance(data["function"], dict)
                    and "name" in data["function"]
                ):
                    # Already in correct format
                    tool_calls.append(data)
                    return tool_calls
            elif isinstance(data, list):
                for item in data:
                    if isinstance(item, dict):
                        if "name" in item:
                            # Convert to proper structure with function wrapper
                            tool_calls.append(
                                {
                                    "id": f"call_{time.time_ns()}",
                                    "type": "function",
                                    "function": {
                                        "name": item["name"],
                                        "arguments": (
                                            json.dumps(item.get("arguments", {}))
                                            if isinstance(item.get("arguments"), dict)
                                            else str(item.get("arguments", ""))
                                        ),
                                    },
                                }
                            )
                        elif (
                            "function" in item
                            and isinstance(item["function"], dict)
                            and "name" in item["function"]
                        ):
                            # Already in correct format
                            tool_calls.append(item)
                if tool_calls:
                    return tool_calls
        except json.JSONDecodeError:
            logger.debug("Failed to parse as pure JSON, trying other patterns")
            pass

        # Extract JSON objects from mixed text content (looking for function call structure)
        # Pattern to find JSON with function call structure
        import re

        json_pattern = r'\{[^{}]*?"type":\s*"function".*?\}'
        potential_json = re.search(json_pattern, text, re.DOTALL)

        if potential_json:
            try:
                # Try to extract the complete JSON object by counting braces
                start_pos = potential_json.start()
                brace_count = 0
                end_pos = start_pos

                for i, char in enumerate(text[start_pos:], start_pos):
                    if char == "{":
                        brace_count += 1
                    elif char == "}":
                        brace_count -= 1
                        if brace_count == 0:
                            end_pos = i + 1
                            break

                if brace_count == 0:
                    json_str = text[start_pos:end_pos]
                    data = json.loads(json_str)
                    logger.debug(
                        f"Successfully extracted JSON from mixed content: {data}"
                    )

                    if isinstance(data, dict) and "function" in data:
                        tool_calls.append(data)
                        return tool_calls

            except json.JSONDecodeError as e:
                logger.debug(f"Failed to parse extracted JSON: {e}")
                pass

        # Pattern 1: Natural language tool calls
        pattern = r"(?:Called|Using|Execute|Run|Invoke|Call)\s+`([^`]+)`\s+with(?:\s+arguments?)?:?\s*{([^}]+)}"
        matches = re.finditer(pattern, text, re.IGNORECASE)

        for match in matches:
            name = match.group(1).strip()
            args_str = match.group(2).strip()
            try:
                # Try to parse as JSON first
                args = json.loads("{" + args_str + "}")
            except json.JSONDecodeError:
                # Fallback to simple key=value parsing
                args = {}
                for pair in args_str.split(","):
                    if "=" in pair:
                        k, v = pair.split("=", 1)
                        args[k.strip()] = v.strip().strip("\"'")

            # Create properly formatted tool call
            tool_calls.append(
                {
                    "id": f"call_{time.time_ns()}",
                    "type": "function",
                    "function": {"name": name, "arguments": json.dumps(args)},
                }
            )  # Pattern 2: JSON blocks
        pattern = r"```(?:json)?\s*([\s\S]*?)\s*```"
        matches = re.finditer(pattern, text)

        for match in matches:
            try:
                data = json.loads(match.group(1))
                if isinstance(data, dict):
                    if "name" in data:
                        # Convert to proper structure with function wrapper
                        tool_calls.append(
                            {
                                "id": f"call_{time.time_ns()}",
                                "type": "function",
                                "function": {
                                    "name": data["name"],
                                    "arguments": (
                                        json.dumps(data.get("arguments", {}))
                                        if isinstance(data.get("arguments"), dict)
                                        else str(data.get("arguments", ""))
                                    ),
                                },
                            }
                        )
                    elif (
                        "function" in data
                        and isinstance(data["function"], dict)
                        and "name" in data["function"]
                    ):
                        # Already in correct format
                        tool_calls.append(data)
                elif isinstance(data, list):
                    for item in data:
                        if isinstance(item, dict):
                            if "name" in item:
                                # Convert to proper structure with function wrapper
                                tool_calls.append(
                                    {
                                        "id": f"call_{time.time_ns()}",
                                        "type": "function",
                                        "function": {
                                            "name": item["name"],
                                            "arguments": (
                                                json.dumps(item.get("arguments", {}))
                                                if isinstance(
                                                    item.get("arguments"), dict
                                                )
                                                else str(item.get("arguments", ""))
                                            ),
                                        },
                                    }
                                )
                            elif (
                                "function" in item
                                and isinstance(item["function"], dict)
                                and "name" in item["function"]
                            ):
                                # Already in correct format
                                tool_calls.append(item)
            except json.JSONDecodeError:
                continue

        # Pattern 3: XML-style function calls
        xml_pattern = r'<(?:antml:)?function_calls>\s*<(?:antml:)?(?:invoke|function)\s+name="([^"]+)">(.*?)</(?:antml:)?(?:invoke|function)>\s*</(?:antml:)?function_calls>'
        xml_matches = re.finditer(xml_pattern, text, re.DOTALL)

        for xml_match in xml_matches:
            try:
                tool_name = xml_match.group(1).strip()
                params_text = xml_match.group(2).strip()
                tool_args = {}

                # Parse individual parameters
                param_pattern = r'<(?:antml:)?parameter\s+name="([^"]+)">(.*?)</(?:antml:)?parameter>'
                param_matches = re.finditer(param_pattern, params_text, re.DOTALL)

                for param_match in param_matches:
                    param_name = param_match.group(1)
                    param_value = param_match.group(2).strip()

                    # Try to parse as JSON if it looks like a JSON value
                    if param_value.startswith("{") or param_value.startswith("["):
                        try:
                            tool_args[param_name] = json.loads(param_value)
                        except json.JSONDecodeError:
                            tool_args[param_name] = param_value
                    else:
                        tool_args[param_name] = param_value

                # Create properly formatted tool call
                tool_calls.append(
                    {
                        "id": f"call_{time.time_ns()}",
                        "type": "function",
                        "function": {
                            "name": tool_name,
                            "arguments": json.dumps(tool_args),
                        },
                    }
                )
            except Exception as e:
                logger.warning(f"Failed to parse XML tool call: {str(e)}")
                continue

        return tool_calls

    except Exception as e:
        logger.error(
            f"Error parsing tool calls from: '{text}'. Error: {str(e)}", exc_info=True
        )
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
            )  # Prepare the prompt
            prompt = self._format_prompt_for_llama(formatted_messages)
            tool_definitions = format_tool_definitions(tools) if tools else ""
            tool_instructions = get_tool_instructions(tool_choice)
            enhanced_prompt = f"{prompt}\n\n{tool_definitions}{tool_instructions}"  # Run model with timeout
            completion = await asyncio.wait_for(
                asyncio.get_event_loop().run_in_executor(
                    self._executor,
                    lambda: self.text_model.create_completion(
                        prompt=enhanced_prompt,
                        max_tokens=min(self.max_tokens, self.MAX_ALLOWED_OUTPUT_TOKENS),
                        temperature=temp,
                        stop=["<|user|>", "<|system|>", "<|eot_id|>"],
                        **kwargs,
                    ),
                ),
                timeout=timeout,
            )

            # Extract and validate completion text
            completion_text = completion.get("choices", [{}])[0].get("text", "").strip()
            if not completion_text:
                raise ValueError("Empty completion text received")  # Parse tool calls
            try:
                logger.debug(
                    f"About to parse tool calls from: {completion_text[:100]}..."
                )
                tool_calls = self._parse_tool_calls(completion_text)
                logger.debug(f"Parsed {len(tool_calls)} tool calls: {tool_calls}")
            except Exception as e:
                logger.error(f"Error parsing tool calls: {str(e)}", exc_info=True)
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
            return _parse_tool_calls(text)  # Patch both LLM and LLMOptimized classes

        for cls in [LLM, LLMOptimized]:
            cls.ask_tool = ask_tool_wrapper
            cls._parse_tool_calls = parse_tool_calls_wrapper

            # Add the missing _format_prompt_for_llama method if it doesn't exist with the right signature
            if not hasattr(cls, "_format_prompt_for_llama") or not callable(
                getattr(cls, "_format_prompt_for_llama")
            ):

                def format_prompt_wrapper(self, messages):
                    """Format messages for Llama models - compatibility wrapper."""
                    if hasattr(self, "format_messages"):
                        return self.format_messages(messages)
                    else:
                        # Fallback implementation
                        prompt = ""
                        for msg in messages:
                            role = msg.get("role", "user")
                            content = msg.get("content", "")
                            if role == "system":
                                prompt += f"<|system|>\n{content}\n"
                            elif role == "user":
                                prompt += f"<|user|>\n{content}\n"
                            elif role == "assistant":
                                prompt += f"<|assistant|>\n{content}\n"
                        prompt += "<|assistant|>\n"
                        return prompt

                cls._format_prompt_for_llama = format_prompt_wrapper

            logger.info(f"Successfully patched {cls.__name__} with tool methods")
    except Exception as e:
        logger.error(f"Failed to patch LLM classes: {e}")
        raise


# Apply patches when module is imported
patch_llm_class()
