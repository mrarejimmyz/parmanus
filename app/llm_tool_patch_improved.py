"""Improved LLM tool patching to fix empty completion issues."""

import asyncio
import json
import logging
import re
import time
from typing import Any, Dict, List, Optional, Union

from app.exceptions import ModelTimeoutError, TokenLimitExceeded
from app.logger import logger
from app.schema import Message, ToolChoice


def calculate_adaptive_timeout(
    messages: List[Any], tools: Optional[List[Dict[str, Any]]] = None
) -> int:
    """Calculate adaptive timeout based on input complexity."""
    base_timeout = 30

    # Calculate message complexity
    total_length = sum(len(str(msg)) for msg in messages)
    if total_length > 5000:
        base_timeout = 60
    elif total_length > 10000:
        base_timeout = 90

    # Add time for tools if present
    if tools and len(tools) > 5:
        base_timeout += 15

    return min(base_timeout, 120)  # Cap at 2 minutes


def format_tool_definitions(tools: Optional[List[Dict[str, Any]]]) -> str:
    """Format tool definitions for the prompt in a model-friendly way."""
    if not tools:
        return ""

    tool_str = "\n## Available Tools\n"
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

        tool_str += f"\n**{name}**: {description}"
        if parameters and "properties" in parameters:
            tool_str += "\n  - Parameters:"
            for param_name, param_info in parameters.get("properties", {}).items():
                required = param_name in parameters.get("required", [])
                tool_str += f"\n    - `{param_name}` ({'required' if required else 'optional'}): {param_info.get('description', '')}"

    tool_str += """\n
## Tool Usage Format
When you need to use a tool, respond with a JSON object like this:
```json
{
  "type": "function",
  "function": {
    "name": "tool_name",
    "arguments": "{\"param1\": \"value1\"}"
  }
}
```
"""
    return tool_str


def get_tool_instructions(tool_choice: Union[str, ToolChoice]) -> str:
    """Get tool instructions based on tool choice."""
    if tool_choice == ToolChoice.AUTO:
        return (
            "\n## Instructions\n"
            "You can use the available tools if they would help complete the user's request, "
            "or respond directly if no tools are needed. Choose the most appropriate approach "
            "based on what the user is asking for."
        )
    elif tool_choice == ToolChoice.REQUIRED:
        return (
            "\n## Instructions\n"
            "You must use one of the available tools to complete this task. "
            "Choose the most appropriate tool and provide the necessary parameters."
        )
    elif tool_choice == ToolChoice.NONE:
        return (
            "\n## Instructions\n"
            "Respond directly to help the user without using any tools."
        )
    return ""


def _format_tool_call(data: Dict[str, Any]) -> Dict[str, Any]:
    """Format a tool call into the expected structure."""
    if "function" in data and isinstance(data["function"], dict):
        # Already in correct format
        return {
            "id": data.get("id", f"call_{time.time_ns()}"),
            "type": "function",
            "function": data["function"],
        }
    elif "name" in data:
        # Convert from simple format
        arguments = data.get("arguments", {})
        if isinstance(arguments, dict):
            arguments = json.dumps(arguments)
        elif not isinstance(arguments, str):
            arguments = str(arguments)

        return {
            "id": f"call_{time.time_ns()}",
            "type": "function",
            "function": {
                "name": data["name"],
                "arguments": arguments,
            },
        }
    else:
        raise ValueError(f"Invalid tool call format: {data}")


def _parse_tool_calls(text: str) -> List[Dict[str, Any]]:
    """Parse tool calls from text with improved error handling."""
    tool_calls = []
    logger.debug(f"Parsing tool calls from: {text[:200]}...")

    try:
        # First try to parse as pure JSON
        try:
            data = json.loads(text.strip())
            if isinstance(data, dict):
                if "function" in data or "name" in data:
                    tool_calls.append(_format_tool_call(data))
                    return tool_calls
            elif isinstance(data, list):
                for item in data:
                    if isinstance(item, dict) and (
                        "function" in item or "name" in item
                    ):
                        tool_calls.append(_format_tool_call(item))
                return tool_calls
        except json.JSONDecodeError:
            pass

        # Look for JSON in code blocks
        json_block_pattern = r"```(?:json)?\s*([\s\S]*?)\s*```"
        json_matches = re.finditer(json_block_pattern, text, re.IGNORECASE)

        for match in json_matches:
            try:
                json_content = match.group(1).strip()
                data = json.loads(json_content)
                if isinstance(data, dict) and ("function" in data or "name" in data):
                    tool_calls.append(_format_tool_call(data))
                elif isinstance(data, list):
                    for item in data:
                        if isinstance(item, dict) and (
                            "function" in item or "name" in item
                        ):
                            tool_calls.append(_format_tool_call(item))
            except json.JSONDecodeError:
                continue

        # Look for JSON objects in mixed text
        json_pattern = r'\{[^{}]*?"(?:function|name)"\s*:[^{}]*?\}'
        json_matches = re.finditer(json_pattern, text, re.DOTALL)

        for match in json_matches:
            try:
                # Try to extract complete JSON by counting braces
                start_pos = match.start()
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
                    if isinstance(data, dict) and (
                        "function" in data or "name" in data
                    ):
                        tool_calls.append(_format_tool_call(data))
            except json.JSONDecodeError:
                continue

        return tool_calls

    except Exception as e:
        logger.error(f"Error parsing tool calls: {e}", exc_info=True)
        return []


async def ask_tool(
    self,
    messages: List[Union[Message, Dict[str, Any]]],
    system_msgs: Optional[List[Union[Message, Dict[str, Any]]]] = None,
    tools: Optional[List[Dict[str, Any]]] = None,
    tool_choice: Union[str, ToolChoice] = ToolChoice.AUTO,
    temp: float = 0.1,  # Slightly higher temperature for more creativity
    timeout: Optional[int] = None,
    max_retries: int = 2,
    **kwargs,
) -> Dict[str, Any]:
    """Enhanced ask_tool method with improved error handling."""

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

            # Format messages properly
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

            # Prepare the prompt with better formatting
            prompt = self._format_prompt_for_llama(formatted_messages)

            # Add tool information if tools are available
            tool_section = ""
            if tools:
                tool_definitions = format_tool_definitions(tools)
                tool_instructions = get_tool_instructions(tool_choice)
                tool_section = f"{tool_definitions}{tool_instructions}"

            # Create enhanced prompt with clear structure
            if tool_section:
                enhanced_prompt = f"{prompt}\n{tool_section}\n\nPlease respond:"
            else:
                enhanced_prompt = prompt

            logger.debug(f"Enhanced prompt length: {len(enhanced_prompt)}")

            # Run model with timeout
            completion = await asyncio.wait_for(
                asyncio.get_event_loop().run_in_executor(
                    self._executor,
                    lambda: self.text_model.create_completion(
                        prompt=enhanced_prompt,
                        max_tokens=min(self.max_tokens, 2048),  # Reasonable limit
                        temperature=temp,
                        stop=[
                            "<|user|>",
                            "<|system|>",
                            "<|eot_id|>",
                            "Human:",
                            "User:",
                        ],
                        **kwargs,
                    ),
                ),
                timeout=timeout,
            )

            # Extract completion text
            completion_text = completion.get("choices", [{}])[0].get("text", "").strip()

            if not completion_text:
                # If we get empty completion, try with different parameters
                if attempt == 0:
                    logger.warning(
                        "Empty completion, retrying with adjusted parameters"
                    )
                    continue
                else:
                    raise ValueError("Empty completion text received after retries")

            logger.debug(f"Completion text: {completion_text[:200]}...")

            # Parse tool calls
            tool_calls = []
            if tools:  # Only try to parse tool calls if tools are available
                try:
                    tool_calls = self._parse_tool_calls(completion_text)
                    logger.debug(f"Parsed {len(tool_calls)} tool calls")
                except Exception as e:
                    logger.error(f"Error parsing tool calls: {e}")
                    tool_calls = []

            # Build response
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
                timeout = min(timeout * 1.2, 180)  # Gradually increase timeout
                continue
            logger.error(f"Tool call timed out after {max_retries + 1} attempts")
            break

        except Exception as e:
            last_exception = e
            if attempt < max_retries and not isinstance(e, TokenLimitExceeded):
                # Try with reduced parameters
                if "max_tokens" in kwargs:
                    kwargs["max_tokens"] = min(kwargs["max_tokens"] // 2, 1024)
                temp = max(temp - 0.1, 0.0)  # Reduce temperature
                continue
            logger.error(f"Tool call failed: {e}")
            break

    # Return partial result if available, otherwise raise exception
    if result:
        result["error"] = str(last_exception)
        return result

    raise last_exception or RuntimeError("Tool call failed with no error details")


def patch_llm_class():
    """Apply LLM patches with improved error handling."""
    from app.llm import LLM
    from app.llm_optimized import LLMOptimized

    try:
        # Create method wrappers
        def ask_tool_wrapper(self, *args, **kwargs):
            return ask_tool(self, *args, **kwargs)

        def parse_tool_calls_wrapper(self, text):
            return _parse_tool_calls(text)

        def format_tool_call_wrapper(self, data):
            return _format_tool_call(data)

        # Patch both LLM classes
        for cls in [LLM, LLMOptimized]:
            cls.ask_tool = ask_tool_wrapper
            cls._parse_tool_calls = parse_tool_calls_wrapper
            cls._format_tool_call = format_tool_call_wrapper

            # Ensure the formatting method exists
            if not hasattr(cls, "_format_prompt_for_llama"):

                def format_prompt_wrapper(self, messages):
                    """Format messages for Llama models."""
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

            logger.info(
                f"Successfully patched {cls.__name__} with improved tool methods"
            )

    except Exception as e:
        logger.error(f"Failed to patch LLM classes: {e}")
        raise


# Apply patches when module is imported
patch_llm_class()
