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


def _parse_tool_calls(text: str) -> List[Dict[str, Any]]:
    """Parse tool calls from text with improved error handling."""
    tool_calls = []

    try:
        logger.debug(f"Raw LLM output to parse: {text}")

        # First try to parse as pure JSON
        try:
            data = json.loads(text)
            logger.debug(f"Successfully parsed as JSON: {data}")
            if isinstance(data, dict) and "name" in data:
                tool_calls.append(data)
                return tool_calls
            elif isinstance(data, list):
                for item in data:
                    if isinstance(item, dict) and "name" in item:
                        tool_calls.append(item)
                return tool_calls
        except json.JSONDecodeError:
            logger.debug("Failed to parse as pure JSON, trying other patterns")
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

            tool_calls.append({"name": name, "arguments": args})

        # Pattern 2: JSON blocks
        pattern = r"```(?:json)?\s*([\s\S]*?)\s*```"
        matches = re.finditer(pattern, text)

        for match in matches:
            try:
                data = json.loads(match.group(1))
                if isinstance(data, dict) and "name" in data:
                    tool_calls.append(data)
                elif isinstance(data, list):
                    for item in data:
                        if isinstance(item, dict) and "name" in item:
                            tool_calls.append(item)
            except json.JSONDecodeError:
                continue

        return tool_calls

    except Exception as e:
        logger.error(f"Error parsing tool calls from: '{text}'. Error: {str(e)}")
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
                raise ValueError("Empty completion text received")

            # Parse tool calls            tool_calls = self._parse_tool_calls(completion_text)

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

        # Patch both LLM and LLMOptimized classes
        for cls in [LLM, LLMOptimized]:
            cls.ask_tool = ask_tool_wrapper
            cls._parse_tool_calls = parse_tool_calls_wrapper
            logger.info(f"Successfully patched {cls.__name__} with tool methods")
    except Exception as e:
        logger.error(f"Failed to patch LLM classes: {e}")
        raise


# Apply patches when module is imported
patch_llm_class()
