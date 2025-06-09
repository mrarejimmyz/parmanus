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
IMPORTANT: When using tools, respond with ONLY a JSON object in this exact format:
{
  "id": "call_123456789",
  "type": "function",
  "function": {
    "name": "tool_name",
    "arguments": "{\"param1\": \"value1\", \"param2\": \"value2\"}"
  }
}

Do NOT include any other text, explanations, or markdown formatting. Only return the JSON function call object."""
    return tool_str


def get_tool_instructions(tool_choice: Union[str, ToolChoice]) -> str:
    """Get tool instructions based on tool choice."""
    if tool_choice == ToolChoice.AUTO:
        return (
            "\nYou can use tools if needed, but don't force their use if not necessary. "
            "If you use a tool, respond with the JSON function call format only."
        )
    elif tool_choice == ToolChoice.REQUIRED:
        return (
            "\nYou MUST use one of the provided tools to complete this task. "
            "Respond with ONLY the JSON function call object - no other text or explanations."
        )
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
            data = json.loads(text.strip())
            logger.debug(f"Successfully parsed as pure JSON: {data}")
            if isinstance(data, dict):
                if "function" in data and isinstance(data["function"], dict):
                    # Already in correct format
                    tool_calls.append(data)
                    return tool_calls
                elif "name" in data:
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
        except json.JSONDecodeError:
            logger.debug("Failed to parse as pure JSON, trying pattern extraction")
            pass

        # Extract JSON objects from mixed text content
        # Look for JSON objects that contain the function call structure
        json_pattern = r'\{[^{}]*?"type":\s*"function"[^{}]*?\}'

        # First try simple pattern match
        potential_matches = re.finditer(json_pattern, text, re.DOTALL)

        for match in potential_matches:
            # Extract complete JSON by balancing braces
            start_pos = match.start()
            brace_count = 0
            end_pos = len(text)

            for i, char in enumerate(text[start_pos:], start_pos):
                if char == "{":
                    brace_count += 1
                elif char == "}":
                    brace_count -= 1
                    if brace_count == 0:
                        end_pos = i + 1
                        break

            try:
                json_str = text[start_pos:end_pos]
                data = json.loads(json_str)
                logger.debug(f"Successfully extracted JSON from mixed content: {data}")

                if isinstance(data, dict) and "function" in data:
                    tool_calls.append(data)

            except json.JSONDecodeError as e:
                logger.debug(f"Failed to parse extracted JSON: {e}")
                continue

        if tool_calls:
            return tool_calls

        # Alternative pattern - look for any JSON object in the text
        json_objects = re.finditer(r"\{[^{}]*(?:\{[^{}]*\}[^{}]*)*\}", text)

        for match in json_objects:
            try:
                json_str = match.group(0)
                data = json.loads(json_str)

                # Check if it looks like a function call
                if isinstance(data, dict):
                    if "function" in data and isinstance(data["function"], dict):
                        if "name" in data["function"]:
                            tool_calls.append(data)
                    elif "name" in data:
                        # Convert to proper format
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

            except json.JSONDecodeError:
                continue

        if tool_calls:
            return tool_calls

        # Pattern for natural language tool calls as fallback
        pattern = r"(?:Called|Using|Execute|Run|Invoke|Call)\s+`([^`]+)`\s+with(?:\s+arguments?)?:?\s*{([^}]+)}"
        matches = re.finditer(pattern, text, re.IGNORECASE)

        for match in matches:
            name = match.group(1).strip()
            args_str = match.group(2).strip()
            try:
                args = json.loads("{" + args_str + "}")
            except json.JSONDecodeError:
                args = {}
                for pair in args_str.split(","):
                    if "=" in pair:
                        k, v = pair.split("=", 1)
                        args[k.strip()] = v.strip().strip("\"'")

            tool_calls.append(
                {
                    "id": f"call_{time.time_ns()}",
                    "type": "function",
                    "function": {"name": name, "arguments": json.dumps(args)},
                }
            )

    except Exception as e:
        logger.error(f"Unexpected error parsing tool calls: {e}", exc_info=True)

    return tool_calls


async def ask_tool(
    self,
    messages: List[Message],
    tools: Optional[List[Dict[str, Any]]] = None,
    tool_choice: Union[str, ToolChoice] = ToolChoice.AUTO,
    temp: float = 0.1,
    max_retries: int = 2,
    timeout: Optional[int] = None,
    **kwargs,
) -> Dict[str, Any]:
    """Ask LLM with tool calling capability with improved error handling."""

    start_time = time.time()
    last_exception = None
    result = None

    if timeout is None:
        timeout = calculate_adaptive_timeout(messages, tools)

    for attempt in range(max_retries + 1):
        try:
            logger.debug(f"Tool call attempt {attempt + 1}")

            # Process system messages
            system_msgs = [msg for msg in messages if msg.role == "system"]
            user_msgs = [msg for msg in messages if msg.role != "system"]

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
                [msg if isinstance(msg, dict) else msg.to_dict() for msg in user_msgs]
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
                        stop=["<|user|>", "<|system|>", "<|eot_id|>"],
                        **kwargs,
                    ),
                ),
                timeout=timeout,
            )

            # Extract and validate completion text
            completion_text = completion.get("choices", [{}])[0].get("text", "").strip()
            if not completion_text:
                raise ValueError("Empty completion text received")

            # Parse tool calls
            try:
                logger.debug(
                    f"About to parse tool calls from: {completion_text[:100]}..."
                )
                tool_calls = _parse_tool_calls(completion_text)
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
            return _parse_tool_calls(text)

        # Patch both LLM and LLMOptimized classes
        for cls in [LLM, LLMOptimized]:
            cls.ask_tool = ask_tool_wrapper
            cls._parse_tool_calls = parse_tool_calls_wrapper

            # Add the missing _format_prompt_for_llama method if it doesn't exist
            if not hasattr(cls, "_format_prompt_for_llama") or not callable(
                getattr(cls, "_format_prompt_for_llama")
            ):

                def _format_prompt_for_llama_wrapper(self, messages):
                    if hasattr(self, "_format_chat_prompt") and callable(
                        getattr(self, "_format_chat_prompt")
                    ):
                        return self._format_chat_prompt(messages)

                    # Fallback implementation
                    formatted = ""
                    for msg in messages:
                        role = msg.get("role", "user")
                        content = msg.get("content", "")
                        formatted += f"<|start_header_id|>{role}<|end_header_id|>\n\n{content}<|eot_id|>"

                    formatted += "<|start_header_id|>assistant<|end_header_id|>\n\n"
                    return formatted

                cls._format_prompt_for_llama = _format_prompt_for_llama_wrapper

        logger.info("Successfully patched LLMOptimized with tool methods")

    except Exception as e:
        logger.error(f"Failed to patch LLM classes: {str(e)}", exc_info=True)
        raise


# Apply the patch when the module is imported
patch_llm_class()
