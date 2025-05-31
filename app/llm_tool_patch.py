import asyncio
import json
import logging
import re
import types
from typing import Any, Dict, List, Optional, Union

from app.schema import TOOL_CHOICE_TYPE, Message, ToolChoice

logger = logging.getLogger(__name__)


class TokenLimitExceeded(Exception):
    """Exception raised when token limit is exceeded."""

    pass


async def ask_tool(
    self,
    messages: List[Union[Message, Dict[str, Any]]],
    system_msgs: Optional[List[Union[Message, Dict[str, Any]]]] = None,
    tools: Optional[List[Dict[str, Any]]] = None,
    tool_choice: TOOL_CHOICE_TYPE = ToolChoice.AUTO,
    temp: float = 0.0,
    timeout: int = 60,
    **kwargs,
) -> Dict[str, Any]:
    """
    Ask the model to use tools based on the messages.

    Args:
        messages: List of messages to send to the model
        system_msgs: Optional system messages to prepend
        tools: List of tool definitions
        tool_choice: Whether tool use is "auto", "required", or "none"
        temp: Temperature for sampling
        timeout: Timeout in seconds
        **kwargs: Additional arguments to pass to the model

    Returns:
        Dictionary containing the model's response and any tool calls
    """
    try:
        # Apply safe max tokens limit
        safe_max_tokens = min(self.max_tokens, self.MAX_ALLOWED_OUTPUT_TOKENS)

        # Create a task for model completion with timeout and run in thread pool
        try:
            # Format messages for the model
            if system_msgs:
                system_msgs = self.format_messages(system_msgs)
                formatted_messages = system_msgs + self.format_messages(messages)
            else:
                formatted_messages = self.format_messages(messages)

            # Prepare tool definitions for the prompt
            tool_definitions = ""
            if tools:
                tool_definitions = "Available tools:\n"
                for tool in tools:
                    # Safely access tool properties with fallbacks
                    tool_name = tool.get("name", "unnamed_tool")
                    tool_description = tool.get(
                        "description", "No description available"
                    )
                    tool_definitions += f"- {tool_name}: {tool_description}\n"

                    # Safely handle parameters if present
                    if "parameters" in tool:
                        try:
                            params_json = json.dumps(tool["parameters"])
                            tool_definitions += f"  Parameters: {params_json}\n"
                        except (TypeError, ValueError) as e:
                            logger.warning(
                                f"Failed to serialize parameters for tool {tool_name}: {e}"
                            )
                            tool_definitions += f"  Parameters: [Error: Could not serialize parameters]\n"
                tool_definitions += "\n"

            # Add tool instructions based on tool_choice
            tool_instructions = ""
            if tool_choice == ToolChoice.REQUIRED:
                tool_instructions = (
                    "You MUST use one of the available tools to respond.\n"
                )
            elif tool_choice == ToolChoice.AUTO:
                tool_instructions = "Use tools when appropriate to complete the task.\n"

            # Format messages into a prompt string
            prompt = self._format_prompt_for_llama(formatted_messages)

            # Enhance prompt with tool information
            enhanced_prompt = f"{prompt}\n\n{tool_definitions}{tool_instructions}"

            # Run model inference in thread pool to avoid blocking the event loop
            completion = await asyncio.wait_for(
                asyncio.get_event_loop().run_in_executor(
                    self._executor,
                    lambda: self.text_model.create_completion(
                        prompt=enhanced_prompt,
                        max_tokens=safe_max_tokens,
                        temperature=temp,
                        stop=["<|user|>", "<|system|>"],
                        **kwargs,
                    ),
                ),
                timeout=timeout,
            )

            # Extract completion text
            completion_text = completion.get("choices", [{}])[0].get("text", "").strip()

            # Parse tool calls from completion text
            tool_calls = self._parse_tool_calls(completion_text)

            # Estimate token counts
            prompt_tokens = self.count_tokens(enhanced_prompt)
            completion_tokens = self.count_tokens(completion_text)

            # Update token counter
            self.update_token_count(prompt_tokens, completion_tokens)

            # Return structured response with tool calls
            return {
                "content": completion_text,
                "tool_calls": tool_calls,
                "usage": {
                    "prompt_tokens": prompt_tokens,
                    "completion_tokens": completion_tokens,
                    "total_tokens": prompt_tokens + completion_tokens,
                },
            }
        except asyncio.TimeoutError:
            logger.error(f"Model completion timed out after {timeout} seconds")
            # Return partial result if available
            return {
                "content": f"[Response incomplete due to timeout after {timeout} seconds]",
                "tool_calls": [],
                "usage": {
                    "prompt_tokens": self.count_tokens(enhanced_prompt),
                    "completion_tokens": 0,
                    "total_tokens": self.count_tokens(enhanced_prompt),
                },
            }
        except Exception as e:
            logger.error(f"Error in model completion: {e}")
            raise
    except TokenLimitExceeded:
        # Re-raise token limit errors without logging
        raise
    except Exception as e:
        logger.error(f"Unexpected error in ask_tool: {e}")
        raise


def _parse_tool_calls(self, text: str) -> List[Dict[str, Any]]:
    """
    Parse tool calls from completion text.

    Args:
        text: Completion text from the model

    Returns:
        List of parsed tool calls
    """
    tool_calls = []

    try:
        # Look for tool call patterns in the text
        # Pattern 1: Function-style calls
        function_pattern = r"(?:function|tool):\s*(\w+)\s*\(\s*([\s\S]*?)\s*\)"
        function_matches = re.findall(function_pattern, text, re.IGNORECASE)

        for name, args_str in function_matches:
            try:
                # Try to parse arguments as JSON
                args = json.loads(f"{{{args_str}}}")
                tool_calls.append({"name": name, "arguments": args})
            except json.JSONDecodeError:
                # If JSON parsing fails, use raw string
                tool_calls.append({"name": name, "arguments": args_str})

        # Pattern 2: JSON-style tool calls
        json_pattern = r"```json\s*([\s\S]*?)\s*```"
        json_matches = re.findall(json_pattern, text)

        for json_str in json_matches:
            try:
                data = json.loads(json_str)
                if isinstance(data, dict) and "name" in data and "arguments" in data:
                    tool_calls.append(data)
                elif isinstance(data, list):
                    for item in data:
                        if (
                            isinstance(item, dict)
                            and "name" in item
                            and "arguments" in item
                        ):
                            tool_calls.append(item)
            except json.JSONDecodeError:
                pass
    except Exception as e:
        logger.error(f"Error parsing tool calls: {e}")
        # Return empty list on parsing error

    return tool_calls


def patch_llm_class():
    """
    Properly patch the LLM class with bound methods.
    This ensures methods are bound to instances and receive 'self' automatically.
    """
    from app.llm import LLM

    # For async methods, we need to modify how they're patched
    # The issue is that we need to ensure the methods are bound to instances
    # We'll use a monkey patching approach that works with async methods
    # Create a wrapper for the LLM.__init__ method to bind our methods to each instance
    original_init = LLM.__init__

    def patched_init(self, *args, **kwargs):
        # Call the original __init__
        original_init(self, *args, **kwargs)

        # Bind our methods to this specific instance using types.MethodType
        # This ensures 'self' is correctly passed when the methods are called
        self.ask_tool = types.MethodType(ask_tool, self)
        self._parse_tool_calls = types.MethodType(_parse_tool_calls, self)

        logger.info("Instance methods bound successfully")

    # Replace the __init__ method with our patched version
    LLM.__init__ = patched_init

    # Verify the patch was applied correctly
    logger.info(f"LLM.__init__ patched to bind instance methods")
    logger.info(f"ask_tool and _parse_tool_calls will be bound to each LLM instance")


# Make the patch_llm_class function available for import
__all__ = ["patch_llm_class", "ask_tool", "_parse_tool_calls"]

# Execute the patch immediately when this module is imported
patch_llm_class()
