import asyncio
import json
import logging
import re
import time
import types
from typing import Any, Dict, List, Optional, Union

from app.schema import TOOL_CHOICE_TYPE, Message, ToolChoice

logger = logging.getLogger(__name__)


class TokenLimitExceeded(Exception):
    """Exception raised when token limit is exceeded."""
    pass


class ModelTimeoutError(Exception):
    """Exception raised when model completion times out."""
    pass


async def ask_tool(
    self,
    messages: List[Union[Message, Dict[str, Any]]],
    system_msgs: Optional[List[Union[Message, Dict[str, Any]]]] = None,
    tools: Optional[List[Dict[str, Any]]] = None,
    tool_choice: TOOL_CHOICE_TYPE = ToolChoice.AUTO,
    temp: float = 0.0,
    timeout: int = None,
    max_retries: int = 2,
    **kwargs,
) -> Dict[str, Any]:
    """
    Ask the model to use tools based on the messages with optimized timeout handling.

    Args:
        messages: List of messages to send to the model
        system_msgs: Optional system messages to prepend
        tools: List of tool definitions
        tool_choice: Whether tool use is "auto", "required", or "none"
        temp: Temperature for sampling
        timeout: Timeout in seconds (adaptive if None)
        max_retries: Maximum number of retries on timeout
        **kwargs: Additional arguments to pass to the model

    Returns:
        Dictionary containing the model's response and any tool calls
    """
    # Adaptive timeout based on message length and complexity
    if timeout is None:
        timeout = _calculate_adaptive_timeout(messages, tools)
    
    start_time = time.time()
    last_exception = None
    
    for attempt in range(max_retries + 1):
        try:
            # Apply safe max tokens limit
            safe_max_tokens = min(self.max_tokens, self.MAX_ALLOWED_OUTPUT_TOKENS)

            # Format messages for the model
            if system_msgs:
                system_msgs = self.format_messages(system_msgs)
                formatted_messages = system_msgs + self.format_messages(messages)
            else:
                formatted_messages = self.format_messages(messages)

            # Prepare tool definitions for the prompt (optimized)
            tool_definitions = _format_tool_definitions(tools)

            # Add tool instructions based on tool_choice
            tool_instructions = _get_tool_instructions(tool_choice)

            # Format messages into a prompt string
            prompt = self._format_prompt_for_llama(formatted_messages)

            # Enhance prompt with tool information
            enhanced_prompt = f"{prompt}\n\n{tool_definitions}{tool_instructions}"

            # Log attempt info (reduced verbosity)
            if attempt > 0:
                logger.warning(f"Model completion retry {attempt}/{max_retries} (timeout: {timeout}s)")
            else:
                logger.debug(f"Model completion attempt (timeout: {timeout}s)")

            # Run model inference in thread pool with timeout
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

            # Log successful completion
            elapsed = time.time() - start_time
            logger.debug(f"Model completion successful in {elapsed:.2f}s")

            # Return structured response with tool calls
            return {
                "content": completion_text,
                "tool_calls": tool_calls,
                "usage": {
                    "prompt_tokens": prompt_tokens,
                    "completion_tokens": completion_tokens,
                    "total_tokens": prompt_tokens + completion_tokens,
                },
                "elapsed_time": elapsed,
                "attempts": attempt + 1,
            }

        except asyncio.TimeoutError as e:
            last_exception = e
            elapsed = time.time() - start_time
            
            if attempt < max_retries:
                # Increase timeout for retry
                timeout = min(timeout * 1.5, 180)  # Cap at 3 minutes
                logger.warning(
                    f"Model completion timed out after {elapsed:.1f}s, "
                    f"retrying with {timeout}s timeout (attempt {attempt + 1}/{max_retries})"
                )
                continue
            else:
                logger.error(
                    f"Model completion failed after {max_retries + 1} attempts "
                    f"(total time: {elapsed:.1f}s)"
                )
                # Return partial result with timeout indication
                return {
                    "content": f"[Response incomplete due to timeout after {elapsed:.1f}s and {max_retries + 1} attempts]",
                    "tool_calls": [],
                    "usage": {
                        "prompt_tokens": self.count_tokens(enhanced_prompt) if 'enhanced_prompt' in locals() else 0,
                        "completion_tokens": 0,
                        "total_tokens": self.count_tokens(enhanced_prompt) if 'enhanced_prompt' in locals() else 0,
                    },
                    "elapsed_time": elapsed,
                    "attempts": attempt + 1,
                    "error": "timeout",
                }

        except Exception as e:
            last_exception = e
            elapsed = time.time() - start_time
            
            if attempt < max_retries and not isinstance(e, TokenLimitExceeded):
                logger.warning(f"Model completion error on attempt {attempt + 1}: {e}")
                continue
            else:
                logger.error(f"Model completion failed permanently: {e}")
                raise

    # This should not be reached, but just in case
    raise last_exception or RuntimeError("Unexpected completion failure")


def _calculate_adaptive_timeout(messages: List, tools: Optional[List] = None) -> int:
    """Calculate adaptive timeout based on input complexity."""
    base_timeout = 30  # Base timeout in seconds
    
    # Calculate message complexity
    total_chars = sum(len(str(msg)) for msg in messages)
    message_factor = min(total_chars / 1000, 5)  # Max 5x multiplier for messages
    
    # Calculate tool complexity
    tool_factor = 0
    if tools:
        tool_factor = min(len(tools) * 0.5, 3)  # Max 3x multiplier for tools
    
    # Calculate final timeout
    timeout = int(base_timeout + (message_factor * 10) + (tool_factor * 5))
    return min(timeout, 120)  # Cap at 2 minutes


def _format_tool_definitions(tools: Optional[List[Dict[str, Any]]]) -> str:
    """Format tool definitions with optimized output."""
    if not tools:
        return ""
    
    tool_definitions = "Available tools:\n"
    for tool in tools:
        tool_name = tool.get("name", "unnamed_tool")
        tool_description = tool.get("description", "No description available")
        tool_definitions += f"- {tool_name}: {tool_description}\n"

        # Safely handle parameters if present (simplified)
        if "parameters" in tool:
            try:
                # Simplified parameter representation
                params = tool["parameters"]
                if isinstance(params, dict) and "properties" in params:
                    param_names = list(params["properties"].keys())
                    tool_definitions += f"  Parameters: {', '.join(param_names[:5])}\n"  # Limit to first 5
                else:
                    tool_definitions += f"  Parameters: Available\n"
            except Exception:
                tool_definitions += f"  Parameters: Available\n"
    
    return tool_definitions + "\n"


def _get_tool_instructions(tool_choice: TOOL_CHOICE_TYPE) -> str:
    """Get tool instructions based on choice."""
    if tool_choice == ToolChoice.REQUIRED:
        return "You MUST use one of the available tools to respond.\n"
    elif tool_choice == ToolChoice.AUTO:
        return "Use tools when appropriate to complete the task.\n"
    return ""


def _parse_tool_calls(self, text: str) -> List[Dict[str, Any]]:
    """
    Parse tool calls from completion text with improved patterns.

    Args:
        text: Completion text from the model

    Returns:
        List of parsed tool calls
    """
    tool_calls = []

    try:
        # Pattern 1: Function-style calls (improved)
        function_pattern = r"(?:function|tool|call):\s*(\w+)\s*\(\s*([\s\S]*?)\s*\)"
        function_matches = re.findall(function_pattern, text, re.IGNORECASE)

        for name, args_str in function_matches:
            try:
                # Try to parse arguments as JSON
                if args_str.strip():
                    # Handle both object and key-value formats
                    if not args_str.strip().startswith('{'):
                        args_str = f"{{{args_str}}}"
                    args = json.loads(args_str)
                else:
                    args = {}
                tool_calls.append({"name": name, "arguments": args})
            except json.JSONDecodeError:
                # If JSON parsing fails, try simple key=value parsing
                args = _parse_simple_args(args_str)
                tool_calls.append({"name": name, "arguments": args})

        # Pattern 2: JSON-style tool calls (improved)
        json_pattern = r"```(?:json)?\s*([\s\S]*?)\s*```"
        json_matches = re.findall(json_pattern, text)

        for json_str in json_matches:
            try:
                data = json.loads(json_str)
                if isinstance(data, dict) and "name" in data:
                    if "arguments" not in data:
                        data["arguments"] = {}
                    tool_calls.append(data)
                elif isinstance(data, list):
                    for item in data:
                        if isinstance(item, dict) and "name" in item:
                            if "arguments" not in item:
                                item["arguments"] = {}
                            tool_calls.append(item)
            except json.JSONDecodeError:
                continue

    except Exception as e:
        logger.debug(f"Error parsing tool calls: {e}")

    return tool_calls


def _parse_simple_args(args_str: str) -> Dict[str, Any]:
    """Parse simple key=value arguments."""
    args = {}
    try:
        # Split by comma and parse key=value pairs
        for pair in args_str.split(','):
            if '=' in pair:
                key, value = pair.split('=', 1)
                key = key.strip().strip('"\'')
                value = value.strip().strip('"\'')
                
                # Try to convert to appropriate type
                if value.lower() in ['true', 'false']:
                    value = value.lower() == 'true'
                elif value.isdigit():
                    value = int(value)
                elif value.replace('.', '').isdigit():
                    value = float(value)
                
                args[key] = value
    except Exception:
        pass
    
    return args


def patch_llm_class():
    """
    Patch the LLM class with optimized methods.
    """
    from app.llm import LLM

    original_init = LLM.__init__

    def patched_init(self, *args, **kwargs):
        # Call the original __init__
        original_init(self, *args, **kwargs)

        # Bind our optimized methods to this specific instance
        self.ask_tool = types.MethodType(ask_tool, self)
        self._parse_tool_calls = types.MethodType(_parse_tool_calls, self)

        logger.debug("Optimized LLM methods bound successfully")

    # Replace the __init__ method with our patched version
    LLM.__init__ = patched_init

    logger.info("LLM class patched with optimized timeout handling")


# Make functions available for import
__all__ = ["patch_llm_class", "ask_tool", "_parse_tool_calls", "ModelTimeoutError"]

# Execute the patch immediately when this module is imported
patch_llm_class()

