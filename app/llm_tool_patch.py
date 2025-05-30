import asyncio
import json
from typing import Any, Dict, List, Optional, Union

from app.config import config
from app.exceptions import TokenLimitExceeded
from app.logger import logger
from app.schema import Message, ToolCall, ToolChoice

async def ask_tool(self, 
                  messages: List[Union[Message, Dict[str, Any]]],
                  system_msgs: Optional[List[Union[Message, Dict[str, Any]]]] = None,
                  tools: Optional[List[Dict[str, Any]]] = None,
                  tool_choice: Optional[Union[str, Dict[str, Any]]] = None,
                  temperature: Optional[float] = None,
                  timeout: int = 120,
                  **kwargs) -> Dict[str, Any]:
    """
    Send a request to the model with tool definitions and get a response with tool calls.
    
    Args:
        messages: List of messages to send to the model
        system_msgs: Optional system messages to prepend
        tools: List of tool definitions
        tool_choice: Tool choice configuration (auto, required, or specific tool)
        temperature: Temperature for sampling (0.0 to 1.0)
        timeout: Timeout in seconds for the request
        **kwargs: Additional arguments to pass to the model
        
    Returns:
        The model's response with tool calls
        
    Raises:
        TokenLimitExceeded: If the input exceeds the model's token limit
        Exception: For other unexpected errors
    """
    try:
        # Format messages
        if system_msgs:
            system_msgs = self.format_messages(system_msgs)
            messages = system_msgs + self.format_messages(messages)
        else:
            messages = self.format_messages(messages)
        
        # Calculate input token count
        input_tokens = self.count_message_tokens(messages)
        
        # Check if token limits are exceeded
        if not self.check_token_limit(input_tokens):
            error_message = self.get_limit_error_message(input_tokens)
            # Raise a special exception that won't be retried
            raise TokenLimitExceeded(error_message)
        
        # Format prompt for tool calling
        prompt = self._format_prompt_for_llama(messages)
        
        # Set temperature
        temp = temperature if temperature is not None else self.temperature
        
        # Apply safe max tokens limit
        safe_max_tokens = min(self.max_tokens, self.MAX_ALLOWED_OUTPUT_TOKENS)
        
        # Create a task for model completion with timeout and run in thread pool
        try:
            # Prepare tool definitions for the prompt
            tool_definitions = ""
            if tools:
                tool_definitions = "Available tools:\n"
                for tool in tools:
                    tool_definitions += f"- {tool['name']}: {tool['description']}\n"
                    if 'parameters' in tool:
                        tool_definitions += f"  Parameters: {json.dumps(tool['parameters'])}\n"
                tool_definitions += "\n"
            
            # Add tool instructions based on tool_choice
            tool_instructions = ""
            if tool_choice == "required":
                tool_instructions = "You MUST use one of the available tools to respond.\n"
            elif tool_choice == "auto":
                tool_instructions = "Use tools when appropriate to complete the task.\n"
            
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
                        **kwargs
                    )
                ),
                timeout=timeout
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
                    "total_tokens": prompt_tokens + completion_tokens
                }
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
                    "total_tokens": self.count_tokens(enhanced_prompt)
                }
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
    
    # Look for tool call patterns in the text
    # Pattern 1: Function-style calls
    function_pattern = r"(?:function|tool):\s*(\w+)\s*\(\s*([\s\S]*?)\s*\)"
    function_matches = re.findall(function_pattern, text, re.IGNORECASE)
    
    for name, args_str in function_matches:
        try:
            # Try to parse arguments as JSON
            args = json.loads(f"{{{args_str}}}")
            tool_calls.append({
                "name": name,
                "arguments": args
            })
        except json.JSONDecodeError:
            # If JSON parsing fails, use raw string
            tool_calls.append({
                "name": name,
                "arguments": args_str
            })
    
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
                    if isinstance(item, dict) and "name" in item and "arguments" in item:
                        tool_calls.append(item)
        except json.JSONDecodeError:
            pass
    
    return tool_calls

# Monkey patch the LLM class to add the ask_tool method
from app.llm import LLM
import re

# Add the methods to the LLM class
LLM.ask_tool = ask_tool
LLM._parse_tool_calls = _parse_tool_calls
