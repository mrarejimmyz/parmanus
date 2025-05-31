import asyncio
import json
import logging
import re
import types
from typing import Dict, List, Any, Optional

logger = logging.getLogger(__name__)

class TokenLimitExceeded(Exception):
    """Exception raised when token limit is exceeded."""
    pass

def ask_tool(
    self,
    prompt: str,
    tools: Optional[List[Dict[str, Any]]] = None,
    tool_choice: str = "auto",
    temp: float = 0.0,
    timeout: int = 60,
    **kwargs
) -> Dict[str, Any]:
    """
    Ask the model to use tools based on the prompt.
    
    Args:
        prompt: The prompt to send to the model
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
            # Prepare tool definitions for the prompt
            tool_definitions = ""
            if tools:
                tool_definitions = "Available tools:\n"
                for tool in tools:
                    # Safely access tool properties with fallbacks
                    tool_name = tool.get('name', 'unnamed_tool')
                    tool_description = tool.get('description', 'No description available')
                    tool_definitions += f"- {tool_name}: {tool_description}\n"
                    
                    # Safely handle parameters if present
                    if 'parameters' in tool:
                        try:
                            params_json = json.dumps(tool['parameters'])
                            tool_definitions += f"  Parameters: {params_json}\n"
                        except (TypeError, ValueError) as e:
                            logger.warning(f"Failed to serialize parameters for tool {tool_name}: {e}")
                            tool_definitions += f"  Parameters: [Error: Could not serialize parameters]\n"
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
    
    try:
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
    
    # Use types.MethodType to properly bind methods to the class
    # This ensures 'self' is passed correctly when methods are called
    
    # First, check if methods are already patched to avoid duplicate patching
    if not hasattr(LLM, 'ask_tool') or not isinstance(LLM.ask_tool, types.FunctionType):
        LLM.ask_tool = ask_tool
        logger.info("Added ask_tool method to LLM class")
    
    if not hasattr(LLM, '_parse_tool_calls') or not isinstance(LLM._parse_tool_calls, types.FunctionType):
        LLM._parse_tool_calls = _parse_tool_calls
        logger.info("Added _parse_tool_calls method to LLM class")
    
    # Verify the patch was applied correctly
    logger.info(f"LLM class now has ask_tool: {hasattr(LLM, 'ask_tool')}")
    logger.info(f"LLM class now has _parse_tool_calls: {hasattr(LLM, '_parse_tool_calls')}")

# Make the patch_llm_class function available for import
__all__ = ['patch_llm_class', 'ask_tool', '_parse_tool_calls']

# Execute the patch immediately when this module is imported
patch_llm_class()
