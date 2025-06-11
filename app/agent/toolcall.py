import asyncio
import json
from typing import Any, List, Optional, Union

from pydantic import Field

from app.agent.react import ReActAgent
from app.exceptions import TokenLimitExceeded
from app.logger import logger
from app.prompt.toolcall import NEXT_STEP_PROMPT, SYSTEM_PROMPT
from app.schema import TOOL_CHOICE_TYPE, AgentState, Message, ToolCall, ToolChoice
from app.tool import CreateChatCompletion, Terminate, ToolCollection

TOOL_CALL_REQUIRED = "Tool calls required but none provided"


class ToolCallAgent(ReActAgent):
    """Base agent class for handling tool/function calls with enhanced abstraction"""

    name: str = "toolcall"
    description: str = "an agent that can execute tool calls."

    system_prompt: str = SYSTEM_PROMPT
    next_step_prompt: str = NEXT_STEP_PROMPT

    available_tools: ToolCollection = ToolCollection(
        CreateChatCompletion(), Terminate()
    )
    tool_choices: TOOL_CHOICE_TYPE = ToolChoice.AUTO  # type: ignore
    special_tool_names: List[str] = Field(default_factory=lambda: [Terminate().name])

    tool_calls: List[ToolCall] = Field(default_factory=list)
    _current_base64_image: Optional[str] = None

    max_steps: int = 30
    max_observe: Optional[Union[int, bool]] = None

    async def think(self) -> bool:
        """Process current state and decide next actions using tools"""
        if self.next_step_prompt:
            user_msg = Message.user_message(self.next_step_prompt)
            self.messages += [user_msg]

        try:
            # Get response with tool options
            response = await self.llm.ask_tool(
                messages=self.messages,
                system_msgs=(
                    [Message.system_message(self.system_prompt)]
                    if self.system_prompt
                    else None
                ),
                tools=self.available_tools.to_params(),
                tool_choice=self.tool_choices,
            )
        except ValueError:
            raise
        except Exception as e:
            # Check if this is a RetryError containing TokenLimitExceeded
            if hasattr(e, "__cause__") and isinstance(e.__cause__, TokenLimitExceeded):
                token_limit_error = e.__cause__
                logger.error(
                    f"🚨 Token limit error (from RetryError): {token_limit_error}"
                )
                self.memory.add_message(
                    Message.assistant_message(
                        f"Maximum token limit reached, cannot continue execution: {str(token_limit_error)}"
                    )
                )
                self.state = AgentState.FINISHED
                return False
            raise

        if not response:
            logger.error("No response received from LLM")
            return False

        self.tool_calls = []
        content = ""

        if isinstance(response, dict):
            self.tool_calls = response.get("tool_calls", [])
            content = response.get("content", "")
        else:
            self.tool_calls = getattr(response, "tool_calls", [])
            content = getattr(response, "content", "")

        # Log response info
        logger.info(f"✨ {self.name}'s thoughts: {content}")
        logger.info(f"🛠️ {self.name} selected {len(self.tool_calls)} tools to use")
        if self.tool_calls:
            # Handle both dict and object formats for tool calls
            tool_names = []
            for call in self.tool_calls:
                if isinstance(call, dict):
                    # Dict format
                    if "function" in call and "name" in call["function"]:
                        tool_names.append(call["function"]["name"])
                    else:
                        tool_names.append(str(call))
                elif hasattr(call, "function") and hasattr(call.function, "name"):
                    # Object format
                    tool_names.append(call.function.name)
                else:
                    tool_names.append(str(call))

            logger.info(f"🧰 Tools being prepared: {tool_names}")

            # Log first tool arguments (handle both formats)
            if self.tool_calls:
                first_call = self.tool_calls[0]
                if isinstance(first_call, dict) and "function" in first_call:
                    args = first_call["function"].get("arguments", "{}")
                elif hasattr(first_call, "function") and hasattr(
                    first_call.function, "arguments"
                ):
                    args = first_call.function.arguments
                else:
                    args = str(first_call)
                logger.info(f"🔧 Tool arguments: {args}")

        try:
            # Handle different tool_choices modes
            if self.tool_choices == ToolChoice.NONE:
                if self.tool_calls:
                    logger.warning(
                        f"🤔 Hmm, {self.name} tried to use tools when they weren't available!"
                    )
                if content:
                    self.memory.add_message(Message.assistant_message(content))
                    return True
                return False

            # Create and add assistant message
            assistant_msg = (
                Message.from_tool_calls(content=content, tool_calls=self.tool_calls)
                if self.tool_calls
                else Message.assistant_message(content)
            )
            self.memory.add_message(assistant_msg)

            if self.tool_choices == ToolChoice.REQUIRED and not self.tool_calls:
                return True  # Will be handled in act()

            # For 'auto' mode, continue with content if no commands but content exists
            if self.tool_choices == ToolChoice.AUTO and not self.tool_calls and content:
                return True

            # Don't return True for ask_human loops - check for repetitive patterns
            if self.tool_calls and len(self.tool_calls) == 1:
                tool_call = self.tool_calls[0]
                tool_name = ""
                if isinstance(tool_call, dict) and "function" in tool_call:
                    tool_name = tool_call["function"].get("name", "")
                elif hasattr(tool_call, "function") and hasattr(
                    tool_call.function, "name"
                ):
                    tool_name = tool_call.function.name

                # If it's ask_human and we have recent messages, check for repetition
                if tool_name == "ask_human":
                    recent_messages = (
                        self.memory.messages[-5:]
                        if hasattr(self, "memory") and self.memory.messages
                        else []
                    )
                    ask_human_count = sum(
                        1
                        for msg in recent_messages
                        if hasattr(msg, "tool_calls")
                        and msg.tool_calls
                        and any(
                            tc.get("function", {}).get("name") == "ask_human"
                            for tc in (
                                msg.tool_calls
                                if isinstance(msg.tool_calls, list)
                                else []
                            )
                        )
                    )

                    if ask_human_count >= 2:
                        logger.warning(
                            "🔄 Detected ask_human loop - providing direct response instead"
                        )
                        self.memory.add_message(
                            Message.assistant_message(
                                "I understand you want me to search for trending AI safety research. Let me use the browser to search for this information instead of asking more questions."
                            )
                        )
                        # Clear tool calls and return True to provide response
                        self.tool_calls = []
                        return True

            return bool(self.tool_calls)
        except Exception as e:
            logger.error(f"🚨 Oops! The {self.name}'s thinking process hit a snag: {e}")
            self.memory.add_message(
                Message.assistant_message(
                    f"Error encountered while processing: {str(e)}"
                )
            )
            return False

    async def act(self) -> str:
        """Execute tool calls and handle their results"""
        if not self.tool_calls:
            if self.tool_choices == ToolChoice.REQUIRED:
                logger.error("Tool calls required but none provided")
                raise ValueError(TOOL_CALL_REQUIRED)

            # Return last message content if no tool calls
            last_message = self.messages[-1] if self.messages else None
            if last_message:
                if isinstance(last_message, dict):
                    return last_message.get(
                        "content", "No content or commands to execute"
                    )
                else:
                    return getattr(
                        last_message, "content", "No content or commands to execute"
                    )
            return "No content or commands to execute"

        results = []
        for command in self.tool_calls:
            # Reset base64_image for each tool call
            self._current_base64_image = None

            result = await self.execute_tool(command)

            if self.max_observe:
                result = result[: self.max_observe]

            # Handle both dict and object formats for logging
            if isinstance(command, dict) and "function" in command:
                tool_name = command["function"].get("name", "unknown")
                tool_id = command.get("id", "unknown")
            elif hasattr(command, "function") and hasattr(command.function, "name"):
                tool_name = command.function.name
                tool_id = command.id
            else:
                tool_name = str(command)
                tool_id = "unknown"

            logger.info(
                f"🎯 Tool '{tool_name}' completed its mission! Result: {result}"
            )

            # Add tool response to memory
            tool_msg = Message.tool_message(
                content=result,
                tool_call_id=tool_id,
                name=tool_name,
                base64_image=self._current_base64_image,
            )
            self.memory.add_message(tool_msg)
            results.append(result)

        return "\n\n".join(results)

    async def execute_tool(self, command: ToolCall) -> str:
        """Execute a single tool call with robust error handling"""
        # Handle both dict and object formats with better error handling
        name = None
        arguments = "{}"

        try:
            if isinstance(command, dict):
                # Dict format: {"id": "...", "type": "function", "function": {"name": "...", "arguments": "..."}}
                if "function" in command and isinstance(command["function"], dict):
                    name = command["function"].get("name")
                    arguments = command["function"].get("arguments", "{}")
                elif "name" in command:
                    # Direct format: {"name": "...", "arguments": "..."}
                    name = command.get("name")
                    arguments = command.get("arguments", "{}")
                else:
                    logger.error(f"Invalid dict command format: {command}")
                    return (
                        "Error: Invalid dict command format - missing function or name"
                    )

            elif hasattr(command, "function"):
                # Object format with function attribute
                if hasattr(command.function, "name"):
                    name = command.function.name
                    arguments = getattr(command.function, "arguments", "{}")
                else:
                    logger.error(
                        f"Invalid object command format: command.function has no name"
                    )
                    return "Error: Invalid object command format - function has no name"

            elif hasattr(command, "name"):
                # Direct object format
                name = command.name
                arguments = getattr(command, "arguments", "{}")
            else:
                logger.error(
                    f"Unrecognized command format: {type(command)} - {command}"
                )
                return f"Error: Unrecognized command format: {type(command)}"

            if not name or not isinstance(name, str):
                logger.error(f"Tool name is invalid: '{name}' from command: {command}")
                return f"Error: Invalid tool name: '{name}'"

        except Exception as e:
            logger.error(f"Error parsing tool command: {e}, command: {command}")
            return f"Error parsing tool command: {str(e)}"

        # Now execute the tool
        if name not in self.available_tools.tool_map:
            available_tools = list(self.available_tools.tool_map.keys())
            logger.error(f"Unknown tool '{name}'. Available tools: {available_tools}")
            return f"Error: Unknown tool '{name}'. Available tools: {available_tools}"

        try:
            # Parse arguments safely
            if isinstance(arguments, str):
                try:
                    args = json.loads(arguments) if arguments.strip() else {}
                except json.JSONDecodeError as e:
                    logger.error(
                        f"Invalid JSON arguments for {name}: {arguments}. Error: {e}"
                    )
                    return f"Error: Invalid JSON arguments for {name}: {arguments}"
            elif isinstance(arguments, dict):
                args = arguments
            else:
                logger.error(
                    f"Arguments must be string or dict, got {type(arguments)}: {arguments}"
                )
                return f"Error: Invalid arguments type {type(arguments)}"

            # Execute the tool
            logger.info(f"🔧 Activating tool: '{name}' with args: {args}")
            result = await self.available_tools.execute(name=name, tool_input=args)

            # Handle special tools
            await self._handle_special_tool(name=name, result=result)

            # Check if result is a ToolResult with base64_image
            if hasattr(result, "base64_image") and result.base64_image:
                # Store the base64_image for later use in tool_message
                self._current_base64_image = result.base64_image

            # Format result for display (standard case)
            observation = (
                f"✅ Tool `{name}` executed successfully:\n{str(result)}"
                if result
                else f"✅ Tool `{name}` completed with no output"
            )

            return observation

        except json.JSONDecodeError:
            error_msg = f"Error parsing arguments for {name}: Invalid JSON format"
            logger.error(f"📝 Invalid JSON arguments for '{name}': {arguments}")
            return f"Error: {error_msg}"
        except Exception as e:
            error_msg = f"⚠️ Tool '{name}' encountered a problem: {str(e)}"
            logger.exception(error_msg)
            return f"Error: {error_msg}"

    async def _handle_special_tool(self, name: str, result: Any, **kwargs):
        """Handle special tool execution and state changes"""
        if not self._is_special_tool(name):
            return

        if self._should_finish_execution(name=name, result=result, **kwargs):
            # Set agent state to finished
            logger.info(f"🏁 Special tool '{name}' has completed the task!")
            self.state = AgentState.FINISHED

    @staticmethod
    def _should_finish_execution(**kwargs) -> bool:
        """Determine if tool execution should finish the agent"""
        return True

    def _is_special_tool(self, name: str) -> bool:
        """Check if tool name is in special tools list"""
        return name.lower() in [n.lower() for n in self.special_tool_names]

    async def cleanup(self):
        """Clean up resources used by the agent's tools."""
        logger.info(f"🧹 Cleaning up resources for agent '{self.name}'...")
        for tool_name, tool_instance in self.available_tools.tool_map.items():
            if hasattr(tool_instance, "cleanup") and asyncio.iscoroutinefunction(
                tool_instance.cleanup
            ):
                try:
                    logger.debug(f"🧼 Cleaning up tool: {tool_name}")
                    await tool_instance.cleanup()
                except Exception as e:
                    logger.error(
                        f"🚨 Error cleaning up tool '{tool_name}': {e}", exc_info=True
                    )
        logger.info(f"✨ Cleanup complete for agent '{self.name}'.")

    async def run(self, request: Optional[str] = None) -> str:
        """Run the agent with cleanup when done."""
        try:
            return await super().run(request)
        finally:
            await self.cleanup()

    def ask_tool(self, query: str, tool_name: str = None, max_retries: int = 3) -> dict:
        """
        Ask a tool to perform an action and return the result
        """
        # Default tool selection if none specified
        if tool_name is None:
            tool_name = self._default_tool

        # Input validation
        if not isinstance(query, str):
            raise ValueError("Query must be a string")

        # Get tool instance
        tool = self._get_tool_instance(tool_name)
        if not tool:
            raise ValueError(f"Tool {tool_name} not found")

        # Format tool call object
        tool_call = {"name": tool_name, "query": query, "args": {}, "metadata": {}}

        # Execute tool with retries
        for attempt in range(max_retries):
            try:
                result = tool.execute(tool_call)
                if result and isinstance(result, dict):
                    return result
                self.logger.warning(f"Tool returned invalid result: {result}")
            except Exception as e:
                self.logger.warning(f"Tool call attempt {attempt + 1} failed: {str(e)}")
                if attempt < max_retries - 1:
                    continue
                else:
                    self.logger.error("All retry attempts failed")
                    raise

        return {"error": "Tool call failed after all retries"}

    def _get_tool_instance(self, tool_name: str):
        """Get a tool instance by name"""
        # Ensure tool name is properly formatted
        tool_name = str(tool_name).strip().lower()

        # Check registered tools
        if tool_name in self._tools:
            return self._tools[tool_name]

        # Try loading tool if not found
        try:
            tool = self._load_tool(tool_name)
            self._tools[tool_name] = tool
            return tool
        except Exception as e:
            self.logger.error(f"Failed to load tool {tool_name}: {str(e)}")
            return None
