import asyncio
import json
from typing import Any, List, Optional, Union

from pydantic import Field

from app.agent.react import ReActAgent
from app.exceptions import AgentTaskComplete, TokenLimitExceeded
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
        """Process current state and decide next actions with tools"""
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
        except AgentTaskComplete as e:
            # Handle successful task completion
            logger.info(f"🎉 Task completed successfully: {e.message}")
            self.memory.add_message(
                Message.system_message(f"Task completed: {e.message}")
            )
            self.state = AgentState.FINISHED
            raise  # Re-raise to ensure proper propagation
        except TokenLimitExceeded as e:
            # Handle token limit exceeded
            logger.error(f"🚨 Token limit exceeded: {e}")
            self.memory.add_message(
                Message.assistant_message(
                    f"Maximum token limit reached, cannot continue execution: {str(e)}"
                )
            )
            self.state = AgentState.FINISHED
            return False
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

        self.tool_calls = tool_calls = (
            response.get("tool_calls")
            if response and isinstance(response, dict)
            else (
                response.tool_calls
                if response and hasattr(response, "tool_calls")
                else []
            )
        )
        content = (
            response.get("content")
            if response and isinstance(response, dict)
            else response.content if response and hasattr(response, "content") else ""
        )

        # Log response info
        logger.info(f"✨ {self.name}'s thoughts: {content}")
        logger.info(
            f"🛠️ {self.name} selected {len(tool_calls) if tool_calls else 0} tools to use"
        )
        if tool_calls:
            # Handle both dict and object formats for tool calls
            tool_names = []
            for call in tool_calls:
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
            if tool_calls:
                first_call = tool_calls[0]
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
            if response is None:
                raise RuntimeError("No response received from the LLM")

            # Handle different tool_choices modes
            if self.tool_choices == ToolChoice.NONE:
                if tool_calls:
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
            if self.tool_choices == ToolChoice.AUTO and not self.tool_calls:
                return bool(content)

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
                raise ValueError(TOOL_CALL_REQUIRED)

            # Return last message content if no tool calls
            return self.messages[-1].content or "No content or commands to execute"

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
                tool_id = getattr(command, "id", "unknown")
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
        try:
            if (
                not command
                or not hasattr(command, "function")
                or not command.function
                or not command.function.name
            ):
                return "Error: Invalid command format"

            name = command.function.name
            if name not in self.available_tools.tool_map:
                return f"Error: Unknown tool '{name}'"

            try:
                arguments = (
                    json.loads(command.function.arguments)
                    if command.function.arguments
                    else {}
                )

                # Execute the tool
                logger.info(f"🔧 Activating tool: '{name}'...")
                result = await self.available_tools.execute(
                    name=name, tool_input=arguments
                )

                # Handle special tools
                await self._handle_special_tool(name=name, result=result)

                # Check if result is a ToolResult with base64_image
                if hasattr(result, "base64_image") and result.base64_image:
                    self._current_base64_image = result.base64_image

                # Format result for display
                observation = (
                    f"Observed output of cmd `{name}` executed:\n{str(result)}"
                    if result
                    else f"Cmd `{name}` completed with no output"
                )

                return observation

            except json.JSONDecodeError:
                error_msg = f"Error parsing arguments for {name}: Invalid JSON format"
                logger.error(
                    f"📝 Invalid JSON arguments for '{name}': {command.function.arguments}"
                )
                return f"Error: {error_msg}"
            except AgentTaskComplete as e:
                # Propagate task completion signal
                logger.info(
                    f"🎉 Task completion signaled during tool execution: {e.message}"
                )
                raise
            except Exception as e:
                error_msg = f"⚠️ Tool '{name}' encountered a problem: {str(e)}"
                logger.exception(error_msg)
                return f"Error: {error_msg}"

        except Exception as e:
            logger.error(f"Error executing tool: {str(e)}")
            return f"Error: Invalid command format - {str(e)}"

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
        cleanup_called = False
        try:
            result = await super().run(request)
            return result
        finally:
            if not cleanup_called:
                cleanup_called = True
                await self.cleanup()
