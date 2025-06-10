from abc import ABC, abstractmethod
from typing import Annotated, Any, Dict, List, Optional, Union

from pydantic import BaseModel, Field, model_validator
from typing_extensions import TypeGuard

from app.agent.browser import BrowserContextHelper  # Add this import
from app.agent.toolcall import ToolCallAgent
from app.config import config
from app.llm_optimized import LLMOptimized
from app.logger import logger
from app.memory import Memory
from app.prompt.manus import NEXT_STEP_PROMPT, SYSTEM_PROMPT
from app.schema import AgentState
from app.tool import Terminate, ToolCollection
from app.tool.ask_human import AskHuman
from app.tool.browser_use_tool import BrowserUseTool
from app.tool.mcp import MCPClients, MCPClientTool
from app.tool.python_execute import PythonExecute
from app.tool.str_replace_editor import StrReplaceEditor


def validate_llm_instance(v) -> bool:
    """Validate that the LLM is a supported instance."""
    # Accept any LLM instance that has the basic required methods
    return (
        hasattr(v, "ask")
        or hasattr(v, "ask_tool")
        or hasattr(v, "chat")
        or hasattr(v, "complete")
    )


class Manus(ToolCallAgent):
    """A versatile general-purpose agent with support for both local and MCP tools."""

    name: str = "Manus"
    description: str = (
        "A versatile agent that can solve various tasks using multiple tools including MCP-based tools"
    )

    # Flag to control if browser features are enabled
    browser_enabled: bool = True

    system_prompt: str = SYSTEM_PROMPT.format(directory=config.workspace_root)
    next_step_prompt: str = NEXT_STEP_PROMPT

    max_observe: int = 10000
    max_steps: int = 20

    # MCP clients for remote tool access
    mcp_clients: MCPClients = Field(default_factory=MCPClients)

    # LLM field with proper type and validation - accept any LLM
    llm: Any = Field(
        default=None,
        description="The LLM instance to use for generating responses",
    )

    # Add general-purpose tools to the tool collection
    available_tools: ToolCollection = Field(
        default_factory=lambda: ToolCollection(
            PythonExecute(),
            BrowserUseTool(),
            StrReplaceEditor(),
            AskHuman(),
            Terminate(),
        )
    )

    @model_validator(mode="after")
    def validate_llm(self):
        """Validate and initialize LLM if needed."""
        from app.llm_factory import create_llm

        try:
            if self.llm is None:
                # Initialize with factory (supports both Ollama and llama.cpp)
                self.llm = create_llm(config.llm)
                logger.info(
                    f"Initialized new LLM instance with model: {getattr(self.llm, 'model', 'unknown')}"
                )
            elif not validate_llm_instance(self.llm):
                # Try to use the provided LLM anyway
                logger.warning(
                    f"LLM instance may not be fully compatible: {type(self.llm)}"
                )
            else:
                # Already valid instance
                logger.debug(f"Using existing LLM instance: {type(self.llm)}")
        except Exception as e:
            # Handle initialization errors
            logger.error(f"Error initializing LLM: {e}")
            # Use the provided LLM anyway if available
            if self.llm is None:
                raise ValueError(f"Failed to initialize LLM: {e}")

        return self

    special_tool_names: list[str] = Field(default_factory=lambda: [Terminate().name])
    browser_context_helper: Optional[BrowserContextHelper] = None

    # Track connected MCP servers
    connected_servers: Dict[str, str] = Field(
        default_factory=dict
    )  # server_id -> url/command
    _initialized: bool = False

    # Optional components
    gpu_manager: Optional[dict] = Field(default_factory=dict)
    mcp_server_urls: Dict[str, str] = Field(default_factory=dict)
    agent_state: Dict[str, Any] = Field(default_factory=dict)

    def _initialize_helper(self) -> None:
        """Initialize browser context helper if not already initialized."""
        if self.browser_context_helper is None:
            browser_tool = self.available_tools.get_tool(BrowserUseTool().name)
            if browser_tool:
                self.browser_context_helper = BrowserContextHelper(self)
                logger.info("Browser context helper initialized")
            else:
                logger.warning(
                    "BrowserUseTool not available, skipping browser context helper initialization"
                )

    @model_validator(mode="after")
    def initialize_components(self) -> "Manus":
        """Initialize basic components synchronously."""
        self._initialize_helper()
        return self

    @classmethod
    async def create(cls, **kwargs) -> "Manus":
        """Factory method to create and properly initialize a Manus instance."""
        instance = cls(**kwargs)
        await instance.initialize_mcp_servers()
        instance._initialized = True
        return instance

    async def initialize_mcp_servers(self) -> None:
        """Initialize connections to configured MCP servers."""
        for server_id, server_config in config.mcp_config.servers.items():
            try:
                if server_config.type == "sse":
                    if server_config.url:
                        await self.connect_mcp_server(server_config.url, server_id)
                        logger.info(
                            f"Connected to MCP server {server_id} at {server_config.url}"
                        )
                elif server_config.type == "stdio":
                    if server_config.command:
                        await self.connect_mcp_server(
                            server_config.command,
                            server_id,
                            use_stdio=True,
                            stdio_args=server_config.args,
                        )
                        logger.info(
                            f"Connected to MCP server {server_id} using command {server_config.command}"
                        )
            except Exception as e:
                logger.error(f"Failed to connect to MCP server {server_id}: {e}")

    async def connect_mcp_server(
        self,
        server_url: str,
        server_id: str = "",
        use_stdio: bool = False,
        stdio_args: List[str] = None,
    ) -> None:
        """Connect to an MCP server and add its tools."""
        if use_stdio:
            await self.mcp_clients.connect_stdio(
                server_url, stdio_args or [], server_id
            )
            self.connected_servers[server_id or server_url] = server_url
        else:
            await self.mcp_clients.connect_sse(server_url, server_id)
            self.connected_servers[server_id or server_url] = server_url

        # Update available tools with only the new tools from this server
        new_tools = [
            tool for tool in self.mcp_clients.tools if tool.server_id == server_id
        ]
        self.available_tools.add_tools(*new_tools)

    async def disconnect_mcp_server(self, server_id: str = "") -> None:
        """Disconnect from an MCP server and remove its tools."""
        await self.mcp_clients.disconnect(server_id)
        if server_id:
            self.connected_servers.pop(server_id, None)
        else:
            self.connected_servers.clear()

        # Rebuild available tools without the disconnected server's tools
        base_tools = [
            tool
            for tool in self.available_tools.tools
            if not isinstance(tool, MCPClientTool)
        ]
        self.available_tools = ToolCollection(*base_tools)
        self.available_tools.add_tools(*self.mcp_clients.tools)

    async def cleanup(self):
        """Clean up Manus agent resources."""
        try:
            # Clean up browser resources
            if self.browser_context_helper:
                await self.browser_context_helper.cleanup_browser()

            # Disconnect from all MCP servers only if we were initialized
            if self._initialized:
                await self.disconnect_mcp_server()
                self._initialized = False

            # Clean up LLM resources if available
            if hasattr(self.llm, "cleanup_models"):
                self.llm.cleanup_models()

        except Exception as e:
            logger.error(f"Error during cleanup: {e}")
            # Continue with cleanup even if there are errors

    async def think(self) -> bool:
        """Process current state and decide next actions with appropriate context."""
        try:
            if not self._initialized:
                await self.initialize_mcp_servers()
                self._initialized = True

            # Ensure memory is properly initialized
            if not hasattr(self, "memory") or self.memory is None:
                logger.error("Memory not properly initialized")
                return False

            try:
                # Get properly formatted messages from memory
                formatted_messages = self.memory.get_messages()
            except AttributeError as e:
                logger.error(f"Error accessing memory: {e}")
                return False

            # Only process browser context if browser features are enabled
            if self.browser_enabled and formatted_messages:
                recent_messages = formatted_messages[-3:]
                browser_in_use = any(
                    tc.get("function", {}).get("name") == BrowserUseTool().name
                    for msg in recent_messages
                    if "tool_calls" in msg
                    for tc in msg.get("tool_calls", [])
                )

                if browser_in_use:
                    original_prompt = self.next_step_prompt
                    self.next_step_prompt = (
                        await self.browser_context_helper.format_next_step_prompt()
                    )
                    result = await super().think()
                    self.next_step_prompt = original_prompt  # Restore original prompt
                    return result

            # Pass formatted messages to LLM
            if hasattr(self, "llm") and self.llm is not None:
                self.llm.messages = formatted_messages
                return await super().think()
            else:
                logger.error("LLM not properly initialized")
                return False

        except Exception as e:
            logger.error(f"Error during think step: {e}", exc_info=True)
            return False

    async def step(self) -> str:
        """
        Execute one step of the agent's decision-making process.
        Returns the result of the step execution.
        """
        try:
            # Check initialization
            if not self._initialized:
                logger.info("Initializing agent before first step")
                await self.initialize_mcp_servers()
                self._initialized = True

            # Execute the thinking step
            should_continue = await self.think()

            if not should_continue:
                logger.info("Agent decided to stop.")
                self.state = AgentState.FINISHED
                return "Agent thinking complete - no further action needed"

            # Execute the action step
            action_result = await self.act()
            return action_result
        except Exception as e:
            logger.error(f"Error during agent step: {e}", exc_info=True)
            await self.cleanup()  # Ensure cleanup on error
            return f"Error during step execution: {str(e)}"

    async def run(self, request: Optional[str] = None) -> str:
        """Execute the agent's main loop with enhanced error handling and monitoring."""
        try:
            if request:
                # Add request as user message
                self.memory.add_message("user", request)

            # Ensure initialization
            if not self._initialized:
                await self.initialize_mcp_servers()
                self._initialized = True

            # Initialize browser context if not done
            if self.browser_enabled and not self.browser_context_helper:
                self._initialize_helper()

            return await super().run(
                None
            )  # Pass None since we already added the message

        except Exception as e:
            error_msg = f"Error processing request: {str(e)}"
            logger.error(error_msg)
            # Ensure cleanup happens even if there's an error
            await self.cleanup()
            return error_msg

    def __init__(self, *args, **kwargs):
        # Initialize pydantic parent class first
        super().__init__(*args, **kwargs)

        # Initialize memory and other attributes
        self.memory = Memory()
        self._initialized = False

        # Initialize pydantic fields if not already set
        if not hasattr(self, "__pydantic_fields_set__"):
            self.__pydantic_fields_set__ = set()

        # Set default values for required fields if not provided in kwargs
        try:
            if "llm" not in kwargs:
                self.llm = LLMOptimized(config.llm)

            if "available_tools" not in kwargs:
                # Initialize tools with proper error handling
                tools = []
                try:
                    tools.append(PythonExecute())
                except Exception as e:
                    logger.warning(f"Failed to initialize PythonExecute: {e}")

                try:
                    tools.append(BrowserUseTool())
                except Exception as e:
                    logger.warning(f"Failed to initialize BrowserUseTool: {e}")

                try:
                    tools.append(StrReplaceEditor())
                except Exception as e:
                    logger.warning(f"Failed to initialize StrReplaceEditor: {e}")

                try:
                    tools.append(AskHuman())
                except Exception as e:
                    logger.warning(f"Failed to initialize AskHuman: {e}")

                try:
                    tools.append(Terminate())
                except Exception as e:
                    logger.warning(f"Failed to initialize Terminate: {e}")

                self.available_tools = ToolCollection(*tools)

            if "mcp_clients" not in kwargs:
                self.mcp_clients = MCPClients()

        except Exception as e:
            logger.error(f"Error during Manus initialization: {e}")
            raise
