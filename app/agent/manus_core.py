import asyncio
import json
import os
import time
from typing import Dict, List, Optional

from pydantic import Field, model_validator

from app.agent.browser import BrowserContextHelper
from app.agent.manus_browser_handler import ManusBrowserHandler
from app.agent.manus_planning import ManusPlanning
from app.agent.manus_utils import ManusUtils
from app.agent.toolcall import ToolCallAgent
from app.config import config
from app.exceptions import AgentTaskComplete
from app.logger import logger
from app.prompt.manus import NEXT_STEP_PROMPT, SYSTEM_PROMPT
from app.reasoning import EnhancedReasoningEngine
from app.schema import AgentState, Function, Message, ToolCall
from app.tool import Terminate, ToolCollection
from app.tool.ask_human import AskHuman
from app.tool.browser_use_tool import BrowserUseTool
from app.tool.mcp import MCPClients, MCPClientTool
from app.tool.python_execute import PythonExecute


class Manus(ToolCallAgent):
    """A versatile general-purpose agent with enhanced planning and reasoning capabilities."""

    name: str = "Manus"
    description: str = (
        "A versatile agent that can solve various tasks using multiple tools with strategic planning"
    )

    system_prompt: str = SYSTEM_PROMPT.format(directory=config.workspace_root)
    next_step_prompt: str = NEXT_STEP_PROMPT

    max_observe: int = config.max_observe
    max_steps: int = config.max_steps

    # Enhanced reasoning and planning
    reasoning_framework: EnhancedReasoningEngine = Field(
        default_factory=EnhancedReasoningEngine
    )
    current_plan: Optional[Dict] = None
    current_phase: int = 0
    current_step: int = 0
    todo_file_path: str = ""

    # MCP clients for remote tool access
    mcp_clients: MCPClients = Field(default_factory=MCPClients)

    # Add general-purpose tools to the tool collection
    available_tools: ToolCollection = Field(
        default_factory=lambda: ToolCollection(
            PythonExecute(),
            BrowserUseTool(),
            AskHuman(),
            Terminate(),
        )
    )

    special_tool_names: list[str] = Field(default_factory=lambda: [Terminate().name])
    browser_context_helper: Optional[BrowserContextHelper] = None

    # Track connected MCP servers
    connected_servers: Dict[str, str] = Field(
        default_factory=dict
    )  # server_id -> url/command

    # Add browser state tracking (renamed from _browser_state to browser_state)
    browser_state: Dict = Field(
        default_factory=lambda: {
            "current_url": None,
            "content_extracted": False,
            "analysis_complete": False,
            "screenshots_taken": False,
            "last_action": None,
            "page_ready": False,
            "structure_analyzed": False,
            "summary_complete": False,
        }
    )

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        logger.debug("Manus __init__ started.")
        self.todo_file_path = os.path.join(config.workspace_root, "todo.md")

        # ENHANCED AI SYSTEM: Initialize reasoning and learning systems
        from app.enhanced_memory import EnhancedMemorySystem
        from app.reasoning import EnhancedReasoningEngine

        self.reasoning_engine = EnhancedReasoningEngine()
        self.memory_system = EnhancedMemorySystem()
        self.optimization_mode = True
        self.deep_reasoning_enabled = True

        self.planning_module = ManusPlanning(self)
        self.browser_handler = ManusBrowserHandler(self)
        self.utils_module = ManusUtils(self)

        logger.info(
            "🧠 ENHANCED AI SYSTEM INITIALIZED: Deep reasoning and learning enabled"
        )
        logger.debug("Manus __init__ completed.")

    @classmethod
    async def create(cls, **kwargs):
        """Asynchronous factory method to create a Manus instance."""
        logger.debug("Manus.create started.")
        try:
            instance = cls(**kwargs)
            logger.debug("Manus instance created.")
            # Any asynchronous initialization logic can go here if needed
            return instance
        except Exception as e:
            logger.error(f"Error during Manus.create: {e}", exc_info=True)
            raise

    async def create_task_plan(self, user_request: str) -> Dict:
        return await self.planning_module.create_task_plan(user_request)

    async def create_todo_list(self, plan: Dict) -> str:
        return await self.planning_module.create_todo_list(plan)

    async def update_todo_progress(self):
        return await self.planning_module.update_todo_progress()

    async def handle_browser_task(self, step: str) -> Optional[Dict]:
        return await self.browser_handler.handle_browser_task(step)

    async def _initialize_browser_state(self):
        return await self.browser_handler._initialize_browser_state()

    async def _extract_url_from_request(self, step: str) -> Optional[str]:
        return self.browser_handler._extract_url_from_request(step)

    async def think(self) -> bool:
        """Think about the next action based on the current plan phase and step"""
        try:
            await self._initialize_browser_state()

            if not await self.utils_module._validate_current_position():
                logger.error("Invalid position in plan")
                return False

            current_phase = await self.utils_module._get_current_phase()
            current_step = await self.utils_module._get_current_step()

            # This should never happen now due to validation, but keep as safety
            if not current_phase or not current_step:
                logger.error("No valid phase or step found in plan")
                return False
            url = await self._extract_url_from_request(current_step)
            if url:
                if not self.browser_state.get("initialized"):
                    browser_args = {"action": "initialize", "url": url}
                    func = Function(
                        name="browser_use", arguments=json.dumps(browser_args)
                    )
                    self.tool_calls = [
                        ToolCall(
                            id="browser_init_" + str(int(time.time())),
                            type="function",
                            function=func,
                        )
                    ]
                    return True

            # Handle steps
            if (
                "Navigate to website" in current_step
                or "navigate" in current_step.lower()
            ):
                await self.utils_module.progress_to_next_step()
                return True

        except AgentTaskComplete:
            raise
        except Exception as e:
            logger.error(f"Error in think(): {str(e)}")
            return False

    async def step(self) -> str:
        """Execute a single step, creating a plan if needed"""
        try:
            # On first step, create plan if none exists
            if self.current_step == 1 and not self.current_plan:
                # Get the last user request from memory
                user_messages = [
                    msg for msg in self.memory.messages if msg.role == "user"
                ]
                if not user_messages:
                    raise ValueError("No user request found in memory")

                request = user_messages[-1].content
                self.current_plan = await self.create_task_plan(request)
                await self.create_todo_list(self.current_plan)
                return "Created initial task plan"

            # For subsequent steps, use think() to determine and take actions
            success = await self.think()
            if success:
                return "Thinking complete - no action needed..."
            else:
                logger.error("Think failed")
                return "Error during think phase"

        except AgentTaskComplete:
            self.state = AgentState.FINISHED
            return "Task completed successfully"
        except Exception as e:
            logger.error(f"Error in step(): {str(e)}")
            return f"Error in step: {str(e)}"
