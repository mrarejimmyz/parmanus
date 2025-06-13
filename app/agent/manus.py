from typing import Dict, List, Optional

from pydantic import Field, model_validator

from app.agent.browser import BrowserContextHelper
from app.agent.toolcall import ToolCallAgent
from app.config import config
from app.logger import logger
from app.prompt.manus import NEXT_STEP_PROMPT, SYSTEM_PROMPT
from app.reasoning import ReasoningFramework, TaskPhase
from app.tool import Terminate, ToolCollection
from app.tool.ask_human import AskHuman
from app.tool.browser_use_tool import BrowserUseTool
from app.tool.mcp import MCPClients, MCPClientTool
from app.tool.python_execute import PythonExecute
from app.tool.str_replace_editor import StrReplaceEditor
from app.schema import Message
import json
import os


class Manus(ToolCallAgent):
    """A versatile general-purpose agent with enhanced planning and reasoning capabilities."""

    name: str = "Manus"
    description: str = "A versatile agent that can solve various tasks using multiple tools with strategic planning"

    system_prompt: str = SYSTEM_PROMPT.format(directory=config.workspace_root)
    next_step_prompt: str = NEXT_STEP_PROMPT

    max_observe: int = 10000
    max_steps: int = 25  # Increased for better planning

    # Enhanced reasoning and planning
    reasoning_framework: ReasoningFramework = Field(default_factory=ReasoningFramework)
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
            StrReplaceEditor(),
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
    
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.todo_file_path = os.path.join(config.workspace_root, "todo.md")
    
    async def create_task_plan(self, user_request: str) -> Dict:
        """Create a comprehensive task plan using the reasoning framework"""
        logger.info(f"Creating strategic plan for: {user_request}")
        
        # Analyze the request
        analysis = self.reasoning_framework.analyze_request(user_request)
        
        # Create execution plan
        plan = self.reasoning_framework.create_execution_plan(analysis)
        
        # Save plan to workspace
        plan_file = os.path.join(config.workspace_root, "task_plan.json")
        try:
            with open(plan_file, 'w') as f:
                json.dump(plan, f, indent=2)
            logger.info(f"Task plan saved to {plan_file}")
        except Exception as e:
            logger.warning(f"Could not save plan file: {e}")
        
        self.current_plan = plan
        self.current_phase = 0
        self.current_step = 0
        
        return plan
    
    async def create_todo_list(self, plan: Dict) -> str:
        """Create a detailed todo list from the plan"""
        todo_content = f"# Task Todo List\n\n"
        todo_content += f"**Goal:** {plan['goal']}\n\n"
        todo_content += f"**Complexity:** {plan['complexity']}\n"
        todo_content += f"**Estimated Duration:** {plan['estimated_duration']}\n\n"
        
        for i, phase in enumerate(plan['phases']):
            status = "🔄 CURRENT" if i == self.current_phase else "⏳ PENDING" if i > self.current_phase else "✅ COMPLETE"
            todo_content += f"## Phase {phase['id']}: {phase['title']} {status}\n\n"
            todo_content += f"**Description:** {phase['description']}\n\n"
            todo_content += f"**Success Criteria:** {phase['success_criteria']}\n\n"
            todo_content += f"**Tools Needed:** {', '.join(phase['tools_needed'])}\n\n"
            todo_content += f"**Steps:**\n"
            
            for j, step in enumerate(phase['steps'], 1):
                checkbox = "- [ ]"
                if i < self.current_phase:
                    checkbox = "- [x]"
                elif i == self.current_phase and j <= self.current_step:
                    checkbox = "- [x]"
                
                todo_content += f"{checkbox} {step}\n"
            
            todo_content += "\n"
        
        # Save todo list
        try:
            with open(self.todo_file_path, 'w') as f:
                f.write(todo_content)
            logger.info(f"Todo list saved to {self.todo_file_path}")
        except Exception as e:
            logger.warning(f"Could not save todo list: {e}")
        
        return todo_content
    
    async def update_todo_progress(self):
        """Update todo list to reflect current progress"""
        if not self.current_plan:
            return
        
        try:
            # Read current todo
            if os.path.exists(self.todo_file_path):
                with open(self.todo_file_path, 'r') as f:
                    content = f.read()
                
                # Update phase status
                lines = content.split('\n')
                updated_lines = []
                
                for line in lines:
                    # Update phase status indicators
                    if line.startswith("## Phase"):
                        phase_num = int(line.split()[2].rstrip(':'))
                        if phase_num <= self.current_phase + 1:
                            if "⏳ PENDING" in line:
                                line = line.replace("⏳ PENDING", "🔄 CURRENT" if phase_num == self.current_phase + 1 else "✅ COMPLETE")
                            elif "🔄 CURRENT" in line and phase_num < self.current_phase + 1:
                                line = line.replace("🔄 CURRENT", "✅ COMPLETE")
                    
                    updated_lines.append(line)
                
                # Save updated content
                with open(self.todo_file_path, 'w') as f:
                    f.write('\n'.join(updated_lines))
                    
        except Exception as e:
            logger.warning(f"Could not update todo progress: {e}")
    
    async def get_current_phase_guidance(self) -> str:
        """Get guidance for the current phase"""
        if not self.current_plan or self.current_phase >= len(self.current_plan['phases']):
            return "No active plan or all phases complete"
        
        current_phase_info = self.current_plan['phases'][self.current_phase]
        total_phases = len(self.current_plan['phases'])
        
        return self.reasoning_framework.get_next_step_guidance(
            self.current_step + 1,
            len(current_phase_info['steps']),
            current_phase_info
        )
    
    async def advance_step(self):
        """Advance to the next step or phase"""
        if not self.current_plan:
            return
        
        current_phase_info = self.current_plan['phases'][self.current_phase]
        
        # Check if current phase is complete
        if self.current_step >= len(current_phase_info['steps']) - 1:
            # Move to next phase
            if self.current_phase < len(self.current_plan['phases']) - 1:
                self.current_phase += 1
                self.current_step = 0
                logger.info(f"Advanced to phase {self.current_phase + 1}")
            else:
                logger.info("All phases completed")
        else:
            # Move to next step in current phase
            self.current_step += 1
            logger.info(f"Advanced to step {self.current_step + 1} in phase {self.current_phase + 1}")
        
        await self.update_todo_progress()
    
    async def step(self) -> str:
        """Enhanced step method with planning integration"""
        # Check if we need to create a plan
        if not self.current_plan:
            # Get user request from memory
            user_messages = [msg for msg in self.memory.messages if msg.role == "user"]
            if user_messages:
                user_request = user_messages[-1].content
                
                # Create plan and todo list
                plan = await self.create_task_plan(user_request)
                todo_list = await self.create_todo_list(plan)
                
                # Add planning message to memory
                planning_msg = f"""
📋 TASK ANALYSIS COMPLETE

🎯 **Goal:** {plan['goal']}
📊 **Complexity:** {plan['complexity']}
⏱️ **Estimated Duration:** {plan['estimated_duration']}
📝 **Phases:** {len(plan['phases'])}

✅ **Plan created and todo list saved to workspace**

Now proceeding with systematic execution...
"""
                self.memory.add_message(Message.assistant_message(planning_msg))
                
                return planning_msg
        
        # Get guidance for current phase
        guidance = await self.get_current_phase_guidance()
        
        # Add guidance to next step prompt
        enhanced_prompt = f"""
{self.next_step_prompt}

{guidance}

CURRENT PROGRESS:
- Phase: {self.current_phase + 1}/{len(self.current_plan['phases']) if self.current_plan else 0}
- Step: {self.current_step + 1}

Execute the next step systematically and update your progress.
"""
        
        # Temporarily update the next step prompt
        original_prompt = self.next_step_prompt
        self.next_step_prompt = enhanced_prompt
        
        try:
            # Execute the parent step method
            result = await super().step()
            
            # Advance progress after successful step
            await self.advance_step()
            
            return result
            
        finally:
            # Restore original prompt
            self.next_step_prompt = original_prompt
    _initialized: bool = False

    @model_validator(mode="after")
    def initialize_helper(self) -> "Manus":
        """Initialize basic components synchronously."""
        self.browser_context_helper = BrowserContextHelper(self)
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
        if self.browser_context_helper:
            await self.browser_context_helper.cleanup_browser()
        # Disconnect from all MCP servers only if we were initialized
        if self._initialized:
            await self.disconnect_mcp_server()
            self._initialized = False

    async def think(self) -> bool:
        """Process current state and decide next actions with appropriate context."""
        if not self._initialized:
            await self.initialize_mcp_servers()
            self._initialized = True

        original_prompt = self.next_step_prompt
        recent_messages = self.memory.messages[-3:] if self.memory.messages else []
        browser_in_use = any(
            tc.function.name == BrowserUseTool().name
            for msg in recent_messages
            if msg.tool_calls
            for tc in msg.tool_calls
        )

        if browser_in_use:
            self.next_step_prompt = (
                await self.browser_context_helper.format_next_step_prompt()
            )

        result = await super().think()

        # Restore original prompt
        self.next_step_prompt = original_prompt

        return result
