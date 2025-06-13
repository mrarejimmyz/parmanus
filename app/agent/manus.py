import json
import os
import time
from typing import Dict, List, Optional

from pydantic import Field, model_validator

from app.agent.browser import BrowserContextHelper
from app.agent.toolcall import ToolCallAgent
from app.config import config
from app.logger import logger
from app.prompt.manus import NEXT_STEP_PROMPT, SYSTEM_PROMPT
from app.reasoning import EnhancedReasoningEngine, TaskPhase
from app.schema import Message
from app.tool import Terminate, ToolCollection
from app.tool.ask_human import AskHuman
from app.tool.browser_use_tool import BrowserUseTool
from app.tool.mcp import MCPClients, MCPClientTool
from app.tool.python_execute import PythonExecute
from app.tool.str_replace_editor import StrReplaceEditor
from app.planning_utils import TaskAnalyzer, PlanGenerator # Import new utilities


class Manus(ToolCallAgent):
    """A versatile general-purpose agent with enhanced planning and reasoning capabilities."""

    name: str = "Manus"
    description: str = (
        "A versatile agent that can solve various tasks using multiple tools with strategic planning"
    )

    system_prompt: str = SYSTEM_PROMPT.format(directory=config.workspace_root)
    next_step_prompt: str = NEXT_STEP_PROMPT

    max_observe: int = 10000
    max_steps: int = 25  # Increased for better planning

    # Enhanced reasoning and planning
    reasoning_framework: EnhancedReasoningEngine = Field(default_factory=EnhancedReasoningEngine)
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

        # ENHANCED AI SYSTEM: Initialize reasoning and learning systems
        from app.enhanced_memory import EnhancedMemorySystem
        from app.reasoning import EnhancedReasoningEngine

        self.reasoning_engine = EnhancedReasoningEngine()
        self.memory_system = EnhancedMemorySystem()
        self.optimization_mode = True
        self.deep_reasoning_enabled = True

        logger.info(
            "🧠 ENHANCED AI SYSTEM INITIALIZED: Deep reasoning and learning enabled"
        )

    async def create_task_plan(self, user_request: str) -> Dict:
        """Create a comprehensive task plan using ENHANCED REASONING FRAMEWORK"""
        logger.info(f"🎯 CREATING STRATEGIC PLAN WITH DEEP REASONING: {user_request}")

        # ENHANCED REASONING: Perform deep multi-layered analysis
        context = {
            "user_request": user_request,
            "complexity": TaskAnalyzer.assess_task_complexity(user_request), # Use TaskAnalyzer
            "optimization_targets": ["quality", "efficiency", "learning"],
            "reasoning_mode": "expert_level",
        }

        # Deep analysis using enhanced reasoning engine
        deep_analysis = await self.reasoning_engine.analyze_task_deeply(
            user_request, context
        )
        logger.info(
            f"🧠 DEEP ANALYSIS COMPLETED: {len(deep_analysis['reasoning_layers'])} reasoning layers applied"
        )

        # Generate optimized strategy
        optimized_strategy = await self.reasoning_engine.generate_optimized_strategy(
            deep_analysis
        )
        logger.info(
            f"⚡ OPTIMIZED STRATEGY GENERATED: {optimized_strategy["approach"]}"
        )

        # Get learned insights from memory system
        learned_strategy = await self.memory_system.get_optimized_strategy(
            task_type=TaskAnalyzer.categorize_task(user_request), context=context # Use TaskAnalyzer
        )
        logger.info(
            f"🎓 LEARNING INSIGHTS APPLIED: Confidence {learned_strategy["confidence_score"]:.2f}"
        )

        # Create enhanced execution plan
        plan = {
            "goal": user_request,
            "created_at": time.strftime("%Y-%m-%d %H:%M:%S"),
            "reasoning_analysis": deep_analysis,
            "optimized_strategy": optimized_strategy,
            "learned_insights": learned_strategy,
            "complexity": context["complexity"],
            "estimated_duration": TaskAnalyzer.estimate_duration(deep_analysis), # Use TaskAnalyzer
            "phases": PlanGenerator.create_enhanced_phases(
                deep_analysis, optimized_strategy, learned_strategy
            ),
            "success_criteria": PlanGenerator.define_success_criteria(deep_analysis),
            "optimization_targets": context["optimization_targets"],
            "quality_standards": ["expert_level", "optimized", "learned"],
            "learning_mode": "continuous_improvement",
        }

        # Save comprehensive plan to workspace
        plan_file = os.path.join(config.workspace_root, "enhanced_task_plan.json")
        try:
            with open(plan_file, "w") as f:
                json.dump(plan, f, indent=2)
            logger.info(f"📋 ENHANCED TASK PLAN SAVED: {plan_file}")
        except Exception as e:
            logger.warning(f"Could not save enhanced plan file: {e}")

        self.current_plan = plan
        self.current_phase = 0
        self.current_step = 0

        logger.info(
            f"🚀 STRATEGIC EXECUTION PLAN READY: {len(plan["phases"])} optimized phases"
        )
        return plan

    async def create_todo_list(self, plan: Dict) -> str:
        """Create a detailed todo list from the plan"""
        todo_content = f"# Task Todo List\n\n"
        todo_content += f"**Goal:** {plan["goal"]}\n\n"
        todo_content += f"**Complexity:** {plan["complexity"]}\n"
        todo_content += f"**Estimated Duration:** {plan["estimated_duration"]}\n\n"

        for i, phase in enumerate(plan["phases"]):
            status = (
                "CURRENT"
                if i == self.current_phase
                else "PENDING" if i > self.current_phase else "COMPLETE"
            )
            todo_content += f"## Phase {phase["id"]}: {phase["title"]} [{status}]\n\n"
            todo_content += f"**Description:** {phase["description"]}\n\n"
            todo_content += f"**Success Criteria:** {phase["success_criteria"]}\n\n"
            todo_content += f"**Tools Needed:** {", ".join(phase["tools_needed"])}\n\n"
            todo_content += f"**Steps:**\n"

            for j, step in enumerate(phase["steps"], 1):
                checkbox = "- [ ]"
                if i < self.current_phase:
                    checkbox = "- [x]"
                elif i == self.current_phase and j <= self.current_step:
                    checkbox = "- [x]"

                todo_content += f"{checkbox} {step}\n"

            todo_content += "\n"

        # Save todo list with UTF-8 encoding
        try:
            with open(self.todo_file_path, "w", encoding="utf-8") as f:
                f.write(todo_content)
            logger.info(f"Todo list saved to {self.todo_file_path}")
        except Exception as e:
            logger.warning(f"Could not save todo list: {e}")
            # Fallback: try with ASCII-only content
            try:
                ascii_content = todo_content.encode("ascii", "ignore").decode("ascii")
                with open(self.todo_file_path, "w", encoding="ascii") as f:
                    f.write(ascii_content)
                logger.info(
                    f"Todo list saved with ASCII encoding to {self.todo_file_path}"
                )
            except Exception as e2:
                logger.error(f"Failed to save todo list even with ASCII: {e2}")

        return todo_content

    async def update_todo_progress(self):
        """Update todo list to reflect current progress - FIXED VERSION"""
        if not self.current_plan:
            return

        try:
            # Read current todo
            if os.path.exists(self.todo_file_path):
                with open(self.todo_file_path, "r", encoding="utf-8") as f:
                    content = f.read()

                # Update phase status
                lines = content.split("\n")
                updated_lines = []

                for line in lines:
                    # Update phase status indicators
                    if line.startswith("## Phase"):
                        try:
                            # EMERGENCY FIX: Add bounds checking
                            phase_parts = line.split()
                            if len(phase_parts) >= 3:
                                phase_num = int(phase_parts[2].rstrip(":"))
                                if phase_num == self.current_phase + 1:
                                    line = line.replace("[PENDING]", "[CURRENT]")
                                elif phase_num < self.current_phase + 1:
                                    line = line.replace("[CURRENT]", "[COMPLETE]")
                                    line = line.replace("[PENDING]", "[COMPLETE]")
                                updated_lines.append(line)
                            else:
                                updated_lines.append(line)
                        except ValueError:
                            updated_lines.append(line) # Append original line if parsing fails
                    else:
                        updated_lines.append(line)

                # Update step checkboxes
                for i, phase in enumerate(self.current_plan["phases"]):
                    if i == self.current_phase:
                        for j, step in enumerate(phase["steps"], 1):
                            old_checkbox = f"- [ ] {step}"
                            new_checkbox = f"- [x] {step}"
                            if j <= self.current_step + 1: # Mark current step as complete
                                for k, line in enumerate(updated_lines):
                                    if old_checkbox in line:
                                        updated_lines[k] = line.replace(old_checkbox, new_checkbox)
                                        break

                updated_content = "\n".join(updated_lines)

                with open(self.todo_file_path, "w", encoding="utf-8") as f:
                    f.write(updated_content)
                logger.info(f"Todo list updated at {self.todo_file_path}")

            else:
                logger.warning(f"Todo file not found at {self.todo_file_path}")
        except Exception as e:
            logger.error(f"Error updating todo list: {e}")


