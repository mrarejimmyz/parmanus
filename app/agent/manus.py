import json
import os
import time
from typing import Dict, List, Optional

from pydantic import Field, model_validator

from app.agent.browser import BrowserContextHelper
from app.agent.toolcall import ToolCallAgent
from app.config import config
from app.logger import logger

# from app.tool.str_replace_editor import StrReplaceEditor # Removed
from app.planning_utils import PlanGenerator, TaskAnalyzer  # Import new utilities
from app.prompt.manus import NEXT_STEP_PROMPT, SYSTEM_PROMPT
from app.reasoning import EnhancedReasoningEngine
from app.schema import Message
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
        """Create a comprehensive task plan using ENHANCED REASONING FRAMEWORK"""
        logger.info(f"🎯 CREATING STRATEGIC PLAN WITH DEEP REASONING: {user_request}")

        # Detect task type first
        task_type = TaskAnalyzer.categorize_task(user_request)
        context = {
            "user_request": user_request,
            "task_type": task_type,
            "complexity": TaskAnalyzer.assess_task_complexity(user_request),
            "optimization_targets": ["quality", "efficiency", "learning"],
            "reasoning_mode": "expert_level",
        }

        # Deep analysis using enhanced reasoning engine
        deep_analysis = await self.reasoning_engine.analyze_task_deeply(
            user_request, context
        )
        deep_analysis["task_type"] = task_type
        logger.info(
            "🧠 DEEP ANALYSIS COMPLETED: {} reasoning layers applied".format(
                len(deep_analysis["reasoning_layers"])
            )
        )

        # Generate optimized strategy
        optimized_strategy = await self.reasoning_engine.generate_optimized_strategy(
            deep_analysis
        )
        logger.info(
            "⚡ OPTIMIZED STRATEGY GENERATED: {}".format(optimized_strategy["approach"])
        )

        # Create enhanced execution plan
        plan = {
            "goal": user_request,
            "task_type": task_type,
            "created_at": time.strftime("%Y-%m-%d %H:%M:%S"),
            "complexity": context["complexity"],
            "phases": PlanGenerator.create_enhanced_phases(
                context, optimized_strategy, {}
            ),
            "success_criteria": PlanGenerator.define_success_criteria(deep_analysis),
        }

        # Reset browser state for new task
        if task_type == "website_review":
            self.browser_state = {
                "current_url": None,
                "content_extracted": False,
                "analysis_complete": False,
                "screenshots_taken": False,
                "last_action": None,
                "page_ready": False,
            }

        self.current_plan = plan
        self.current_phase = 0
        self.current_step = 0

        logger.info(
            "🚀 STRATEGIC EXECUTION PLAN READY: {} optimized phases".format(
                len(plan["phases"])
            )
        )
        return plan

    async def create_todo_list(self, plan: Dict) -> str:
        """Create a detailed todo list from the plan"""
        todo_content = f"# Task Todo List\n\n"
        todo_content += "**Goal:** {}\n\n".format(plan["goal"])
        todo_content += "**Complexity:** {}\n".format(plan["complexity"])
        todo_content += "**Estimated Duration:** {}\n\n".format(
            plan["estimated_duration"]
        )

        for i, phase in enumerate(plan["phases"]):
            status = (
                "CURRENT"
                if i == self.current_phase
                else "PENDING" if i > self.current_phase else "COMPLETE"
            )
            todo_content += "## Phase {}: {} [{}]\n\n".format(
                phase["id"], phase["title"], status
            )
            todo_content += "**Description:** {}\n\n".format(phase["description"])
            todo_content += "**Success Criteria:** {}\n\n".format(
                phase["success_criteria"]
            )
            todo_content += "**Tools Needed:** {}\n\n".format(
                ", ".join(phase["tools_needed"])
            )
            todo_content += "**Steps:**\n"

            for j, step in enumerate(phase["steps"], 1):
                checkbox = "- [ ]"
                if i < self.current_phase:
                    todo_content += "{} {}\n".format(checkbox, step)

            todo_content += "\n"

        # Use direct file write for todo.md
        try:
            with open(self.todo_file_path, "w", encoding="utf-8") as f:
                f.write(todo_content)
            logger.info(f"Todo list saved to {self.todo_file_path}.")
        except Exception as e:
            logger.error(f"Error saving todo list: {e}", exc_info=True)

        return todo_content

    async def update_todo_progress(self):
        """Update todo list to reflect current progress - FIXED VERSION"""
        if not self.current_plan:
            return

        try:
            # Read current todo content directly
            if os.path.exists(self.todo_file_path):
                with open(self.todo_file_path, "r", encoding="utf-8") as f:
                    current_content = f.read()

                # Update phase status and step checkboxes in memory
                lines = current_content.split("\n")
                updated_lines = []

                for line in lines:
                    # Update phase status indicators
                    if line.startswith("## Phase"):
                        try:
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
                            updated_lines.append(
                                line
                            )  # Append original line if parsing fails
                    else:
                        updated_lines.append(line)

                # Update step checkboxes
                for i, phase in enumerate(self.current_plan["phases"]):
                    if i == self.current_phase:
                        for j, step in enumerate(phase["steps"], 1):
                            old_checkbox = f"- [ ] {step}"
                            new_checkbox = f"- [x] {step}"
                            if (
                                j <= self.current_step + 1
                            ):  # Mark current step as complete
                                for k, line in enumerate(updated_lines):
                                    if old_checkbox in line:
                                        updated_lines[k] = line.replace(
                                            old_checkbox, new_checkbox
                                        )
                                        break

                new_todo_content = "\n".join(updated_lines)

                # Use direct file write to update the file
                with open(self.todo_file_path, "w", encoding="utf-8") as f:
                    f.write(new_todo_content)
                logger.info(f"Todo list updated at {self.todo_file_path}.")
            else:
                logger.warning(
                    f"Todo file not found at {self.todo_file_path}, cannot update."
                )
        except Exception as e:
            logger.error(f"Error updating todo list: {e}", exc_info=True)

    async def think(self) -> bool:
        """Enhanced thinking process with improved state management"""
        if not self.current_plan:
            self.current_plan = await self.create_task_plan(
                self.memory.get_user_request()
            )
            await self.create_todo_list(self.current_plan)
            return True

        task_type = self.current_plan.get("task_type")
        if task_type == "website_review":
            current_phase = self.current_plan["phases"][self.current_phase]
            current_step = current_phase["steps"][self.current_step]

            next_action = await self.handle_browser_task(current_step)
            if next_action:
                self.tool_calls = [{"name": "browser_use", "arguments": next_action}]
                return True

            # Check if we're done
            if (
                self.current_phase >= len(self.current_plan["phases"]) - 1
                and self.current_step >= len(current_phase["steps"]) - 1
            ):
                logger.info("✅ Website review completed successfully")
                return False

            # Continue to next step
            return True

        return await super().think()

    async def handle_browser_task(self, step: str) -> Optional[Dict]:
        """Advanced browser task handling with state management"""
        logger.debug(f"Handling browser task: {step}")
        logger.debug(f"Current browser state: {self.browser_state}")

        # Reset if we're starting a new review
        if step == "Navigate to website" and not self.browser_state["page_ready"]:
            url = self.memory.get_user_request().split()[-1]
            self.browser_state.update(
                {"current_url": url, "page_ready": True, "last_action": step}
            )
            return {"action": "go_to_url", "url": url}

        # Only proceed if page is ready
        if not self.browser_state.get("page_ready"):
            logger.warning("Page not ready, waiting for navigation")
            return None

        # Skip duplicate actions
        if self.browser_state.get("last_action") == step:
            logger.debug(f"Skipping duplicate action: {step}")
            # Move to next step
            current_phase = self.current_plan["phases"][self.current_phase]
            if self.current_step + 1 < len(current_phase["steps"]):
                self.current_step += 1
            elif self.current_phase + 1 < len(self.current_plan["phases"]):
                self.current_phase += 1
                self.current_step = 0
            return None

        self.browser_state["last_action"] = step

        # Handle each step appropriately
        if (
            step == "Extract main content"
            and not self.browser_state["content_extracted"]
        ):
            self.browser_state["content_extracted"] = True
            return {
                "action": "extract_content",
                "css_selector": "main,article,#content,.content,body",
            }

        elif (
            step == "Capture screenshots"
            and not self.browser_state["screenshots_taken"]
        ):
            self.browser_state["screenshots_taken"] = True
            return {
                "action": "screenshot",
                "path": os.path.join(config.workspace_root, "screenshots"),
            }

        elif step == "Analyze page structure" and not self.browser_state.get(
            "structure_analyzed"
        ):
            self.browser_state["structure_analyzed"] = True
            content_path = os.path.join(config.workspace_root, "analysis")
            os.makedirs(content_path, exist_ok=True)
            return {"action": "analyze_structure", "output_path": content_path}

        elif (
            "Generate analysis.md" in step
            and not self.browser_state["analysis_complete"]
        ):
            self.browser_state["analysis_complete"] = True
            analysis_path = os.path.join(config.workspace_root, "analysis.md")
            report = self.generate_analysis_report()
            with open(analysis_path, "w", encoding="utf-8") as f:
                f.write(report)
            logger.info(f"Analysis report generated at {analysis_path}")
            return None

        elif "Create summary.md" in step and not self.browser_state.get(
            "summary_complete"
        ):
            self.browser_state["summary_complete"] = True
            summary_path = os.path.join(config.workspace_root, "summary.md")
            with open(summary_path, "w", encoding="utf-8") as f:
                f.write(self.generate_summary())
            logger.info(f"Summary generated at {summary_path}")
            return None

        return None

    def generate_summary(self) -> str:
        """Generate a concise summary of the website review"""
        return f"""# Website Review Summary

## Overview
- URL: {self.browser_state.get('current_url')}
- Review Date: {time.strftime('%Y-%m-%d %H:%M:%S')}

## Key Findings
- Content Extraction: {"Complete" if self.browser_state.get('content_extracted') else "Pending"}
- Visual Documentation: {"Complete" if self.browser_state.get('screenshots_taken') else "Pending"}
- Structure Analysis: {"Complete" if self.browser_state.get('structure_analyzed') else "Pending"}

## Status
Review Status: {"Complete" if self.browser_state.get('analysis_complete') else "In Progress"}
"""
