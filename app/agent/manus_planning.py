import json
import os
import time
from typing import Dict

from app.config import config
from app.logger import logger
from app.planning_utils import PlanGenerator, TaskAnalyzer


class ManusPlanning:
    def __init__(self, agent_instance):
        self.agent = agent_instance
        self.todo_file_path = os.path.join(config.workspace_root, "todo.md")

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
        deep_analysis = await self.agent.reasoning_engine.analyze_task_deeply(
            user_request, context
        )
        deep_analysis["task_type"] = task_type
        logger.info(
            "🧠 DEEP ANALYSIS COMPLETED: {} reasoning layers applied".format(
                len(deep_analysis["reasoning_layers"])
            )
        )

        # Generate optimized strategy
        optimized_strategy = (
            await self.agent.reasoning_engine.generate_optimized_strategy(deep_analysis)
        )
        logger.info(
            "⚡ OPTIMIZED STRATEGY GENERATED: {}".format(optimized_strategy["approach"])
        )

        # Create enhanced execution plan with estimated duration
        plan = {
            "goal": user_request,
            "task_type": task_type,
            "created_at": time.strftime("%Y-%m-%d %H:%M:%S"),
            "complexity": context["complexity"],
            "estimated_duration": "5-10 minutes",  # Add default estimated duration
            "phases": PlanGenerator.create_enhanced_phases(
                context, optimized_strategy, {}
            ),
            "success_criteria": PlanGenerator.define_success_criteria(deep_analysis),
        }

        # Reset browser state for new task
        if task_type == "website_review":
            self.agent.browser_state = {
                "current_url": None,
                "content_extracted": False,
                "analysis_complete": False,
                "screenshots_taken": False,
                "last_action": None,
                "page_ready": False,
                "structure_analyzed": False,
                "summary_complete": False,
            }

        self.agent.current_plan = plan
        self.agent.current_phase = 0
        self.agent.current_step = 0

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
                if i == self.agent.current_phase
                else "PENDING" if i > self.agent.current_phase else "COMPLETE"
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
                if i < self.agent.current_phase:
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
        if not self.agent.current_plan:
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
                                if phase_num == self.agent.current_phase + 1:
                                    line = line.replace("[PENDING]", "[CURRENT]")
                                elif phase_num < self.agent.current_phase + 1:
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
                for i, phase in enumerate(self.agent.current_plan["phases"]):
                    if i == self.agent.current_phase:
                        for j, step in enumerate(phase["steps"], 1):
                            old_checkbox = f"- [ ] {step}"
                            new_checkbox = f"- [x] {step}"
                            if (
                                j <= self.agent.current_step + 1
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
