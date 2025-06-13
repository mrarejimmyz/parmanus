import asyncio
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
            self.browser_state = {
                "current_url": None,
                "content_extracted": False,
                "analysis_complete": False,
                "screenshots_taken": False,
                "last_action": None,
                "page_ready": False,
                "structure_analyzed": False,
                "summary_complete": False,
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
        try:
            if not self.current_plan:
                # Get user request from recent messages
                user_messages = [
                    msg for msg in self.memory.messages if msg.role == "user"
                ]
                if not user_messages:
                    logger.error("No user request found in memory")
                    return False
                user_request = user_messages[-1].content

                self.current_plan = await self.create_task_plan(user_request)
                await self.create_todo_list(self.current_plan)
                return True

            task_type = self.current_plan.get("task_type")
            if task_type == "website_review":
                # Check completion state first
                if self.browser_state.get(
                    "analysis_complete"
                ) and self.browser_state.get("summary_complete"):
                    logger.info("✅ Website review already completed")
                    return False

                # Validate current phase and step
                if self.current_phase >= len(self.current_plan["phases"]):
                    logger.info("✅ Website review completed - all phases done")
                    return False

                current_phase = self.current_plan["phases"][self.current_phase]

                # Check if current phase is complete
                if not current_phase.get("steps") or self.current_step >= len(
                    current_phase["steps"]
                ):
                    if self.current_phase + 1 < len(self.current_plan["phases"]):
                        self.current_phase += 1
                        self.current_step = 0
                        logger.info(f"Moving to phase {self.current_phase}")
                        await self.update_todo_progress()
                        return True
                    else:
                        self.browser_state["analysis_complete"] = True
                        self.browser_state["summary_complete"] = True
                        logger.info("✅ Website review completed successfully")
                        return False

                # Get current step
                current_step = current_phase["steps"][self.current_step]
                logger.debug(f"Processing step {self.current_step}: {current_step}")

                # Handle any navigation step immediately
                if (
                    "navigate" in current_step.lower()
                    or current_step == "Navigate to website"
                ):
                    logger.info(f"Beginning navigation step: {current_step}")
                    next_action = await self.handle_browser_task(current_step)
                    if next_action:
                        self.tool_calls = [
                            {"name": "browser_use", "arguments": next_action}
                        ]
                        return True

                # For non-navigation steps, require and verify page readiness
                if not self.browser_state.get("page_ready"):
                    if self.browser_state.get("current_url"):
                        # Navigation happened but page not ready yet
                        self.browser_state["page_ready"] = True
                        logger.info("Navigation complete, marking page as ready")
                        await self.progress_to_next_step()
                    logger.debug("Waiting for page to be ready")
                    return True

                # Execute the current step
                next_action = await self.handle_browser_task(current_step)
                if next_action:
                    self.tool_calls = [
                        {"name": "browser_use", "arguments": next_action}
                    ]
                    return True

                # Check if we're done with all phases and steps
                if (
                    self.current_phase >= len(self.current_plan["phases"]) - 1
                    and self.current_step >= len(current_phase["steps"]) - 1
                ):
                    logger.info("✅ Website review completed successfully")
                    self.browser_state["analysis_complete"] = True
                    self.browser_state["summary_complete"] = True
                    return False

                # Progress to next step if appropriate
                if self.browser_state.get("page_ready"):
                    await self.progress_to_next_step()
                return True

            return await super().think()

        except Exception as e:
            logger.error(f"Error in think(): {str(e)}")
            return False

    async def handle_browser_task(self, step: str) -> Optional[Dict]:
        """Advanced browser task handling with state management"""
        logger.debug(f"Handling browser task: {step}")
        logger.debug(f"Current browser state: {self.browser_state}")

        try:
            if not self.current_plan or "phases" not in self.current_plan:
                logger.error("No valid plan exists")
                return None

            # Handle website navigation
            if step == "Navigate to website" or "navigate" in step.lower():
                # Only navigate if we haven't already navigated or page isn't ready
                if not self.browser_state.get(
                    "current_url"
                ) or not self.browser_state.get("page_ready"):
                    # Get user request from recent messages
                    user_messages = [
                        msg for msg in self.memory.messages if msg.role == "user"
                    ]
                    if not user_messages:
                        logger.error("No user request found in memory")
                        return None

                    # Extract URL from user request
                    message = user_messages[-1].content.lower()
                    words = message.split()

                    # Try to find the URL after "review" or at the end
                    url = None
                    try:
                        if "review" in words:
                            url_index = words.index("review") + 1
                            if url_index < len(words):
                                url = words[url_index]
                        if not url:
                            # Fallback to last word
                            url = words[-1]
                    except:
                        url = words[-1]  # Fallback

                    # Normalize URL
                    if not url:
                        logger.error("No URL found in user request")
                        return None

                    # Clean up URL
                    url = url.rstrip(".")
                    if not url.startswith(("http://", "https://")):
                        url = f"https://{url}"

                    logger.info(f"Extracted and navigating to URL: {url}")

                    # Update state and initiate navigation
                    self.browser_state.update(
                        {
                            "current_url": url,
                            "page_ready": False,
                            "last_action": None,  # Reset to allow retry if needed
                        }
                    )
                    return {"action": "go_to_url", "url": url}
                elif not self.browser_state.get("page_ready"):
                    # Navigation completed, update state and progress
                    self.browser_state["page_ready"] = True
                    logger.info("Navigation completed successfully")
                    await self.progress_to_next_step()
                return None

            # Only proceed with other steps if page is ready and we've navigated
            if not self.browser_state.get("current_url"):
                logger.error("No URL set, cannot proceed with browser task")
                return None

            # Wait for page to be ready if needed
            if not self.browser_state.get("page_ready"):
                logger.debug("Waiting for page to be ready before proceeding")
                await self.ensure_page_ready()
                if not self.browser_state.get("page_ready"):
                    return None

            # Validate current phase and step
            if self.current_phase >= len(self.current_plan["phases"]):
                logger.error("Current phase index out of range")
                return None

            current_phase = self.current_plan["phases"][self.current_phase]

            if not current_phase.get("steps") or self.current_step >= len(
                current_phase["steps"]
            ):
                logger.debug("Current step index out of range, moving to next phase")
                await self.progress_to_next_step()
                return None

            # Skip duplicate actions
            if self.browser_state.get("last_action") == step:
                logger.debug(f"Skipping duplicate action: {step}")
                await self.progress_to_next_step()
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
                await self.progress_to_next_step()
                return None

            elif "Create summary.md" in step and not self.browser_state.get(
                "summary_complete"
            ):
                self.browser_state["summary_complete"] = True
                summary_path = os.path.join(config.workspace_root, "summary.md")
                with open(summary_path, "w", encoding="utf-8") as f:
                    f.write(self.generate_summary())
                logger.info(f"Summary generated at {summary_path}")
                await self.progress_to_next_step()
                return None

            return None

        except Exception as e:
            logger.error(f"Error in handle_browser_task: {str(e)}")
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

    async def ensure_page_ready(self) -> None:
        """Ensure the page is ready for interaction"""
        if not self.browser_state.get("page_ready"):
            logger.warning("Page not ready, waiting for navigation to complete")
            try:
                # Try to verify page is loaded by checking current URL
                if self.browser_state.get("current_url"):
                    self.browser_state["page_ready"] = True
                    logger.info("Page is now ready for interaction")
                else:
                    logger.warning("No URL set, page cannot be ready")
                    await asyncio.sleep(1)  # Brief wait before retry
            except Exception as e:
                logger.error(f"Error checking page readiness: {str(e)}")
                await asyncio.sleep(1)  # Wait before retry

    async def execute_browser_step(self, step: str) -> None:
        """Execute a single browser step with validation and retries"""
        max_retries = 3
        retry_count = 0

        while retry_count < max_retries:
            try:
                # Validate page readiness
                if not self.browser_state["is_ready"]:
                    await self.ensure_page_ready()

                # Execute step based on type
                if step.startswith("click"):
                    await self.handle_click_action(step)
                elif step.startswith("type"):
                    await self.handle_input_action(step)
                elif step.startswith("wait"):
                    await self.handle_wait_action(step)
                elif step.startswith("review"):
                    await self.handle_review_action(step)
                else:
                    logger.warning(f"Unknown step type: {step}")

                # Step completed successfully
                return

            except Exception as e:
                retry_count += 1
                if retry_count >= max_retries:
                    logger.error(f"Step failed after {max_retries} attempts: {str(e)}")
                    raise
                await asyncio.sleep(1)  # Wait before retry

    async def handle_review_action(self, step: str) -> None:
        """Handle website review steps with improved validation"""
        try:
            # Extract review parameters
            params = self.parse_review_parameters(step)

            # Validate parameters
            if not params.get("element"):
                raise ValueError("Review step missing required element parameter")

            # Perform the review
            review_result = await self.browser.evaluate(
                f"""
                document.querySelector('{params["element"]}')?.textContent || ''
            """
            )

            # Store review result
            if review_result:
                self.memory.add_fact(
                    {
                        "type": "review_result",
                        "element": params["element"],
                        "content": review_result,
                        "timestamp": time.time(),
                    }
                )
            else:
                logger.warning(f"No content found for element: {params['element']}")

        except Exception as e:
            logger.error(f"Review action failed: {str(e)}")
            raise

    async def progress_to_next_step(self) -> None:
        """Safely progress to the next step or phase"""
        try:
            if not self.current_plan or "phases" not in self.current_plan:
                logger.warning("No valid plan exists, cannot progress")
                return

            if self.current_phase >= len(self.current_plan["phases"]):
                logger.warning("Current phase index out of range")
                return

            current_phase = self.current_plan["phases"][self.current_phase]
            if not current_phase.get("steps"):
                logger.warning("Current phase has no steps")
                return

            # Try to move to next step in current phase
            if self.current_step + 1 < len(current_phase["steps"]):
                self.current_step += 1
                logger.info(
                    f"Moved to step {self.current_step} in phase {self.current_phase}"
                )
            # If no more steps in current phase, try to move to next phase
            elif self.current_phase + 1 < len(self.current_plan["phases"]):
                self.current_phase += 1
                self.current_step = 0
                logger.info(
                    f"Moved to phase {self.current_phase}, step {self.current_step}"
                )
            else:
                logger.info("Reached end of all phases and steps")
                return

            # Update progress in todo list
            await self.update_todo_progress()

        except Exception as e:
            logger.error(f"Error progressing to next step: {str(e)}")
            # Ensure we don't leave the agent in an invalid state
            if self.current_step >= len(
                self.current_plan["phases"][self.current_phase]["steps"]
            ):
                self.current_step = (
                    len(self.current_plan["phases"][self.current_phase]["steps"]) - 1
                )

    def parse_review_parameters(self, step: str) -> Dict:
        """Parse review step parameters"""
        params = {"element": None}

        if "main content" in step.lower():
            params["element"] = "main,article,#content,.content,body"
        elif "header" in step.lower():
            params["element"] = "header,#header,.header"
        elif "navigation" in step.lower():
            params["element"] = "nav,#nav,.nav,.navigation"
        elif "footer" in step.lower():
            params["element"] = "footer,#footer,.footer"

        return params

    def generate_analysis_report(self) -> str:
        """Generate a detailed analysis report of the website"""
        return f"""# Analysis Report

## Overview
- URL: {self.browser_state.get('current_url')}
- Analysis Date: {time.strftime('%Y-%m-%d %H:%M:%S')}

## Content Extraction
- Status: {"Complete" if self.browser_state.get('content_extracted') else "Pending"}
- Extracted Elements:
  - Main Content: {"Yes" if self.browser_state.get('content_extracted') else "No"}

## Visual Documentation
- Status: {"Complete" if self.browser_state.get('screenshots_taken') else "Pending"}
- Screenshot Path: {os.path.join(config.workspace_root, "screenshots")}

## Structure Analysis
- Status: {"Complete" if self.browser_state.get('structure_analyzed') else "Pending"}
- Analysis Details: See `analysis` folder

## Summary
- Status: {"Complete" if self.browser_state.get('summary_complete') else "Pending"}
- Summary Path: {os.path.join(config.workspace_root, "summary.md")}

## Review Status
- Overall Status: {"Complete" if self.browser_state.get('analysis_complete') else "In Progress"}
- Pending Actions:
  - Content Extraction: {"No" if self.browser_state.get('content_extracted') else "Yes"}
  - Visual Documentation: {"No" if self.browser_state.get('screenshots_taken') else "Yes"}
  - Structure Analysis: {"No" if self.browser_state.get('structure_analyzed') else "Yes"}
  - Summary Creation: {"No" if self.browser_state.get('summary_complete') else "Yes"}
"""
