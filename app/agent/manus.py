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
            "complexity": self._assess_task_complexity(user_request),
            "optimization_targets": ["quality", "efficiency", "learning"],
            "reasoning_mode": "expert_level",
        }

        # Deep analysis using enhanced reasoning engine
        deep_analysis = await self.reasoning_engine.analyze_task_deeply(
            user_request, context
        )
        logger.info(
            f"🧠 DEEP ANALYSIS COMPLETED: {len(deep_analysis["reasoning_layers"])} reasoning layers applied"
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
            task_type=self._categorize_task(user_request), context=context
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
            "estimated_duration": self._estimate_duration(deep_analysis),
            "phases": self._create_enhanced_phases(
                deep_analysis, optimized_strategy, learned_strategy
            ),
            "success_criteria": self._define_success_criteria(deep_analysis),
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

    def _assess_task_complexity(self, task: str) -> str:
        """Assess task complexity for enhanced reasoning"""
        task_lower = task.lower()

        # High complexity indicators
        high_complexity_keywords = [
            "analyze",
            "comprehensive",
            "detailed",
            "complex",
            "optimize",
            "strategic",
            "multi-step",
            "research",
            "investigate",
            "evaluate",
        ]

        # Medium complexity indicators
        medium_complexity_keywords = [
            "review",
            "check",
            "compare",
            "summarize",
            "create",
            "build",
        ]

        high_count = sum(
            1 for keyword in high_complexity_keywords if keyword in task_lower
        )
        medium_count = sum(
            1 for keyword in medium_complexity_keywords if keyword in task_lower
        )

        if high_count >= 2 or len(task.split()) > 10:
            return "high"
        elif high_count >= 1 or medium_count >= 2:
            return "medium"
        else:
            return "simple"

    def _categorize_task(self, task: str) -> str:
        """Categorize task type for learning system"""
        task_lower = task.lower()

        if any(
            keyword in task_lower
            for keyword in ["browse", "website", "url", "web", "google"]
        ):
            return "web_browsing"
        elif any(
            keyword in task_lower for keyword in ["analyze", "research", "investigate"]
        ):
            return "analysis"
        elif any(keyword in task_lower for keyword in ["create", "build", "generate"]):
            return "creation"
        elif any(keyword in task_lower for keyword in ["review", "check", "verify"]):
            return "review"
        else:
            return "general"

    def _estimate_duration(self, analysis: Dict) -> str:
        """Estimate task duration based on analysis"""
        complexity = (
            analysis.get("reasoning_layers", {})
            .get("surface", {})
            .get("obvious_constraints", [])
        )

        if len(complexity) > 5:
            return "15-30 minutes"
        elif len(complexity) > 3:
            return "8-15 minutes"
        else:
            return "3-8 minutes"

    def _create_enhanced_phases(
        self, analysis: Dict, strategy: Dict, learned: Dict
    ) -> List[Dict]:
        """Create enhanced phases with deep reasoning integration"""
        phases = []

        # Phase 1: Deep Analysis and Planning
        phases.append(
            {
                "id": 1,
                "title": "Deep Analysis and Strategic Planning",
                "description": "Perform comprehensive analysis using multi-layered reasoning",
                "steps": [
                    "Apply expert-level reasoning analysis",
                    "Integrate learned insights and patterns",
                    "Optimize strategy based on analysis",
                    "Create detailed execution roadmap",
                ],
                "tools_needed": ["str_replace_editor", "reasoning_engine"],
                "success_criteria": "Deep analysis completed with optimization strategy",
                "reasoning_focus": "Multi-layered strategic analysis",
                "optimization_targets": ["quality", "efficiency"],
            }
        )

        # Phase 2: Optimized Execution
        phases.append(
            {
                "id": 2,
                "title": "Optimized Task Execution",
                "description": "Execute task using optimized approach with continuous learning",
                "steps": [
                    "Apply optimized execution strategy",
                    "Monitor quality and efficiency metrics",
                    "Adapt approach based on real-time feedback",
                    "Integrate learning insights during execution",
                ],
                "tools_needed": ["browser_use", "str_replace_editor"],
                "success_criteria": "Task executed with high quality and efficiency",
                "reasoning_focus": "Adaptive execution with optimization",
                "optimization_targets": ["performance", "learning"],
            }
        )

        # Phase 3: Quality Enhancement and Learning
        phases.append(
            {
                "id": 3,
                "title": "Quality Enhancement and Learning Integration",
                "description": "Enhance output quality and capture learning insights",
                "steps": [
                    "Apply quality enhancement techniques",
                    "Validate against success criteria",
                    "Extract learning insights for future optimization",
                    "Document optimization achievements",
                ],
                "tools_needed": ["str_replace_editor"],
                "success_criteria": "High-quality output with learning insights captured",
                "reasoning_focus": "Quality optimization and learning synthesis",
                "optimization_targets": ["quality", "learning", "future_improvement"],
            }
        )

        # Phase 4: Results Delivery and Optimization
        phases.append(
            {
                "id": 4,
                "title": "Results Delivery and Continuous Optimization",
                "description": "Deliver optimized results and apply continuous improvement",
                "steps": [
                    "Compile comprehensive results",
                    "Apply final optimization enhancements",
                    "Present results with reasoning transparency",
                    "Update learning system with experience",
                ],
                "tools_needed": ["str_replace_editor"],
                "success_criteria": "Optimized results delivered with learning integration",
                "reasoning_focus": "Results optimization and learning capture",
                "optimization_targets": ["delivery_quality", "system_learning"],
            }
        )

        return phases

    def _define_success_criteria(self, analysis: Dict) -> List[str]:
        """Define success criteria based on deep analysis"""
        criteria = [
            "Expert-level reasoning applied throughout execution",
            "Multi-layered analysis completed successfully",
            "Optimization strategies implemented effectively",
            "Learning insights integrated and captured",
            "High-quality output delivered to user",
            "Efficiency targets met or exceeded",
            "Continuous improvement demonstrated",
            "User satisfaction achieved",
        ]

        return criteria

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




