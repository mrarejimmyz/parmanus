"""
Enhanced Planning Agent for ParManusAI
Provides strategic task planning, todo list creation, and execution guidance
"""

import json
import time
import os
from typing import Dict, List, Optional, Any
from pydantic import Field

from app.agent.base import BaseAgent
from app.logger import logger
from app.schema import Message, AgentState
from app.tool import ToolCollection
from app.tool.python_execute import PythonExecute # Use PythonExecute for file operations


class TaskPlan:
    """Represents a structured task plan with phases and steps"""
    
    def __init__(self, goal: str, phases: List[Dict[str, Any]]):
        self.goal = goal
        self.phases = phases
        self.current_phase = 0
        self.created_at = time.time()
        self.status = "active"
    
    def get_current_phase(self) -> Optional[Dict[str, Any]]:
        """Get the current phase being executed"""
        if self.current_phase < len(self.phases):
            return self.phases[self.current_phase]
        return None
    
    def advance_phase(self) -> bool:
        """Move to the next phase"""
        if self.current_phase < len(self.phases) - 1:
            self.current_phase += 1
            return True
        return False
    
    def is_complete(self) -> bool:
        """Check if all phases are complete"""
        return self.current_phase >= len(self.phases)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert plan to dictionary for serialization"""
        return {
            "goal": self.goal,
            "phases": self.phases,
            "current_phase": self.current_phase,
            "created_at": self.created_at,
            "status": self.status
        }


class PlannerAgent(BaseAgent):
    """
    Enhanced Planning Agent that creates structured task plans and todo lists
    """
    
    name = "planner"
    description = "Strategic planning agent that breaks down complex tasks into manageable steps"
    
    # Planning-specific tools
    available_tools: ToolCollection = Field(
        default_factory=lambda: ToolCollection(
            PythonExecute(), # Use PythonExecute for file operations
        )
    )
    
    # Current task plan
    current_plan: Optional[TaskPlan] = None
    
    # Planning configuration
    max_phases: int = 10
    max_steps_per_phase: int = 5
    
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.system_prompt = self._get_planning_system_prompt()
        self.next_step_prompt = self._get_planning_next_step_prompt()
    
    def _get_planning_system_prompt(self) -> str:
        """Get the system prompt for planning tasks"""
        return """You are an expert strategic planning agent. Your role is to:

1. ANALYZE the user\"s request thoroughly
2. CREATE a structured plan with clear phases and steps
3. GENERATE a detailed todo list for execution
4. GUIDE the execution process step by step

For every task, you must:
- Break it down into logical phases (2-10 phases)
- Create specific, actionable steps for each phase
- Identify required tools and resources
- Establish success criteria for each step
- Create a comprehensive todo list

Always think strategically and plan before acting. Never jump directly to tool usage without proper planning."""
    
    def _get_planning_next_step_prompt(self) -> str:
        """Get the next step prompt for planning"""
        return """Based on the current task and any previous progress:

1. If no plan exists: CREATE a comprehensive task plan
2. If plan exists: EXECUTE the next step in the current phase
3. If phase complete: ADVANCE to the next phase
4. If all phases complete: SUMMARIZE results and terminate

Always update the todo list to reflect current progress and next steps."""
    
    async def create_task_plan(self, user_request: str) -> TaskPlan:
        """Create a structured task plan for the user request"""
        logger.info(f"Creating task plan for: {user_request}")
        
        # Analyze the request and create phases
        planning_prompt = f"""
Analyze this user request and create a comprehensive task plan:

REQUEST: {user_request}

Create a structured plan with:
1. Clear goal statement
2. 3-8 logical phases
3. Specific steps for each phase
4. Required tools/resources
5. Success criteria

Format as JSON:
{{
    "goal": "Clear goal statement",
    "phases": [
        {{
            "id": 1,
            "title": "Phase title",
            "description": "What this phase accomplishes",
            "steps": [
                "Specific actionable step 1",
                "Specific actionable step 2"
            ],
            "tools_needed": ["tool1", "tool2"],
            "success_criteria": "How to know this phase is complete"
        }}
    ]
}}
"""
        
        # Get planning response
        response = await self.llm.ask(
            messages=[Message.user_message(planning_prompt)],
            system_msgs=[Message.system_message(self.system_prompt)]
        )
        
        try:
            # Parse the JSON response
            plan_data = json.loads(response.content)
            plan = TaskPlan(plan_data["goal"], plan_data["phases"])
            self.current_plan = plan
            
            logger.info(f"Created plan with {len(plan.phases)} phases")
            return plan
            
        except (json.JSONDecodeError, KeyError) as e:
            logger.error(f"Failed to parse planning response: {e}")
            # Create a fallback simple plan
            fallback_plan = TaskPlan(
                goal=user_request,
                phases=[
                    {
                        "id": 1,
                        "title": "Analyze Request",
                        "description": "Understand what the user wants",
                        "steps": ["Break down the request", "Identify key requirements"],
                        "tools_needed": ["analysis"],
                        "success_criteria": "Clear understanding of requirements"
                    },
                    {
                        "id": 2,
                        "title": "Execute Task",
                        "description": "Perform the main task",
                        "steps": ["Use appropriate tools", "Complete the task"],
                        "tools_needed": ["browser_use", "python_execute"],
                        "success_criteria": "Task completed successfully"
                    },
                    {
                        "id": 3,
                        "title": "Deliver Results",
                        "description": "Provide results to user",
                        "steps": ["Summarize findings", "Present results"],
                        "tools_needed": ["python_execute"],
                        "success_criteria": "User receives complete results"
                    }
                ]
            )
            self.current_plan = fallback_plan
            return fallback_plan
    
    async def create_todo_list(self, plan: TaskPlan) -> str:
        """Create a detailed todo list from the task plan"""
        todo_content = f"# Task Todo List\n\n"
        todo_content += f"**Goal:** {plan.goal}\n\n"
        todo_content += f"**Created:** {time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(plan.created_at))}\n\n"
        
        for i, phase in enumerate(plan.phases):
            status = "🔄 CURRENT" if i == plan.current_phase else "⏳ PENDING" if i > plan.current_phase else "✅ COMPLETE"
            todo_content += f"## Phase {phase["id"]}: {phase["title"]} {status}\n\n"
            todo_content += f"**Description:** {phase["description"]}\n\n"
            todo_content += f"**Success Criteria:** {phase["success_criteria"]}\n\n"
            todo_content += f"**Tools Needed:** {", ".join(phase["tools_needed"])}\n\n"
            todo_content += f"**Steps:**\n"
            
            for j, step in enumerate(phase["steps"], 1):
                checkbox = "- [ ]" if i >= plan.current_phase else "- [x]"
                todo_content += f"{checkbox} {step}\n"
            
            todo_content += "\n"
        
        # Save todo list to file using direct file write
        todo_file = os.path.join(self.config.workspace_root, "todo.md")
        try:
            with open(todo_file, "w", encoding="utf-8") as f:
                f.write(todo_content)
            logger.info(f"Todo list saved to {todo_file}")
        except Exception as e:
            logger.error(f"Failed to save todo list: {e}")
        
        return todo_content
    
    async def update_todo_progress(self, phase_id: int, step_index: int, completed: bool = True):
        """Update todo list to mark steps as completed"""
        todo_file = os.path.join(self.config.workspace_root, "todo.md")
        try:
            # Read current todo
            if not os.path.exists(todo_file):
                logger.warning(f"Todo file not found at {todo_file}")
                return

            with open(todo_file, "r", encoding="utf-8") as f:
                current_content = f.read()
            
            # Update the specific step
            lines = current_content.split("\n")
            updated_lines = []
            current_phase_in_file = None
            step_count = 0
            
            for line in lines:
                if line.startswith(f"## Phase {phase_id}:"):
                    current_phase_in_file = phase_id
                    step_count = 0
                elif line.startswith("## Phase"):
                    current_phase_in_file = None
                    step_count = 0
                elif current_phase_in_file == phase_id and line.strip().startswith("- ["):
                    if step_count == step_index:
                        if completed:
                            line = line.replace("- [ ]", "- [x]")
                        else:
                            line = line.replace("- [x]", "- [ ]")
                    step_count += 1
                
                updated_lines.append(line)
            
            # Save updated todo
            updated_content = "\n".join(updated_lines)
            with open(todo_file, "w", encoding="utf-8") as f:
                f.write(updated_content)
            logger.info(f"Todo list updated at {todo_file}")
            
        except Exception as e:
            logger.error(f"Failed to update todo progress: {e}")
    
    async def get_next_action(self) -> Optional[Dict[str, Any]]:
        """Get the next action to take based on current plan"""
        if not self.current_plan:
            return None
        
        current_phase = self.current_plan.get_current_phase()
        if not current_phase:
            return {"action": "complete", "message": "All phases completed"}
        
        return {
            "action": "execute_phase",
            "phase": current_phase,
            "phase_number": self.current_plan.current_phase + 1,
            "total_phases": len(self.current_plan.phases)
        }
    
    async def step(self) -> str:
        """Execute one planning step"""
        # If no plan exists, create one
        if not self.current_plan:
            # Get the user request from memory
            user_messages = [msg for msg in self.memory.messages if msg.role == "user"]
            if not user_messages:
                return "No user request found to plan for"
            
            user_request = user_messages[-1].content
            plan = await self.create_task_plan(user_request)
            todo_list = await self.create_todo_list(plan)
            
            return f"Created comprehensive task plan with {len(plan.phases)} phases. Todo list saved to workspace."
        
        # Get next action
        next_action = await self.get_next_action()
        if not next_action:
            return "Planning complete - no further actions needed"
        
        if next_action["action"] == "complete":
            self.state = AgentState.FINISHED
            return "Task planning and execution complete"
        
        # Execute current phase
        current_phase = next_action["phase"]
        phase_num = next_action["phase_number"]
        total_phases = next_action["total_phases"]
        
        guidance = f"""
CURRENT PHASE: {phase_num}/{total_phases} - {current_phase["title"]}

DESCRIPTION: {current_phase["description"]}

STEPS TO COMPLETE:
{chr(10).join(f"- {step}" for step in current_phase["steps"]) }

TOOLS AVAILABLE: {", ".join(current_phase["tools_needed"]) }

SUCCESS CRITERIA: {current_phase["success_criteria"]}

Execute the steps for this phase systematically. Update the todo list as you complete each step.
"""
        
        return guidance
    
    @classmethod
    async def create(cls, **kwargs):
        """Factory method to create and properly initialize a PlannerAgent instance."""
        instance = cls(**kwargs)
        return instance


