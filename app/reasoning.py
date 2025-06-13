"""
Enhanced Reasoning Framework for ParManusAI
Provides structured thinking and step-by-step execution guidance
"""

from typing import Dict, List, Optional, Any
from enum import Enum
import time

class ReasoningStep(Enum):
    """Types of reasoning steps"""
    ANALYZE = "analyze"
    PLAN = "plan"
    EXECUTE = "execute"
    VERIFY = "verify"
    ADAPT = "adapt"

class TaskPhase(Enum):
    """Task execution phases"""
    UNDERSTANDING = "understanding"
    PLANNING = "planning"
    EXECUTION = "execution"
    VERIFICATION = "verification"
    COMPLETION = "completion"

class ReasoningFramework:
    """
    Structured reasoning framework for systematic task execution
    """
    
    def __init__(self):
        self.current_phase = TaskPhase.UNDERSTANDING
        self.reasoning_history = []
        self.task_context = {}
        self.success_criteria = []
        
    def analyze_request(self, user_request: str) -> Dict[str, Any]:
        """Analyze user request systematically"""
        analysis = {
            "original_request": user_request,
            "task_type": self._identify_task_type(user_request),
            "complexity": self._assess_complexity(user_request),
            "required_tools": self._identify_required_tools(user_request),
            "success_criteria": self._define_success_criteria(user_request),
            "potential_challenges": self._identify_challenges(user_request),
            "estimated_phases": self._estimate_phases(user_request)
        }
        
        self.task_context = analysis
        self.current_phase = TaskPhase.PLANNING
        
        return analysis
    
    def _identify_task_type(self, request: str) -> str:
        """Identify the type of task being requested"""
        request_lower = request.lower()
        
        if any(word in request_lower for word in ["review", "analyze", "examine", "check"]):
            return "analysis"
        elif any(word in request_lower for word in ["create", "build", "make", "generate"]):
            return "creation"
        elif any(word in request_lower for word in ["fix", "debug", "solve", "repair"]):
            return "problem_solving"
        elif any(word in request_lower for word in ["find", "search", "lookup", "get"]):
            return "information_retrieval"
        elif any(word in request_lower for word in ["browse", "visit", "navigate", "website"]):
            return "web_interaction"
        else:
            return "general"
    
    def _assess_complexity(self, request: str) -> str:
        """Assess task complexity"""
        # Simple heuristics for complexity assessment
        word_count = len(request.split())
        has_multiple_actions = len([w for w in request.lower().split() if w in ["and", "then", "also", "plus"]]) > 0
        
        if word_count < 5 and not has_multiple_actions:
            return "simple"
        elif word_count < 15 and not has_multiple_actions:
            return "moderate"
        else:
            return "complex"
    
    def _identify_required_tools(self, request: str) -> List[str]:
        """Identify tools likely needed for the task"""
        tools = []
        request_lower = request.lower()
        
        if any(word in request_lower for word in ["website", "url", "browse", "web", "page"]):
            tools.append("browser_use")
        if any(word in request_lower for word in ["code", "program", "script", "python"]):
            tools.append("python_execute")
        if any(word in request_lower for word in ["file", "write", "create", "edit", "save"]):
            tools.append("str_replace_editor")
        if any(word in request_lower for word in ["search", "find", "lookup"]):
            tools.append("web_search")
        
        return tools if tools else ["browser_use"]  # Default to browser if unclear
    
    def _define_success_criteria(self, request: str) -> List[str]:
        """Define what success looks like for this task"""
        task_type = self._identify_task_type(request)
        
        criteria = ["Task completed as requested", "User requirements met"]
        
        if task_type == "analysis":
            criteria.extend([
                "Comprehensive analysis provided",
                "Key insights identified",
                "Findings clearly presented"
            ])
        elif task_type == "web_interaction":
            criteria.extend([
                "Website successfully accessed",
                "Relevant information extracted",
                "Content properly analyzed"
            ])
        elif task_type == "creation":
            criteria.extend([
                "Deliverable created successfully",
                "Quality standards met",
                "Requirements fulfilled"
            ])
        
        return criteria
    
    def _identify_challenges(self, request: str) -> List[str]:
        """Identify potential challenges"""
        challenges = []
        request_lower = request.lower()
        
        if "website" in request_lower or "url" in request_lower:
            challenges.extend([
                "Website may be inaccessible",
                "Content may be dynamic or require JavaScript",
                "Rate limiting or blocking possible"
            ])
        
        if self._assess_complexity(request) == "complex":
            challenges.append("Task complexity may require multiple iterations")
        
        return challenges
    
    def _estimate_phases(self, request: str) -> int:
        """Estimate number of phases needed"""
        complexity = self._assess_complexity(request)
        task_type = self._identify_task_type(request)
        
        base_phases = {
            "simple": 3,
            "moderate": 4,
            "complex": 6
        }
        
        phases = base_phases.get(complexity, 4)
        
        # Adjust based on task type
        if task_type in ["analysis", "web_interaction"]:
            phases += 1  # Extra phase for thorough analysis
        
        return min(phases, 8)  # Cap at 8 phases
    
    def create_execution_plan(self, analysis: Dict[str, Any]) -> Dict[str, Any]:
        """Create detailed execution plan based on analysis"""
        phases = []
        
        # Phase 1: Understanding and Setup
        phases.append({
            "id": 1,
            "title": "Task Understanding and Setup",
            "description": "Analyze requirements and prepare for execution",
            "steps": [
                "Confirm understanding of user request",
                "Identify required tools and resources",
                "Set up workspace and files",
                "Create todo list for tracking"
            ],
            "tools_needed": ["str_replace_editor"],
            "success_criteria": "Clear understanding established and workspace ready"
        })
        
        # Generate task-specific phases based on analysis
        task_type = analysis.get("task_type", "general")
        
        if task_type == "web_interaction":
            phases.extend([
                {
                    "id": 2,
                    "title": "Website Access and Navigation",
                    "description": "Access the target website and navigate to relevant content",
                    "steps": [
                        "Navigate to the specified website",
                        "Verify page loads correctly",
                        "Identify key content areas",
                        "Check for any access issues"
                    ],
                    "tools_needed": ["browser_use"],
                    "success_criteria": "Website accessed and content visible"
                },
                {
                    "id": 3,
                    "title": "Content Analysis and Extraction",
                    "description": "Extract and analyze relevant content from the website",
                    "steps": [
                        "Extract main content from the page",
                        "Identify key information and features",
                        "Analyze structure and functionality",
                        "Document findings systematically"
                    ],
                    "tools_needed": ["browser_use", "str_replace_editor"],
                    "success_criteria": "Relevant content extracted and analyzed"
                }
            ])
        
        # Add verification and completion phases
        phases.extend([
            {
                "id": len(phases) + 1,
                "title": "Results Verification",
                "description": "Verify all requirements have been met",
                "steps": [
                    "Review completed work against requirements",
                    "Check all success criteria are met",
                    "Identify any gaps or issues",
                    "Make final adjustments if needed"
                ],
                "tools_needed": ["str_replace_editor"],
                "success_criteria": "All requirements verified as complete"
            },
            {
                "id": len(phases) + 2,
                "title": "Results Delivery",
                "description": "Compile and deliver final results to user",
                "steps": [
                    "Compile all findings and results",
                    "Create comprehensive summary",
                    "Present results clearly to user",
                    "Provide any additional recommendations"
                ],
                "tools_needed": ["str_replace_editor"],
                "success_criteria": "Results delivered and user satisfied"
            }
        ])
        
        plan = {
            "goal": analysis["original_request"],
            "phases": phases,
            "estimated_duration": f"{len(phases) * 2}-{len(phases) * 4} minutes",
            "complexity": analysis["complexity"],
            "success_criteria": analysis["success_criteria"]
        }
        
        self.current_phase = TaskPhase.EXECUTION
        return plan
    
    def get_next_step_guidance(self, current_step: int, total_steps: int, phase_info: Dict[str, Any]) -> str:
        """Provide guidance for the next step"""
        guidance = f"""
🎯 CURRENT STEP: {current_step}/{total_steps}

📋 PHASE: {phase_info['title']}
📝 DESCRIPTION: {phase_info['description']}

🔧 TOOLS AVAILABLE: {', '.join(phase_info['tools_needed'])}

✅ SUCCESS CRITERIA: {phase_info['success_criteria']}

📌 NEXT ACTIONS:
"""
        
        if current_step <= len(phase_info['steps']):
            current_step_desc = phase_info['steps'][current_step - 1]
            guidance += f"- Execute: {current_step_desc}\n"
            
            if current_step < len(phase_info['steps']):
                next_step_desc = phase_info['steps'][current_step]
                guidance += f"- Prepare for: {next_step_desc}\n"
        
        guidance += """
🧠 REASONING APPROACH:
1. Think through the step carefully
2. Choose the most appropriate tool
3. Execute with clear intent
4. Verify the result
5. Update progress and move forward

Remember: Quality over speed. Take time to do each step properly.
"""
        
        return guidance
    
    def should_adapt_plan(self, current_results: str, expected_outcome: str) -> bool:
        """Determine if the plan needs adaptation based on results"""
        # Simple heuristics for plan adaptation
        if "error" in current_results.lower() or "failed" in current_results.lower():
            return True
        
        if len(current_results) < 50:  # Very short results might indicate issues
            return True
        
        return False
    
    def adapt_plan(self, current_plan: Dict[str, Any], issue_description: str) -> Dict[str, Any]:
        """Adapt the plan based on encountered issues"""
        # Add an adaptation phase
        adaptation_phase = {
            "id": len(current_plan["phases"]) + 1,
            "title": "Plan Adaptation",
            "description": f"Address issue: {issue_description}",
            "steps": [
                "Analyze the encountered issue",
                "Identify alternative approaches",
                "Implement the best alternative",
                "Verify the solution works"
            ],
            "tools_needed": ["browser_use", "str_replace_editor"],
            "success_criteria": "Issue resolved and task back on track"
        }
        
        # Insert before the final phases
        current_plan["phases"].insert(-2, adaptation_phase)
        
        # Renumber phases
        for i, phase in enumerate(current_plan["phases"]):
            phase["id"] = i + 1
        
        return current_plan

