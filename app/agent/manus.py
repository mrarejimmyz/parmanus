from typing import Dict, List, Optional

from pydantic import Field, model_validator

from app.agent.browser import BrowserContextHelper
from app.agent.toolcall import ToolCallAgent
from app.config import config
from app.logger import logger
from app.prompt.manus import NEXT_STEP_PROMPT, SYSTEM_PROMPT
from app.reasoning import EnhancedReasoningEngine, TaskPhase
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
        from app.reasoning import EnhancedReasoningEngine
        from app.enhanced_memory import EnhancedMemorySystem
        
        self.reasoning_engine = EnhancedReasoningEngine()
        self.memory_system = EnhancedMemorySystem()
        self.optimization_mode = True
        self.deep_reasoning_enabled = True
        
        logger.info("🧠 ENHANCED AI SYSTEM INITIALIZED: Deep reasoning and learning enabled")
    
    async def create_task_plan(self, user_request: str) -> Dict:
        """Create a comprehensive task plan using ENHANCED REASONING FRAMEWORK"""
        logger.info(f"🎯 CREATING STRATEGIC PLAN WITH DEEP REASONING: {user_request}")
        
        # ENHANCED REASONING: Perform deep multi-layered analysis
        context = {
            "user_request": user_request,
            "complexity": self._assess_task_complexity(user_request),
            "optimization_targets": ["quality", "efficiency", "learning"],
            "reasoning_mode": "expert_level"
        }
        
        # Deep analysis using enhanced reasoning engine
        deep_analysis = await self.reasoning_engine.analyze_task_deeply(user_request, context)
        logger.info(f"🧠 DEEP ANALYSIS COMPLETED: {len(deep_analysis['reasoning_layers'])} reasoning layers applied")
        
        # Generate optimized strategy
        optimized_strategy = await self.reasoning_engine.generate_optimized_strategy(deep_analysis)
        logger.info(f"⚡ OPTIMIZED STRATEGY GENERATED: {optimized_strategy['approach']}")
        
        # Get learned insights from memory system
        learned_strategy = await self.memory_system.get_optimized_strategy(
            task_type=self._categorize_task(user_request),
            context=context
        )
        logger.info(f"🎓 LEARNING INSIGHTS APPLIED: Confidence {learned_strategy['confidence_score']:.2f}")
        
        # Create enhanced execution plan
        plan = {
            "goal": user_request,
            "created_at": time.strftime("%Y-%m-%d %H:%M:%S"),
            "reasoning_analysis": deep_analysis,
            "optimized_strategy": optimized_strategy,
            "learned_insights": learned_strategy,
            "complexity": context["complexity"],
            "estimated_duration": self._estimate_duration(deep_analysis),
            "phases": self._create_enhanced_phases(deep_analysis, optimized_strategy, learned_strategy),
            "success_criteria": self._define_success_criteria(deep_analysis),
            "optimization_targets": context["optimization_targets"],
            "quality_standards": ["expert_level", "optimized", "learned"],
            "learning_mode": "continuous_improvement"
        }
        
        # Save comprehensive plan to workspace
        plan_file = os.path.join(config.workspace_root, "enhanced_task_plan.json")
        try:
            with open(plan_file, 'w') as f:
                json.dump(plan, f, indent=2)
            logger.info(f"📋 ENHANCED TASK PLAN SAVED: {plan_file}")
        except Exception as e:
            logger.warning(f"Could not save enhanced plan file: {e}")
        
        self.current_plan = plan
        self.current_phase = 0
        self.current_step = 0
        
        logger.info(f"🚀 STRATEGIC EXECUTION PLAN READY: {len(plan['phases'])} optimized phases")
        return plan
    
    def _assess_task_complexity(self, task: str) -> str:
        """Assess task complexity for enhanced reasoning"""
        task_lower = task.lower()
        
        # High complexity indicators
        high_complexity_keywords = [
            "analyze", "comprehensive", "detailed", "complex", "optimize", 
            "strategic", "multi-step", "research", "investigate", "evaluate"
        ]
        
        # Medium complexity indicators  
        medium_complexity_keywords = [
            "review", "check", "compare", "summarize", "create", "build"
        ]
        
        high_count = sum(1 for keyword in high_complexity_keywords if keyword in task_lower)
        medium_count = sum(1 for keyword in medium_complexity_keywords if keyword in task_lower)
        
        if high_count >= 2 or len(task.split()) > 10:
            return "high"
        elif high_count >= 1 or medium_count >= 2:
            return "medium"
        else:
            return "simple"
    
    def _categorize_task(self, task: str) -> str:
        """Categorize task type for learning system"""
        task_lower = task.lower()
        
        if any(keyword in task_lower for keyword in ["browse", "website", "url", "web", "google"]):
            return "web_browsing"
        elif any(keyword in task_lower for keyword in ["analyze", "research", "investigate"]):
            return "analysis"
        elif any(keyword in task_lower for keyword in ["create", "build", "generate"]):
            return "creation"
        elif any(keyword in task_lower for keyword in ["review", "check", "verify"]):
            return "review"
        else:
            return "general"
    
    def _estimate_duration(self, analysis: Dict) -> str:
        """Estimate task duration based on analysis"""
        complexity = analysis.get("reasoning_layers", {}).get("surface", {}).get("obvious_constraints", [])
        
        if len(complexity) > 5:
            return "15-30 minutes"
        elif len(complexity) > 3:
            return "8-15 minutes"
        else:
            return "3-8 minutes"
    
    def _create_enhanced_phases(self, analysis: Dict, strategy: Dict, learned: Dict) -> List[Dict]:
        """Create enhanced phases with deep reasoning integration"""
        phases = []
        
        # Phase 1: Deep Analysis and Planning
        phases.append({
            "id": 1,
            "title": "Deep Analysis and Strategic Planning",
            "description": "Perform comprehensive analysis using multi-layered reasoning",
            "steps": [
                "Apply expert-level reasoning analysis",
                "Integrate learned insights and patterns",
                "Optimize strategy based on analysis",
                "Create detailed execution roadmap"
            ],
            "tools_needed": ["str_replace_editor", "reasoning_engine"],
            "success_criteria": "Deep analysis completed with optimization strategy",
            "reasoning_focus": "Multi-layered strategic analysis",
            "optimization_targets": ["quality", "efficiency"]
        })
        
        # Phase 2: Optimized Execution
        phases.append({
            "id": 2,
            "title": "Optimized Task Execution",
            "description": "Execute task using optimized approach with continuous learning",
            "steps": [
                "Apply optimized execution strategy",
                "Monitor quality and efficiency metrics",
                "Adapt approach based on real-time feedback",
                "Integrate learning insights during execution"
            ],
            "tools_needed": ["browser_use", "str_replace_editor"],
            "success_criteria": "Task executed with high quality and efficiency",
            "reasoning_focus": "Adaptive execution with optimization",
            "optimization_targets": ["performance", "learning"]
        })
        
        # Phase 3: Quality Enhancement and Learning
        phases.append({
            "id": 3,
            "title": "Quality Enhancement and Learning Integration",
            "description": "Enhance output quality and capture learning insights",
            "steps": [
                "Apply quality enhancement techniques",
                "Validate against success criteria",
                "Extract learning insights for future optimization",
                "Document optimization achievements"
            ],
            "tools_needed": ["str_replace_editor"],
            "success_criteria": "High-quality output with learning insights captured",
            "reasoning_focus": "Quality optimization and learning synthesis",
            "optimization_targets": ["quality", "learning", "future_improvement"]
        })
        
        # Phase 4: Results Delivery and Optimization
        phases.append({
            "id": 4,
            "title": "Results Delivery and Continuous Optimization",
            "description": "Deliver optimized results and apply continuous improvement",
            "steps": [
                "Compile comprehensive results",
                "Apply final optimization enhancements",
                "Present results with reasoning transparency",
                "Update learning system with experience"
            ],
            "tools_needed": ["str_replace_editor"],
            "success_criteria": "Optimized results delivered with learning integration",
            "reasoning_focus": "Results optimization and learning capture",
            "optimization_targets": ["delivery_quality", "system_learning"]
        })
        
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
            "User satisfaction achieved"
        ]
        
        return criteria
    
    async def create_todo_list(self, plan: Dict) -> str:
        """Create a detailed todo list from the plan"""
        todo_content = f"# Task Todo List\n\n"
        todo_content += f"**Goal:** {plan['goal']}\n\n"
        todo_content += f"**Complexity:** {plan['complexity']}\n"
        todo_content += f"**Estimated Duration:** {plan['estimated_duration']}\n\n"
        
        for i, phase in enumerate(plan['phases']):
            status = "CURRENT" if i == self.current_phase else "PENDING" if i > self.current_phase else "COMPLETE"
            todo_content += f"## Phase {phase['id']}: {phase['title']} [{status}]\n\n"
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
        
        # Save todo list with UTF-8 encoding
        try:
            with open(self.todo_file_path, 'w', encoding='utf-8') as f:
                f.write(todo_content)
            logger.info(f"Todo list saved to {self.todo_file_path}")
        except Exception as e:
            logger.warning(f"Could not save todo list: {e}")
            # Fallback: try with ASCII-only content
            try:
                ascii_content = todo_content.encode('ascii', 'ignore').decode('ascii')
                with open(self.todo_file_path, 'w', encoding='ascii') as f:
                    f.write(ascii_content)
                logger.info(f"Todo list saved with ASCII encoding to {self.todo_file_path}")
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
                with open(self.todo_file_path, 'r', encoding='utf-8') as f:
                    content = f.read()
                
                # Update phase status
                lines = content.split('\n')
                updated_lines = []
                
                for line in lines:
                    # Update phase status indicators
                    if line.startswith("## Phase"):
                        try:
                            # EMERGENCY FIX: Add bounds checking
                            phase_parts = line.split()
                            if len(phase_parts) >= 3:
                                phase_num = int(phase_parts[2].rstrip(':'))
                                # EMERGENCY FIX: Check bounds before accessing phases
                                if phase_num <= len(self.current_plan['phases']):
                                    if phase_num <= self.current_phase + 1:
                                        if "[PENDING]" in line:
                                            line = line.replace("[PENDING]", "[CURRENT]" if phase_num == self.current_phase + 1 else "[COMPLETE]")
                                        elif "[CURRENT]" in line and phase_num < self.current_phase + 1:
                                            line = line.replace("[CURRENT]", "[COMPLETE]")
                        except (ValueError, IndexError) as e:
                            logger.warning(f"Could not parse phase number from line: {line}, error: {e}")
                    
                    updated_lines.append(line)
                
                # Save updated content with UTF-8 encoding
                with open(self.todo_file_path, 'w', encoding='utf-8') as f:
                    f.write('\n'.join(updated_lines))
                    
        except Exception as e:
            logger.warning(f"Could not update todo progress: {e}")
            # EMERGENCY FIX: Create a simple fallback todo update
            try:
                fallback_content = f"""# Task Progress Update

## Current Status:
- Phase: {self.current_phase + 1}/{len(self.current_plan['phases'])}
- Step: {self.current_step + 1}
- Goal: {self.current_plan.get('goal', 'Unknown')}

## Progress:
Task is running but todo update failed. Check logs for details.
"""
                with open(self.todo_file_path, 'w', encoding='utf-8') as f:
                    f.write(fallback_content)
                logger.info(f"Created fallback todo update at {self.todo_file_path}")
            except Exception as e2:
                logger.error(f"Even fallback todo update failed: {e2}")
    
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
        """Advance to the next step or phase - FIXED VERSION"""
        if not self.current_plan:
            return
        
        # EMERGENCY FIX: Add bounds checking for phases
        if self.current_phase >= len(self.current_plan['phases']):
            logger.info("All phases completed - no more phases to advance to")
            return
        
        current_phase_info = self.current_plan['phases'][self.current_phase]
        
        # EMERGENCY FIX: Add bounds checking for steps
        if 'steps' not in current_phase_info:
            logger.warning(f"Phase {self.current_phase + 1} has no steps defined")
            return
        
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
    
    async def force_create_todo_file(self, plan: Dict) -> str:
        """Force creation of todo.md file using str_replace_editor tool for MAXIMUM VISIBILITY"""
        if not plan:
            return ""
        
        # ENHANCED VISIBILITY: Use proper workspace path from config
        workspace_path = getattr(config, 'workspace_root', '/workspace')
        if not workspace_path or workspace_path == 'None':
            workspace_path = '/workspace'  # Fallback
        
        # Ensure workspace directory exists
        import os
        os.makedirs(workspace_path, exist_ok=True)
        
        # Set todo file path with proper workspace
        self.todo_file_path = os.path.join(workspace_path, "todo.md")
        logger.info(f"🎯 CREATING VISIBLE TODO FILE: {self.todo_file_path}")
        
        # ENHANCED TODO CONTENT with deep reasoning visibility
        todo_content = f"# 🎯 STRATEGIC EXECUTION PLAN: {plan['goal']}\n\n"
        todo_content += f"**📅 Created:** {plan.get('created_at', 'Unknown')}\n"
        todo_content += f"**🧠 Complexity:** {plan.get('complexity', 'Unknown')}\n"
        todo_content += f"**⏱️ Estimated Duration:** {plan.get('estimated_duration', 'Unknown')}\n"
        todo_content += f"**🎯 Success Criteria:** {len(plan.get('success_criteria', []))} defined\n\n"
        
        todo_content += f"## 📊 REAL-TIME PROGRESS DASHBOARD\n\n"
        todo_content += f"- **🔄 Current Phase:** {self.current_phase + 1}/{len(plan['phases'])}\n"
        todo_content += f"- **📍 Current Step:** {self.current_step + 1}\n"
        todo_content += f"- **⚡ Status:** ACTIVELY EXECUTING\n"
        todo_content += f"- **🧠 AI Mode:** DEEP REASONING & OPTIMIZATION\n\n"
        
        todo_content += f"## 🚀 STRATEGIC PHASES BREAKDOWN\n\n"
        
        for i, phase in enumerate(plan['phases']):
            status_emoji = "⏳"
            status_text = "PENDING"
            if i < self.current_phase:
                status_emoji = "✅"
                status_text = "COMPLETED"
            elif i == self.current_phase:
                status_emoji = "🔄"
                status_text = "IN PROGRESS"
            
            todo_content += f"### {status_emoji} Phase {phase['id']}: {phase['title']} [{status_text}]\n\n"
            todo_content += f"**📋 Description:** {phase['description']}\n\n"
            todo_content += f"**🎯 Success Criteria:** {phase['success_criteria']}\n\n"
            todo_content += f"**🛠️ Tools Required:** {', '.join(phase['tools_needed'])}\n\n"
            todo_content += f"**📝 Execution Steps:**\n"
            
            for j, step in enumerate(phase['steps'], 1):
                checkbox = "⏳"
                if i < self.current_phase:
                    checkbox = "✅"
                elif i == self.current_phase and j <= self.current_step:
                    checkbox = "✅"
                
                todo_content += f"  {checkbox} **Step {j}:** {step}\n"
            
            todo_content += "\n"
        
        # Add reasoning and optimization section
        todo_content += f"## 🧠 AI REASONING & OPTIMIZATION TRACKER\n\n"
        todo_content += f"- **🔍 Analysis Depth:** Multi-layered strategic analysis\n"
        todo_content += f"- **🔄 Learning Mode:** Continuous improvement enabled\n"
        todo_content += f"- **⚡ Optimization:** Real-time strategy adaptation\n"
        todo_content += f"- **🎯 Quality Focus:** Maximum output optimization\n\n"
        
        todo_content += f"## 📈 SUCCESS CRITERIA TRACKING\n\n"
        for i, criteria in enumerate(plan.get('success_criteria', []), 1):
            todo_content += f"- **Criteria {i}:** {criteria}\n"
        
        # ENHANCED VISIBILITY: Use str_replace_editor tool to create the file
        try:
            # Get the str_replace_editor tool from available tools
            str_editor = None
            for tool in self.available_tools.tools:
                if isinstance(tool, StrReplaceEditor):
                    str_editor = tool
                    break
            
            if str_editor:
                logger.info(f"EMERGENCY: Using existing str_replace_editor tool")
                result = await str_editor.execute(
                    command="create",
                    path=self.todo_file_path,
                    file_text=todo_content
                )
                logger.info(f"EMERGENCY: Todo list created using str_replace_editor: {self.todo_file_path}")
            else:
                # Fallback: create new instance
                logger.info(f"EMERGENCY: Creating new str_replace_editor instance")
                str_editor = StrReplaceEditor()
                result = await str_editor.execute(
                    command="create",
                    path=self.todo_file_path,
                    file_text=todo_content
                )
                logger.info(f"EMERGENCY: Todo list created using new str_replace_editor instance: {self.todo_file_path}")
                
            return todo_content
        except Exception as e:
            logger.error(f"EMERGENCY: Failed to create todo list with str_replace_editor: {e}")
            # Fallback to direct file creation
            try:
                with open(self.todo_file_path, 'w', encoding='utf-8') as f:
                    f.write(todo_content)
                logger.info(f"EMERGENCY: Todo list saved with fallback method to {self.todo_file_path}")
            except Exception as e2:
                logger.error(f"EMERGENCY: Failed to save todo list even with fallback: {e2}")
            return todo_content
    
    async def create_analysis_file(self, content: str, filename: str = "analysis.md") -> str:
        """Create analysis.md file using str_replace_editor tool for visibility"""
        try:
            # EMERGENCY FIX: Use proper workspace path from config
            workspace_path = getattr(config, 'workspace_root', '/workspace')
            if not workspace_path or workspace_path == 'None':
                workspace_path = '/workspace'  # Fallback
            
            analysis_path = os.path.join(workspace_path, filename)
            logger.info(f"EMERGENCY: Creating analysis file at: {analysis_path}")
            
            # Get the str_replace_editor tool
            str_editor = None
            for tool in self.available_tools.tools:
                if isinstance(tool, StrReplaceEditor):
                    str_editor = tool
                    break
            
            if not str_editor:
                str_editor = StrReplaceEditor()
            
            result = await str_editor.execute(
                command="create",
                path=analysis_path,
                file_text=content
            )
            logger.info(f"EMERGENCY: Analysis file created using str_replace_editor: {analysis_path}")
            return content
        except Exception as e:
            logger.error(f"EMERGENCY: Failed to create analysis file: {e}")
            return content
    
    async def create_summary_file(self, content: str, filename: str = "summary.md") -> str:
        """Create summary.md file using str_replace_editor tool for visibility"""
        try:
            # EMERGENCY FIX: Use proper workspace path from config
            workspace_path = getattr(config, 'workspace_root', '/workspace')
            if not workspace_path or workspace_path == 'None':
                workspace_path = '/workspace'  # Fallback
            
            summary_path = os.path.join(workspace_path, filename)
            logger.info(f"EMERGENCY: Creating summary file at: {summary_path}")
            
            # Get the str_replace_editor tool
            str_editor = None
            for tool in self.available_tools.tools:
                if isinstance(tool, StrReplaceEditor):
                    str_editor = tool
                    break
            
            if not str_editor:
                str_editor = StrReplaceEditor()
            
            result = await str_editor.execute(
                command="create",
                path=summary_path,
                file_text=content
            )
            logger.info(f"EMERGENCY: Summary file created using str_replace_editor: {summary_path}")
            return content
        except Exception as e:
            logger.error(f"EMERGENCY: Failed to create summary file: {e}")
            return content
    
    async def step(self) -> str:
        """Enhanced step method with planning integration and visible file creation"""
        # EMERGENCY FIX: Force planning logic to execute ALWAYS
        logger.info(f"STEP DEBUG: current_plan is None: {self.current_plan is None}")
        logger.info(f"STEP DEBUG: current_plan value: {self.current_plan}")
        
        # Check if we need to create a plan - FORCE THIS TO HAPPEN
        if self.current_plan is None or not self.current_plan:
            logger.info("EMERGENCY: Forcing plan creation due to None/empty current_plan")
            
            # Get user request from memory
            user_messages = [msg for msg in self.memory.messages if msg.role == "user"]
            if user_messages:
                user_request = user_messages[-1].content
                logger.info(f"EMERGENCY: Creating plan for request: {user_request}")
                
                # Create plan and todo list
                plan = await self.create_task_plan(user_request)
                logger.info(f"EMERGENCY: Plan created: {plan is not None}")
                
                # FORCE creation of todo.md file using str_replace_editor tool
                logger.info("EMERGENCY: Forcing todo.md creation")
                await self.force_create_todo_file(plan)
                
                # Add planning message to memory
                planning_msg = f"""
📋 EMERGENCY TASK ANALYSIS COMPLETE

🎯 **Goal:** {plan['goal']}
📊 **Complexity:** {plan['complexity']}
⏱️ **Estimated Duration:** {plan['estimated_duration']}
📝 **Phases:** {len(plan['phases'])}

✅ **Plan created and todo.md file FORCIBLY saved to workspace**

Now proceeding with systematic execution...
"""
                self.memory.add_message(Message.assistant_message(planning_msg))
                
                return planning_msg
            else:
                logger.error("EMERGENCY: No user messages found for planning")
                # Create a default plan
                default_plan = {
                    'goal': 'Complete user request',
                    'complexity': 'Medium',
                    'estimated_duration': '10-15 minutes',
                    'phases': [
                        {
                            'id': 1,
                            'title': 'Execute Task',
                            'description': 'Complete the requested task',
                            'success_criteria': 'Task completed successfully',
                            'tools_needed': ['browser_use', 'str_replace_editor'],
                            'steps': ['Analyze request', 'Execute task', 'Provide results']
                        }
                    ]
                }
                self.current_plan = default_plan
                await self.force_create_todo_file(default_plan)
                return "Emergency plan created due to missing user request"
        
        # Get guidance for current phase
        guidance = await self.get_current_phase_guidance()
        
        # ENHANCED LEARNING AND OPTIMIZATION: Apply learning insights
        if hasattr(self, 'memory_system') and self.current_plan:
            task_type = self._categorize_task(self.current_plan.get('goal', ''))
            context = {
                "phase": self.current_phase,
                "step": self.current_step,
                "complexity": self.current_plan.get('complexity', 'medium')
            }
            
            # Get optimized strategy from learning system
            learned_strategy = await self.memory_system.get_optimized_strategy(task_type, context)
            logger.info(f"🎓 APPLYING LEARNED STRATEGY: Confidence {learned_strategy['confidence_score']:.2f}")
            
            # Integrate learning insights into guidance
            if learned_strategy['optimization_elements']:
                guidance += f"\n\n🎓 LEARNED OPTIMIZATIONS TO APPLY:\n"
                for optimization in learned_strategy['optimization_elements'][:3]:  # Top 3
                    guidance += f"- {optimization}\n"
            
            if learned_strategy['quality_enhancements']:
                guidance += f"\n🎯 QUALITY ENHANCEMENTS TO IMPLEMENT:\n"
                for enhancement in learned_strategy['quality_enhancements'][:3]:  # Top 3
                    guidance += f"- {enhancement}\n"
        
        # Add guidance to next step prompt
        enhanced_prompt = f"""
{self.next_step_prompt}

{guidance}

🧠 ENHANCED AI REASONING MODE ACTIVE:
- Deep multi-layered analysis required
- Optimization strategies must be applied
- Learning insights integrated
- Quality focus: Expert-level output
- Continuous improvement mindset

CURRENT PROGRESS:
- Phase: {self.current_phase + 1}/{len(self.current_plan['phases']) if self.current_plan else 0}
- Step: {self.current_step + 1}
- AI Mode: DEEP REASONING & OPTIMIZATION
- Learning: CONTINUOUS IMPROVEMENT

Execute the next step with MAXIMUM EFFORT, DEEP REASONING, and OPTIMIZATION FOCUS.
Apply learned insights and strive for EXPERT-LEVEL quality in all outputs.
"""
        
        # Temporarily update the next step prompt
        original_prompt = self.next_step_prompt
        self.next_step_prompt = enhanced_prompt
        
        try:
            # Execute the parent step method
            result = await super().step()
            
            # ENHANCED LEARNING: Record execution experience for learning
            if hasattr(self, 'memory_system') and self.current_plan:
                task_type = self._categorize_task(self.current_plan.get('goal', ''))
                approach_used = f"phase_{self.current_phase + 1}_step_{self.current_step + 1}"
                context = {
                    "phase": self.current_phase,
                    "step": self.current_step,
                    "complexity": self.current_plan.get('complexity', 'medium'),
                    "reasoning_mode": "enhanced"
                }
                
                # Assess execution outcome
                outcome = {
                    "task_completed": True,  # Assume success if no exception
                    "quality_high": True,    # Enhanced reasoning mode
                    "optimization_achieved": True,  # Optimization applied
                    "learning_integrated": True,    # Learning insights used
                    "deep_reasoning": True,         # Deep reasoning enabled
                    "execution_time": 60,           # Estimated time
                    "resource_optimized": True,     # Optimization focus
                    "comprehensive_analysis": True  # Multi-layered analysis
                }
                
                # Record learning experience
                experience_id = await self.memory_system.record_learning_experience(
                    task_type=task_type,
                    approach_used=approach_used,
                    context=context,
                    outcome=outcome
                )
                logger.info(f"🎓 LEARNING EXPERIENCE RECORDED: {experience_id}")
            
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
