from typing import Dict, List

class TaskAnalyzer:
    """Utility class for analyzing task complexity, categorization, and duration."""

    @staticmethod
    def assess_task_complexity(task: str) -> str:
        """Assess task complexity for enhanced reasoning"""
        task_lower = task.lower()

        # High complexity indicators
        high_complexity_keywords = [
            "analyze", "comprehensive", "detailed", "complex", "optimize",
            "strategic", "multi-step", "research", "investigate", "evaluate",
        ]

        # Medium complexity indicators
        medium_complexity_keywords = [
            "review", "check", "compare", "summarize", "create", "build",
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

    @staticmethod
    def categorize_task(task: str) -> str:
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

    @staticmethod
    def estimate_duration(analysis: Dict) -> str:
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

class PlanGenerator:
    """Utility class for generating task phases and success criteria."""

    @staticmethod
    def create_enhanced_phases(
        analysis: Dict, strategy: Dict, learned: Dict
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

    @staticmethod
    def define_success_criteria(analysis: Dict) -> List[str]:
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


