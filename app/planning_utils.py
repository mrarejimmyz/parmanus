"""Planning and task analysis utilities for the ParManus system."""

import re
from typing import Dict, List, Optional


class TaskAnalyzer:
    """Advanced task analysis and categorization."""

    @staticmethod
    def categorize_task(user_request: str) -> str:
        """Determine task type from user request with enhanced pattern matching."""
        request_lower = user_request.lower()

        # News and current information gathering tasks
        if any(
            x in request_lower
            for x in ["news", "current", "today", "latest", "recent", "headlines", "breaking"]
        ) or any(
            phrase in request_lower
            for phrase in ["top 10", "top ten", "what's happening", "current events"]
        ):
            return "news_gathering"

        # Website related tasks
        if any(
            x in request_lower
            for x in ["review", "analyze", "check", "browse", "visit"]
        ):
            if any(x in request_lower for x in ["http", "www", ".com", ".org", ".net"]):
                return "website_review"

        # File operations
        if any(
            x in request_lower
            for x in ["file", "read", "write", "create", "delete", "modify"]
        ):
            return "file_operation"

        # Code tasks
        if any(
            x in request_lower
            for x in ["code", "program", "script", "function", "class"]
        ):
            return "code_task"

        return "general_task"

    @staticmethod
    def assess_task_complexity(user_request: str) -> str:
        """Advanced complexity assessment based on multiple factors."""
        words = user_request.split()
        subtasks = len(re.findall(r"and|then|after|before|while", user_request.lower()))

        complexity_score = len(words) * 0.5 + subtasks * 2

        if complexity_score < 5:
            return "simple"
        elif complexity_score < 15:
            return "moderate"
        return "complex"

    @staticmethod
    def estimate_duration(analysis: Dict) -> str:
        """Estimate task duration based on comprehensive analysis."""
        complexity = analysis.get("complexity", "moderate")
        task_type = analysis.get("task_type", "general_task")

        base_durations = {"simple": 5, "moderate": 15, "complex": 30}

        type_multipliers = {
            "news_gathering": 1.3,
            "website_review": 1.2,
            "file_operation": 0.8,
            "code_task": 1.5,
            "general_task": 1.0,
        }

        base_time = base_durations.get(complexity, 15)
        multiplier = type_multipliers.get(task_type, 1.0)

        estimated_minutes = int(base_time * multiplier)
        return f"{max(5, estimated_minutes-5)}-{estimated_minutes+5} minutes"


class PlanGenerator:
    """Advanced plan generation with task-specific templates."""

    @staticmethod
    def create_enhanced_phases(
        context: Dict, strategy: Dict, insights: Dict
    ) -> List[Dict]:
        """Create execution phases based on task type and context."""
        task_type = context.get("task_type", "general_task")

        if task_type == "news_gathering":
            return [
                {
                    "id": 1,
                    "title": "Research Planning",
                    "description": "Plan news sources and research strategy",
                    "tools_needed": ["browser_use"],
                    "steps": ["Identify reliable news sources", "Plan research approach"],
                    "success_criteria": "Research strategy established",
                },
                {
                    "id": 2,
                    "title": "News Collection",
                    "description": "Browse news websites and gather current information",
                    "tools_needed": ["browser_use"],
                    "steps": [
                        "Visit major news websites",
                        "Extract current headlines",
                        "Gather detailed information",
                        "Verify information from multiple sources"
                    ],
                    "success_criteria": "Current news information collected",
                },
                {
                    "id": 3,
                    "title": "Content Creation",
                    "description": "Compile and format news into requested format",
                    "tools_needed": ["python_execute"],
                    "steps": ["Organize collected news", "Create formatted output", "Generate news_report.md"],
                    "success_criteria": "News report created with real current information",
                },
            ]

        elif task_type == "website_review":
            return [
                {
                    "id": 1,
                    "title": "Initial Access",
                    "description": "Navigate and verify website access",
                    "tools_needed": ["browser_use"],
                    "steps": ["Navigate to website", "Verify access"],
                    "success_criteria": "Successfully accessed target URL",
                },
                {
                    "id": 2,
                    "title": "Content Analysis",
                    "description": "Extract and analyze website content",
                    "tools_needed": ["browser_use"],
                    "steps": [
                        "Extract main content",
                        "Capture screenshots",
                        "Analyze page structure",
                    ],
                    "success_criteria": "Content analyzed and documented",
                },
                {
                    "id": 3,
                    "title": "Documentation",
                    "description": "Document findings and create report",
                    "tools_needed": ["python_execute"],
                    "steps": ["Generate analysis.md", "Create summary.md"],
                    "success_criteria": "Documentation complete",
                },
            ]

        elif task_type == "file_operation":
            return [
                {
                    "id": 1,
                    "title": "Preparation",
                    "description": "Analyze file operation requirements",
                    "tools_needed": ["python_execute"],
                    "steps": ["Validate paths", "Check permissions"],
                    "success_criteria": "Operation validated",
                },
                {
                    "id": 2,
                    "title": "Execution",
                    "description": "Perform file operations",
                    "tools_needed": ["python_execute"],
                    "steps": ["Execute operation", "Verify results"],
                    "success_criteria": "Operation completed",
                },
            ]

        # Default general task phases
        return [
            {
                "id": 1,
                "title": "Analysis",
                "description": "Analyze requirements",
                "tools_needed": ["python_execute"],
                "steps": ["Analyze request", "Create plan"],
                "success_criteria": "Requirements understood",
            },
            {
                "id": 2,
                "title": "Execution",
                "description": "Execute planned actions",
                "tools_needed": ["python_execute", "browser_use"],
                "steps": ["Execute actions", "Validate results"],
                "success_criteria": "Actions completed",
            },
        ]

    @staticmethod
    def define_success_criteria(analysis: Dict) -> List[str]:
        """Define detailed success criteria based on task analysis."""
        task_type = analysis.get("task_type", "general_task")
        complexity = analysis.get("complexity", "moderate")

        base_criteria = [
            "All planned actions completed successfully",
            "Results properly documented",
        ]

        if task_type == "website_review":
            base_criteria.extend(
                [
                    "Website content extracted and analyzed",
                    "Screenshots captured",
                    "Analysis report generated",
                ]
            )
        elif complexity == "complex":
            base_criteria.extend(
                ["Edge cases handled", "Performance optimized", "Error recovery tested"]
            )

        return base_criteria
