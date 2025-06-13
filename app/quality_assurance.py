"""
Success Validation and Quality Assurance System for ParManusAI
"""

import asyncio
import json
import time
from typing import Dict, List, Optional, Any, Tuple
from enum import Enum
from dataclasses import dataclass

# Use standard logging instead of loguru for compatibility
import logging
logger = logging.getLogger(__name__)


class ValidationLevel(Enum):
    """Different levels of validation rigor"""
    BASIC = "basic"
    STANDARD = "standard"
    COMPREHENSIVE = "comprehensive"


class QualityMetric(Enum):
    """Quality metrics for task completion assessment"""
    COMPLETENESS = "completeness"
    ACCURACY = "accuracy"
    RELEVANCE = "relevance"
    DEPTH = "depth"
    CLARITY = "clarity"
    ACTIONABILITY = "actionability"


@dataclass
class ValidationResult:
    """Result of a validation check"""
    passed: bool
    score: float  # 0.0 to 1.0
    details: str
    suggestions: List[str]
    critical_issues: List[str]


@dataclass
class QualityAssessment:
    """Comprehensive quality assessment of task completion"""
    overall_score: float
    metric_scores: Dict[QualityMetric, float]
    validation_results: Dict[str, ValidationResult]
    completion_status: str
    recommendations: List[str]
    critical_issues: List[str]


class SuccessValidator:
    """
    Validates task completion success based on multiple criteria
    """
    
    def __init__(self, llm=None):
        self.llm = llm
        self.validation_history: List[Dict] = []
        
    async def validate_task_completion(
        self, 
        task_goal: str, 
        execution_results: List[Dict], 
        validation_level: ValidationLevel = ValidationLevel.STANDARD
    ) -> QualityAssessment:
        """
        Comprehensive validation of task completion
        """
        logger.info(f"Validating task completion for: {task_goal}")
        
        # Initialize assessment
        assessment = QualityAssessment(
            overall_score=0.0,
            metric_scores={},
            validation_results={},
            completion_status="unknown",
            recommendations=[],
            critical_issues=[]
        )
        
        # Run different validation checks based on level
        if validation_level in [ValidationLevel.BASIC, ValidationLevel.STANDARD, ValidationLevel.COMPREHENSIVE]:
            assessment.validation_results["basic_completion"] = await self._validate_basic_completion(
                task_goal, execution_results
            )
            
        if validation_level in [ValidationLevel.STANDARD, ValidationLevel.COMPREHENSIVE]:
            assessment.validation_results["content_quality"] = await self._validate_content_quality(
                task_goal, execution_results
            )
            assessment.validation_results["goal_alignment"] = await self._validate_goal_alignment(
                task_goal, execution_results
            )
            
        if validation_level == ValidationLevel.COMPREHENSIVE:
            assessment.validation_results["depth_analysis"] = await self._validate_depth_and_insight(
                task_goal, execution_results
            )
            assessment.validation_results["actionability"] = await self._validate_actionability(
                task_goal, execution_results
            )
        
        # Calculate quality metrics
        assessment.metric_scores = await self._calculate_quality_metrics(
            task_goal, execution_results, assessment.validation_results
        )
        
        # Calculate overall score
        assessment.overall_score = sum(assessment.metric_scores.values()) / len(assessment.metric_scores)
        
        # Determine completion status
        assessment.completion_status = self._determine_completion_status(assessment)
        
        # Generate recommendations
        assessment.recommendations = self._generate_recommendations(assessment)
        
        # Identify critical issues
        assessment.critical_issues = self._identify_critical_issues(assessment)
        
        # Record validation
        self._record_validation(task_goal, assessment)
        
        return assessment
    
    async def _validate_basic_completion(self, task_goal: str, execution_results: List[Dict]) -> ValidationResult:
        """
        Basic validation: Did the task produce any meaningful output?
        """
        try:
            # Check if there are any results
            if not execution_results:
                return ValidationResult(
                    passed=False,
                    score=0.0,
                    details="No execution results found",
                    suggestions=["Ensure task execution produces output"],
                    critical_issues=["Task produced no results"]
                )
            
            # Check for successful tool executions
            successful_executions = [r for r in execution_results if r.get("success", False)]
            success_rate = len(successful_executions) / len(execution_results)
            
            # Check for meaningful content
            has_content = any(
                len(str(r.get("output", ""))) > 50 for r in execution_results
            )
            
            # Check for error patterns
            error_count = sum(1 for r in execution_results if r.get("error"))
            error_rate = error_count / len(execution_results)
            
            # Calculate score
            score = (success_rate * 0.4) + (0.6 if has_content else 0.0) - (error_rate * 0.2)
            score = max(0.0, min(1.0, score))
            
            passed = score >= 0.6 and has_content
            
            details = f"Success rate: {success_rate:.1%}, Content found: {has_content}, Error rate: {error_rate:.1%}"
            
            suggestions = []
            if success_rate < 0.7:
                suggestions.append("Improve tool execution success rate")
            if not has_content:
                suggestions.append("Ensure meaningful content is generated")
            if error_rate > 0.3:
                suggestions.append("Reduce error frequency")
            
            critical_issues = []
            if not has_content:
                critical_issues.append("No meaningful content generated")
            if error_rate > 0.5:
                critical_issues.append("High error rate indicates systematic issues")
            
            return ValidationResult(
                passed=passed,
                score=score,
                details=details,
                suggestions=suggestions,
                critical_issues=critical_issues
            )
            
        except Exception as e:
            logger.error(f"Basic completion validation failed: {e}")
            return ValidationResult(
                passed=False,
                score=0.0,
                details=f"Validation error: {str(e)}",
                suggestions=["Fix validation system error"],
                critical_issues=["Validation system failure"]
            )
    
    async def _validate_content_quality(self, task_goal: str, execution_results: List[Dict]) -> ValidationResult:
        """
        Validate the quality of generated content
        """
        try:
            # Extract all content from results
            all_content = []
            for result in execution_results:
                output = result.get("output", "")
                if output and len(str(output)) > 20:
                    all_content.append(str(output))
            
            if not all_content:
                return ValidationResult(
                    passed=False,
                    score=0.0,
                    details="No substantial content found",
                    suggestions=["Generate more substantial content"],
                    critical_issues=["Insufficient content generated"]
                )
            
            # Analyze content quality
            combined_content = "\n".join(all_content)
            
            # Basic quality checks
            word_count = len(combined_content.split())
            has_structure = any(marker in combined_content.lower() for marker in 
                              ["analysis", "summary", "overview", "conclusion", "findings"])
            has_details = word_count > 100
            has_formatting = any(marker in combined_content for marker in 
                               ["**", "*", "##", "###", "- ", "1.", "2."])
            
            # Calculate quality score
            quality_factors = [
                (word_count > 50, 0.2),
                (word_count > 200, 0.2),
                (has_structure, 0.3),
                (has_details, 0.2),
                (has_formatting, 0.1)
            ]
            
            score = sum(weight for condition, weight in quality_factors if condition)
            
            passed = score >= 0.6
            
            details = f"Word count: {word_count}, Structure: {has_structure}, Details: {has_details}, Formatting: {has_formatting}"
            
            suggestions = []
            if word_count < 100:
                suggestions.append("Provide more detailed content")
            if not has_structure:
                suggestions.append("Add clear structure and organization")
            if not has_formatting:
                suggestions.append("Improve content formatting and readability")
            
            critical_issues = []
            if word_count < 50:
                critical_issues.append("Content too brief to be useful")
            
            return ValidationResult(
                passed=passed,
                score=score,
                details=details,
                suggestions=suggestions,
                critical_issues=critical_issues
            )
            
        except Exception as e:
            logger.error(f"Content quality validation failed: {e}")
            return ValidationResult(
                passed=False,
                score=0.0,
                details=f"Content validation error: {str(e)}",
                suggestions=["Fix content validation system"],
                critical_issues=["Content validation failure"]
            )
    
    async def _validate_goal_alignment(self, task_goal: str, execution_results: List[Dict]) -> ValidationResult:
        """
        Validate that results align with the original task goal
        """
        try:
            if not self.llm:
                # Fallback validation without LLM
                return await self._validate_goal_alignment_basic(task_goal, execution_results)
            
            # Extract content for analysis
            content_summary = self._extract_content_summary(execution_results)
            
            # Use LLM to assess goal alignment
            messages = [
                {
                    "role": "system",
                    "content": """You are an expert task completion assessor. Evaluate how well the execution results align with the original task goal.
                    
                    Provide a score from 0.0 to 1.0 and detailed analysis of:
                    1. Goal fulfillment completeness
                    2. Relevance of results to the goal
                    3. Any missing elements
                    4. Quality of goal achievement
                    
                    Respond in JSON format with: {"score": float, "analysis": string, "missing_elements": [list], "strengths": [list]}"""
                },
                {
                    "role": "user",
                    "content": f"""
                    TASK GOAL: {task_goal}
                    
                    EXECUTION RESULTS SUMMARY:
                    {content_summary}
                    
                    Please assess how well these results fulfill the original task goal.
                    """
                }
            ]
            
            response = await self.llm.ask(messages)
            
            try:
                assessment = json.loads(response)
                score = float(assessment.get("score", 0.0))
                analysis = assessment.get("analysis", "No analysis provided")
                missing_elements = assessment.get("missing_elements", [])
                strengths = assessment.get("strengths", [])
                
                passed = score >= 0.7
                
                suggestions = []
                if missing_elements:
                    suggestions.extend([f"Address missing element: {elem}" for elem in missing_elements])
                if score < 0.7:
                    suggestions.append("Improve goal alignment and completeness")
                
                critical_issues = []
                if score < 0.4:
                    critical_issues.append("Poor alignment with task goal")
                if len(missing_elements) > 2:
                    critical_issues.append("Multiple critical elements missing")
                
                return ValidationResult(
                    passed=passed,
                    score=score,
                    details=f"Goal alignment analysis: {analysis}",
                    suggestions=suggestions,
                    critical_issues=critical_issues
                )
                
            except (json.JSONDecodeError, ValueError) as e:
                logger.warning(f"Failed to parse LLM assessment: {e}")
                return await self._validate_goal_alignment_basic(task_goal, execution_results)
            
        except Exception as e:
            logger.error(f"Goal alignment validation failed: {e}")
            return await self._validate_goal_alignment_basic(task_goal, execution_results)
    
    async def _validate_goal_alignment_basic(self, task_goal: str, execution_results: List[Dict]) -> ValidationResult:
        """
        Basic goal alignment validation without LLM
        """
        # Extract key terms from goal
        goal_terms = set(task_goal.lower().split())
        goal_terms = {term for term in goal_terms if len(term) > 3}
        
        # Extract content
        all_content = " ".join(str(r.get("output", "")) for r in execution_results).lower()
        
        # Check term coverage
        covered_terms = sum(1 for term in goal_terms if term in all_content)
        coverage_ratio = covered_terms / len(goal_terms) if goal_terms else 0.0
        
        # Basic heuristics
        has_analysis = any(word in all_content for word in ["analysis", "review", "assessment", "evaluation"])
        has_specific_content = len(all_content) > 200
        
        score = (coverage_ratio * 0.6) + (0.2 if has_analysis else 0.0) + (0.2 if has_specific_content else 0.0)
        passed = score >= 0.6
        
        return ValidationResult(
            passed=passed,
            score=score,
            details=f"Term coverage: {coverage_ratio:.1%}, Has analysis: {has_analysis}",
            suggestions=["Improve goal-specific content"] if score < 0.6 else [],
            critical_issues=["Poor goal alignment"] if score < 0.4 else []
        )
    
    async def _validate_depth_and_insight(self, task_goal: str, execution_results: List[Dict]) -> ValidationResult:
        """
        Validate depth of analysis and insights provided
        """
        content = self._extract_content_summary(execution_results)
        
        # Check for depth indicators
        depth_indicators = [
            ("detailed analysis", 0.2),
            ("technical details", 0.15),
            ("specific examples", 0.15),
            ("recommendations", 0.2),
            ("insights", 0.15),
            ("implications", 0.15)
        ]
        
        score = 0.0
        found_indicators = []
        
        for indicator, weight in depth_indicators:
            if indicator.replace(" ", "") in content.lower().replace(" ", ""):
                score += weight
                found_indicators.append(indicator)
        
        # Additional depth checks
        word_count = len(content.split())
        if word_count > 500:
            score += 0.1
        if word_count > 1000:
            score += 0.1
        
        passed = score >= 0.6
        
        suggestions = []
        if score < 0.6:
            missing = [indicator for indicator, _ in depth_indicators if indicator not in found_indicators]
            suggestions.extend([f"Add {indicator}" for indicator in missing[:3]])
        
        return ValidationResult(
            passed=passed,
            score=min(1.0, score),
            details=f"Depth indicators found: {', '.join(found_indicators)}",
            suggestions=suggestions,
            critical_issues=["Lacks analytical depth"] if score < 0.3 else []
        )
    
    async def _validate_actionability(self, task_goal: str, execution_results: List[Dict]) -> ValidationResult:
        """
        Validate if results provide actionable insights or information
        """
        content = self._extract_content_summary(execution_results)
        
        # Check for actionable elements
        actionable_indicators = [
            "recommendation", "suggest", "should", "could", "next steps",
            "action", "implement", "improve", "consider", "strategy"
        ]
        
        actionable_count = sum(1 for indicator in actionable_indicators if indicator in content.lower())
        actionable_ratio = min(1.0, actionable_count / 5)  # Normalize to 0-1
        
        # Check for specific actionable content
        has_recommendations = "recommendation" in content.lower()
        has_next_steps = "next step" in content.lower()
        has_specific_actions = any(word in content.lower() for word in ["implement", "execute", "apply"])
        
        score = (actionable_ratio * 0.5) + (0.2 if has_recommendations else 0.0) + \
                (0.2 if has_next_steps else 0.0) + (0.1 if has_specific_actions else 0.0)
        
        passed = score >= 0.5
        
        suggestions = []
        if not has_recommendations:
            suggestions.append("Add specific recommendations")
        if not has_next_steps:
            suggestions.append("Include clear next steps")
        if score < 0.5:
            suggestions.append("Make results more actionable")
        
        return ValidationResult(
            passed=passed,
            score=score,
            details=f"Actionable elements: {actionable_count}, Recommendations: {has_recommendations}",
            suggestions=suggestions,
            critical_issues=["Results not actionable"] if score < 0.3 else []
        )
    
    async def _calculate_quality_metrics(
        self, 
        task_goal: str, 
        execution_results: List[Dict], 
        validation_results: Dict[str, ValidationResult]
    ) -> Dict[QualityMetric, float]:
        """
        Calculate quality metrics based on validation results
        """
        metrics = {}
        
        # Map validation results to quality metrics
        if "basic_completion" in validation_results:
            metrics[QualityMetric.COMPLETENESS] = validation_results["basic_completion"].score
        
        if "content_quality" in validation_results:
            metrics[QualityMetric.CLARITY] = validation_results["content_quality"].score
        
        if "goal_alignment" in validation_results:
            metrics[QualityMetric.RELEVANCE] = validation_results["goal_alignment"].score
            metrics[QualityMetric.ACCURACY] = validation_results["goal_alignment"].score
        
        if "depth_analysis" in validation_results:
            metrics[QualityMetric.DEPTH] = validation_results["depth_analysis"].score
        
        if "actionability" in validation_results:
            metrics[QualityMetric.ACTIONABILITY] = validation_results["actionability"].score
        
        # Fill in missing metrics with default values
        for metric in QualityMetric:
            if metric not in metrics:
                metrics[metric] = 0.5  # Neutral score for unmeasured metrics
        
        return metrics
    
    def _determine_completion_status(self, assessment: QualityAssessment) -> str:
        """
        Determine overall completion status
        """
        if assessment.overall_score >= 0.8:
            return "excellent"
        elif assessment.overall_score >= 0.7:
            return "good"
        elif assessment.overall_score >= 0.6:
            return "satisfactory"
        elif assessment.overall_score >= 0.4:
            return "needs_improvement"
        else:
            return "poor"
    
    def _generate_recommendations(self, assessment: QualityAssessment) -> List[str]:
        """
        Generate improvement recommendations based on assessment
        """
        recommendations = []
        
        # Collect suggestions from all validation results
        for validation_result in assessment.validation_results.values():
            recommendations.extend(validation_result.suggestions)
        
        # Add metric-specific recommendations
        for metric, score in assessment.metric_scores.items():
            if score < 0.6:
                if metric == QualityMetric.COMPLETENESS:
                    recommendations.append("Ensure all task components are addressed")
                elif metric == QualityMetric.DEPTH:
                    recommendations.append("Provide more detailed analysis and insights")
                elif metric == QualityMetric.ACTIONABILITY:
                    recommendations.append("Include specific recommendations and next steps")
        
        # Remove duplicates and limit to top recommendations
        unique_recommendations = list(dict.fromkeys(recommendations))
        return unique_recommendations[:5]
    
    def _identify_critical_issues(self, assessment: QualityAssessment) -> List[str]:
        """
        Identify critical issues that must be addressed
        """
        critical_issues = []
        
        # Collect critical issues from validation results
        for validation_result in assessment.validation_results.values():
            critical_issues.extend(validation_result.critical_issues)
        
        # Add overall critical issues
        if assessment.overall_score < 0.4:
            critical_issues.append("Overall task completion quality is poor")
        
        # Remove duplicates
        return list(dict.fromkeys(critical_issues))
    
    def _extract_content_summary(self, execution_results: List[Dict]) -> str:
        """
        Extract and summarize content from execution results
        """
        all_content = []
        for result in execution_results:
            output = result.get("output", "")
            if output and len(str(output)) > 20:
                all_content.append(str(output))
        
        combined = "\n".join(all_content)
        
        # Truncate if too long
        if len(combined) > 2000:
            combined = combined[:2000] + "..."
        
        return combined
    
    def _record_validation(self, task_goal: str, assessment: QualityAssessment) -> None:
        """
        Record validation results for learning and improvement
        """
        record = {
            "timestamp": time.time(),
            "task_goal": task_goal,
            "overall_score": assessment.overall_score,
            "completion_status": assessment.completion_status,
            "metric_scores": {metric.value: score for metric, score in assessment.metric_scores.items()},
            "critical_issues_count": len(assessment.critical_issues),
            "recommendations_count": len(assessment.recommendations)
        }
        
        self.validation_history.append(record)
        
        # Keep only last 100 validations
        if len(self.validation_history) > 100:
            self.validation_history = self.validation_history[-100:]
        
        logger.info(f"Validation recorded: {assessment.completion_status} ({assessment.overall_score:.2f})")


class QualityAssuranceManager:
    """
    Manages overall quality assurance for the ParManusAI system
    """
    
    def __init__(self, llm=None):
        self.validator = SuccessValidator(llm)
        self.quality_standards = self._initialize_quality_standards()
        
    def _initialize_quality_standards(self) -> Dict[str, float]:
        """
        Initialize quality standards for different task types
        """
        return {
            "website_analysis": 0.75,
            "research_task": 0.80,
            "content_generation": 0.70,
            "data_analysis": 0.85,
            "general_task": 0.70
        }
    
    async def assess_task_quality(
        self, 
        task_goal: str, 
        execution_results: List[Dict],
        task_type: str = "general_task"
    ) -> Tuple[bool, QualityAssessment]:
        """
        Assess if task meets quality standards
        """
        # Determine validation level based on task type
        validation_level = ValidationLevel.STANDARD
        if task_type in ["research_task", "data_analysis"]:
            validation_level = ValidationLevel.COMPREHENSIVE
        
        # Run validation
        assessment = await self.validator.validate_task_completion(
            task_goal, execution_results, validation_level
        )
        
        # Check against quality standards
        required_score = self.quality_standards.get(task_type, 0.70)
        meets_standards = assessment.overall_score >= required_score
        
        logger.info(f"Quality assessment: {assessment.overall_score:.2f} (required: {required_score:.2f})")
        
        return meets_standards, assessment
    
    def get_quality_report(self) -> Dict[str, Any]:
        """
        Generate a quality report based on validation history
        """
        if not self.validator.validation_history:
            return {"message": "No validation history available"}
        
        history = self.validator.validation_history
        recent_validations = history[-10:]  # Last 10 validations
        
        avg_score = sum(v["overall_score"] for v in recent_validations) / len(recent_validations)
        
        status_distribution = {}
        for validation in recent_validations:
            status = validation["completion_status"]
            status_distribution[status] = status_distribution.get(status, 0) + 1
        
        return {
            "total_validations": len(history),
            "recent_average_score": avg_score,
            "status_distribution": status_distribution,
            "quality_trend": "improving" if len(history) > 5 and 
                           sum(v["overall_score"] for v in history[-5:]) > 
                           sum(v["overall_score"] for v in history[-10:-5]) else "stable"
        }

