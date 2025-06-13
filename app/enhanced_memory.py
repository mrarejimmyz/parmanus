"""
Enhanced Memory and Learning System for ParManusAI
"""

import json
import time
import asyncio
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass, asdict
from collections import defaultdict
import hashlib

@dataclass
class LearningExperience:
    """Represents a learning experience with context and outcomes"""
    task_type: str
    approach_used: str
    context: Dict
    outcome: Dict
    success_score: float
    efficiency_score: float
    quality_score: float
    lessons_learned: List[str]
    optimization_insights: List[str]
    timestamp: float

@dataclass
class StrategyPattern:
    """Represents a successful strategy pattern"""
    pattern_id: str
    pattern_type: str
    context_conditions: Dict
    strategy_elements: List[str]
    success_rate: float
    efficiency_gain: float
    quality_improvement: float
    usage_count: int
    last_used: float

@dataclass
class OptimizationInsight:
    """Represents an optimization insight learned from experience"""
    insight_id: str
    insight_type: str
    context: Dict
    optimization_target: str
    improvement_method: str
    expected_gain: float
    validation_count: int
    confidence: float

class EnhancedMemorySystem:
    """Enhanced memory system with learning and optimization capabilities"""
    
    def __init__(self):
        self.learning_experiences: List[LearningExperience] = []
        self.strategy_patterns: Dict[str, StrategyPattern] = {}
        self.optimization_insights: Dict[str, OptimizationInsight] = {}
        self.performance_trends: Dict[str, List[float]] = defaultdict(list)
        self.learning_metrics = {
            "total_experiences": 0,
            "successful_patterns": 0,
            "optimization_insights": 0,
            "average_improvement": 0.0
        }
        
    async def record_learning_experience(self, 
                                       task_type: str,
                                       approach_used: str, 
                                       context: Dict,
                                       outcome: Dict) -> str:
        """Record a learning experience for future optimization"""
        
        # Calculate performance scores
        success_score = self._calculate_success_score(outcome)
        efficiency_score = self._calculate_efficiency_score(outcome)
        quality_score = self._calculate_quality_score(outcome)
        
        # Extract lessons learned
        lessons_learned = self._extract_lessons(approach_used, outcome)
        optimization_insights = self._extract_optimization_insights(context, outcome)
        
        # Create learning experience
        experience = LearningExperience(
            task_type=task_type,
            approach_used=approach_used,
            context=context,
            outcome=outcome,
            success_score=success_score,
            efficiency_score=efficiency_score,
            quality_score=quality_score,
            lessons_learned=lessons_learned,
            optimization_insights=optimization_insights,
            timestamp=time.time()
        )
        
        self.learning_experiences.append(experience)
        self.learning_metrics["total_experiences"] += 1
        
        # Update performance trends
        self.performance_trends["success"].append(success_score)
        self.performance_trends["efficiency"].append(efficiency_score)
        self.performance_trends["quality"].append(quality_score)
        
        # Analyze for patterns and insights
        await self._analyze_for_patterns(experience)
        await self._extract_optimization_insights_advanced(experience)
        
        return f"experience_{len(self.learning_experiences)}"
    
    async def get_optimized_strategy(self, task_type: str, context: Dict) -> Dict:
        """Get optimized strategy based on learned experiences"""
        
        # Find relevant experiences
        relevant_experiences = self._find_relevant_experiences(task_type, context)
        
        # Find applicable patterns
        applicable_patterns = self._find_applicable_patterns(task_type, context)
        
        # Get optimization insights
        relevant_insights = self._get_relevant_insights(task_type, context)
        
        # Synthesize optimized strategy
        strategy = {
            "base_approach": self._determine_base_approach(relevant_experiences),
            "optimization_elements": self._compile_optimizations(applicable_patterns, relevant_insights),
            "quality_enhancements": self._suggest_quality_enhancements(relevant_experiences),
            "efficiency_improvements": self._suggest_efficiency_improvements(relevant_experiences),
            "learning_adaptations": self._suggest_learning_adaptations(relevant_experiences),
            "confidence_score": self._calculate_strategy_confidence(relevant_experiences, applicable_patterns),
            "expected_improvements": self._predict_improvements(applicable_patterns, relevant_insights)
        }
        
        return strategy
    
    async def learn_from_feedback(self, experience_id: str, feedback: Dict) -> None:
        """Learn from feedback on previous experiences"""
        
        # Find the experience
        experience_index = int(experience_id.split("_")[1]) - 1
        if experience_index < len(self.learning_experiences):
            experience = self.learning_experiences[experience_index]
            
            # Update experience with feedback
            experience.outcome.update(feedback)
            
            # Recalculate scores
            experience.success_score = self._calculate_success_score(experience.outcome)
            experience.efficiency_score = self._calculate_efficiency_score(experience.outcome)
            experience.quality_score = self._calculate_quality_score(experience.outcome)
            
            # Re-analyze for patterns
            await self._analyze_for_patterns(experience)
    
    async def get_learning_insights(self) -> Dict:
        """Get comprehensive learning insights and recommendations"""
        
        insights = {
            "performance_trends": self._analyze_performance_trends(),
            "successful_patterns": self._get_top_patterns(),
            "optimization_opportunities": self._identify_optimization_opportunities(),
            "learning_recommendations": self._generate_learning_recommendations(),
            "strategy_adaptations": self._suggest_strategy_adaptations(),
            "quality_improvements": self._recommend_quality_improvements(),
            "efficiency_gains": self._recommend_efficiency_gains()
        }
        
        return insights
    
    def _calculate_success_score(self, outcome: Dict) -> float:
        """Calculate success score from outcome"""
        # Enhanced success calculation
        base_score = 0.7  # Base success
        
        if outcome.get("task_completed", False):
            base_score += 0.2
        if outcome.get("quality_high", False):
            base_score += 0.1
        if outcome.get("user_satisfied", False):
            base_score += 0.1
        if outcome.get("optimization_achieved", False):
            base_score += 0.1
            
        return min(base_score, 1.0)
    
    def _calculate_efficiency_score(self, outcome: Dict) -> float:
        """Calculate efficiency score from outcome"""
        base_score = 0.6
        
        execution_time = outcome.get("execution_time", 100)
        if execution_time < 50:
            base_score += 0.3
        elif execution_time < 80:
            base_score += 0.2
        elif execution_time < 100:
            base_score += 0.1
            
        if outcome.get("resource_optimized", False):
            base_score += 0.1
            
        return min(base_score, 1.0)
    
    def _calculate_quality_score(self, outcome: Dict) -> float:
        """Calculate quality score from outcome"""
        base_score = 0.6
        
        if outcome.get("comprehensive_analysis", False):
            base_score += 0.15
        if outcome.get("deep_reasoning", False):
            base_score += 0.15
        if outcome.get("optimization_applied", False):
            base_score += 0.1
        if outcome.get("learning_integrated", False):
            base_score += 0.1
            
        return min(base_score, 1.0)
    
    def _extract_lessons(self, approach: str, outcome: Dict) -> List[str]:
        """Extract lessons learned from approach and outcome"""
        lessons = []
        
        if outcome.get("success_score", 0) > 0.8:
            lessons.append(f"Approach '{approach}' highly effective for this context")
        
        if outcome.get("efficiency_score", 0) > 0.8:
            lessons.append(f"Approach '{approach}' demonstrates high efficiency")
            
        if outcome.get("quality_score", 0) > 0.8:
            lessons.append(f"Approach '{approach}' produces high quality results")
            
        # Add specific lessons based on outcome details
        if outcome.get("deep_reasoning", False):
            lessons.append("Deep reasoning significantly improves outcomes")
            
        if outcome.get("optimization_applied", False):
            lessons.append("Optimization strategies enhance performance")
            
        return lessons
    
    def _extract_optimization_insights(self, context: Dict, outcome: Dict) -> List[str]:
        """Extract optimization insights from context and outcome"""
        insights = []
        
        if outcome.get("optimization_achieved", False):
            insights.append("Multi-layered optimization approach successful")
            
        if outcome.get("learning_integrated", False):
            insights.append("Learning integration improves results")
            
        if context.get("complexity") == "high" and outcome.get("success_score", 0) > 0.8:
            insights.append("Deep analysis essential for complex tasks")
            
        return insights
    
    async def _analyze_for_patterns(self, experience: LearningExperience) -> None:
        """Analyze experience for successful patterns"""
        
        if experience.success_score > 0.8:
            # Create pattern ID
            pattern_context = {
                "task_type": experience.task_type,
                "context_complexity": experience.context.get("complexity", "medium")
            }
            pattern_id = self._generate_pattern_id(pattern_context)
            
            if pattern_id in self.strategy_patterns:
                # Update existing pattern
                pattern = self.strategy_patterns[pattern_id]
                pattern.usage_count += 1
                pattern.success_rate = (pattern.success_rate * (pattern.usage_count - 1) + experience.success_score) / pattern.usage_count
                pattern.last_used = experience.timestamp
            else:
                # Create new pattern
                pattern = StrategyPattern(
                    pattern_id=pattern_id,
                    pattern_type="successful_approach",
                    context_conditions=pattern_context,
                    strategy_elements=[experience.approach_used],
                    success_rate=experience.success_score,
                    efficiency_gain=experience.efficiency_score,
                    quality_improvement=experience.quality_score,
                    usage_count=1,
                    last_used=experience.timestamp
                )
                self.strategy_patterns[pattern_id] = pattern
                self.learning_metrics["successful_patterns"] += 1
    
    async def _extract_optimization_insights_advanced(self, experience: LearningExperience) -> None:
        """Extract advanced optimization insights"""
        
        for insight_text in experience.optimization_insights:
            insight_id = hashlib.md5(insight_text.encode()).hexdigest()[:8]
            
            if insight_id not in self.optimization_insights:
                insight = OptimizationInsight(
                    insight_id=insight_id,
                    insight_type="performance_optimization",
                    context=experience.context,
                    optimization_target="quality_and_efficiency",
                    improvement_method=insight_text,
                    expected_gain=0.1,  # Conservative estimate
                    validation_count=1,
                    confidence=0.7
                )
                self.optimization_insights[insight_id] = insight
                self.learning_metrics["optimization_insights"] += 1
            else:
                # Update existing insight
                insight = self.optimization_insights[insight_id]
                insight.validation_count += 1
                insight.confidence = min(insight.confidence + 0.1, 1.0)
    
    def _find_relevant_experiences(self, task_type: str, context: Dict) -> List[LearningExperience]:
        """Find relevant experiences for the given task and context"""
        relevant = []
        
        for exp in self.learning_experiences:
            if exp.task_type == task_type:
                # Calculate context similarity
                similarity = self._calculate_context_similarity(exp.context, context)
                if similarity > 0.5:  # Threshold for relevance
                    relevant.append(exp)
        
        # Sort by success score and recency
        relevant.sort(key=lambda x: (x.success_score, x.timestamp), reverse=True)
        return relevant[:10]  # Top 10 most relevant
    
    def _find_applicable_patterns(self, task_type: str, context: Dict) -> List[StrategyPattern]:
        """Find applicable strategy patterns"""
        applicable = []
        
        for pattern in self.strategy_patterns.values():
            if pattern.context_conditions.get("task_type") == task_type:
                applicable.append(pattern)
        
        # Sort by success rate and usage count
        applicable.sort(key=lambda x: (x.success_rate, x.usage_count), reverse=True)
        return applicable
    
    def _get_relevant_insights(self, task_type: str, context: Dict) -> List[OptimizationInsight]:
        """Get relevant optimization insights"""
        relevant = []
        
        for insight in self.optimization_insights.values():
            if insight.confidence > 0.6:  # High confidence insights only
                relevant.append(insight)
        
        # Sort by confidence and validation count
        relevant.sort(key=lambda x: (x.confidence, x.validation_count), reverse=True)
        return relevant[:5]  # Top 5 insights
    
    def _determine_base_approach(self, experiences: List[LearningExperience]) -> str:
        """Determine the best base approach from experiences"""
        if not experiences:
            return "deep_analytical_approach"
        
        # Find the most successful approach
        best_exp = max(experiences, key=lambda x: x.success_score)
        return best_exp.approach_used
    
    def _compile_optimizations(self, patterns: List[StrategyPattern], insights: List[OptimizationInsight]) -> List[str]:
        """Compile optimization elements from patterns and insights"""
        optimizations = []
        
        for pattern in patterns:
            optimizations.extend(pattern.strategy_elements)
        
        for insight in insights:
            optimizations.append(insight.improvement_method)
        
        return list(set(optimizations))  # Remove duplicates
    
    def _suggest_quality_enhancements(self, experiences: List[LearningExperience]) -> List[str]:
        """Suggest quality enhancements based on experiences"""
        enhancements = []
        
        high_quality_experiences = [exp for exp in experiences if exp.quality_score > 0.8]
        
        for exp in high_quality_experiences:
            enhancements.extend(exp.lessons_learned)
        
        return list(set(enhancements))
    
    def _suggest_efficiency_improvements(self, experiences: List[LearningExperience]) -> List[str]:
        """Suggest efficiency improvements based on experiences"""
        improvements = []
        
        efficient_experiences = [exp for exp in experiences if exp.efficiency_score > 0.8]
        
        for exp in efficient_experiences:
            improvements.extend(exp.optimization_insights)
        
        return list(set(improvements))
    
    def _suggest_learning_adaptations(self, experiences: List[LearningExperience]) -> List[str]:
        """Suggest learning adaptations based on experiences"""
        adaptations = [
            "Apply deep reasoning for complex tasks",
            "Integrate optimization at every step",
            "Use multi-layered analysis approach",
            "Implement continuous learning cycles",
            "Focus on quality enhancement"
        ]
        
        return adaptations
    
    def _calculate_strategy_confidence(self, experiences: List[LearningExperience], patterns: List[StrategyPattern]) -> float:
        """Calculate confidence in the suggested strategy"""
        if not experiences and not patterns:
            return 0.5  # Neutral confidence
        
        exp_confidence = sum(exp.success_score for exp in experiences) / len(experiences) if experiences else 0.5
        pattern_confidence = sum(pattern.success_rate for pattern in patterns) / len(patterns) if patterns else 0.5
        
        return (exp_confidence + pattern_confidence) / 2
    
    def _predict_improvements(self, patterns: List[StrategyPattern], insights: List[OptimizationInsight]) -> Dict:
        """Predict expected improvements from applying patterns and insights"""
        improvements = {
            "efficiency_gain": 0.0,
            "quality_improvement": 0.0,
            "success_probability": 0.0
        }
        
        if patterns:
            improvements["efficiency_gain"] = sum(p.efficiency_gain for p in patterns) / len(patterns)
            improvements["quality_improvement"] = sum(p.quality_improvement for p in patterns) / len(patterns)
            improvements["success_probability"] = sum(p.success_rate for p in patterns) / len(patterns)
        
        if insights:
            avg_gain = sum(i.expected_gain for i in insights) / len(insights)
            improvements["efficiency_gain"] += avg_gain
            improvements["quality_improvement"] += avg_gain
        
        return improvements
    
    def _calculate_context_similarity(self, context1: Dict, context2: Dict) -> float:
        """Calculate similarity between two contexts"""
        # Simple similarity calculation based on common keys and values
        common_keys = set(context1.keys()) & set(context2.keys())
        if not common_keys:
            return 0.0
        
        matches = sum(1 for key in common_keys if context1[key] == context2[key])
        return matches / len(common_keys)
    
    def _generate_pattern_id(self, context: Dict) -> str:
        """Generate a unique pattern ID from context"""
        context_str = json.dumps(context, sort_keys=True)
        return hashlib.md5(context_str.encode()).hexdigest()[:8]
    
    def _analyze_performance_trends(self) -> Dict:
        """Analyze performance trends over time"""
        trends = {}
        
        for metric, values in self.performance_trends.items():
            if len(values) >= 2:
                recent_avg = sum(values[-5:]) / min(5, len(values))
                overall_avg = sum(values) / len(values)
                trends[metric] = {
                    "recent_average": recent_avg,
                    "overall_average": overall_avg,
                    "trend": "improving" if recent_avg > overall_avg else "stable" if recent_avg == overall_avg else "declining"
                }
        
        return trends
    
    def _get_top_patterns(self) -> List[Dict]:
        """Get top performing patterns"""
        top_patterns = sorted(
            self.strategy_patterns.values(),
            key=lambda x: (x.success_rate, x.usage_count),
            reverse=True
        )[:5]
        
        return [asdict(pattern) for pattern in top_patterns]
    
    def _identify_optimization_opportunities(self) -> List[str]:
        """Identify optimization opportunities"""
        opportunities = []
        
        # Analyze performance trends for opportunities
        trends = self._analyze_performance_trends()
        
        for metric, trend_data in trends.items():
            if trend_data["trend"] == "declining":
                opportunities.append(f"Improve {metric} performance")
            elif trend_data["recent_average"] < 0.8:
                opportunities.append(f"Optimize {metric} to reach excellence threshold")
        
        return opportunities
    
    def _generate_learning_recommendations(self) -> List[str]:
        """Generate learning recommendations"""
        recommendations = [
            "Continue deep reasoning approach for complex tasks",
            "Integrate optimization insights into all strategies",
            "Focus on quality enhancement in all outputs",
            "Maintain learning cycles for continuous improvement",
            "Apply successful patterns to similar contexts"
        ]
        
        return recommendations
    
    def _suggest_strategy_adaptations(self) -> List[str]:
        """Suggest strategy adaptations based on learning"""
        adaptations = [
            "Enhance reasoning depth for better outcomes",
            "Integrate more optimization checkpoints",
            "Apply learned patterns more systematically",
            "Focus on quality metrics in strategy selection",
            "Implement adaptive learning in real-time"
        ]
        
        return adaptations
    
    def _recommend_quality_improvements(self) -> List[str]:
        """Recommend quality improvements"""
        improvements = [
            "Implement multi-layered analysis for all tasks",
            "Apply optimization frameworks systematically",
            "Integrate learning insights into quality assessment",
            "Use deep reasoning for complex problem solving",
            "Maintain excellence standards in all outputs"
        ]
        
        return improvements
    
    def _recommend_efficiency_gains(self) -> List[str]:
        """Recommend efficiency gains"""
        gains = [
            "Apply learned optimization patterns",
            "Use efficient strategy selection based on context",
            "Implement smart resource allocation",
            "Apply time-tested successful approaches",
            "Optimize execution paths based on experience"
        ]
        
        return gains

