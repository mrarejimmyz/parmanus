"""
Intelligent Error Handling and Adaptive Recovery System for ParManusAI
"""

import asyncio
import json
import time
from typing import Dict, List, Optional, Any
from enum import Enum

# Use standard logging instead of loguru for compatibility
import logging
logger = logging.getLogger(__name__)


class ErrorType(Enum):
    """Classification of different error types for targeted recovery"""
    CONTENT_EXTRACTION_FAILURE = "content_extraction_failure"
    NAVIGATION_FAILURE = "navigation_failure"
    AUTHENTICATION_REQUIRED = "authentication_required"
    NETWORK_CONNECTIVITY = "network_connectivity"
    JAVASCRIPT_TIMEOUT = "javascript_timeout"
    ANTI_BOT_PROTECTION = "anti_bot_protection"
    CONFIGURATION_ERROR = "configuration_error"
    UNKNOWN_ERROR = "unknown_error"


class RecoveryStrategy(Enum):
    """Available recovery strategies"""
    RETRY_WITH_DELAY = "retry_with_delay"
    ALTERNATIVE_APPROACH = "alternative_approach"
    FALLBACK_TOOL = "fallback_tool"
    HUMAN_ESCALATION = "human_escalation"
    SKIP_AND_CONTINUE = "skip_and_continue"
    ABORT_TASK = "abort_task"


class IntelligentErrorHandler:
    """
    Intelligent error handling system with context-aware recovery strategies
    """
    
    def __init__(self):
        self.error_history: List[Dict] = []
        self.recovery_attempts: Dict[str, int] = {}
        self.max_recovery_attempts = 3
        self.error_patterns = self._initialize_error_patterns()
        
    def _initialize_error_patterns(self) -> Dict[ErrorType, Dict]:
        """Initialize error patterns and their corresponding recovery strategies"""
        return {
            ErrorType.CONTENT_EXTRACTION_FAILURE: {
                "keywords": ["no content", "extraction failed", "empty content", "javascript", "dynamic"],
                "recovery_strategies": [
                    RecoveryStrategy.RETRY_WITH_DELAY,
                    RecoveryStrategy.ALTERNATIVE_APPROACH,
                    RecoveryStrategy.FALLBACK_TOOL
                ],
                "escalation_threshold": 2
            },
            ErrorType.NAVIGATION_FAILURE: {
                "keywords": ["navigation failed", "page not found", "timeout", "connection refused"],
                "recovery_strategies": [
                    RecoveryStrategy.RETRY_WITH_DELAY,
                    RecoveryStrategy.ALTERNATIVE_APPROACH
                ],
                "escalation_threshold": 2
            },
            ErrorType.AUTHENTICATION_REQUIRED: {
                "keywords": ["login required", "authentication", "unauthorized", "access denied"],
                "recovery_strategies": [
                    RecoveryStrategy.HUMAN_ESCALATION
                ],
                "escalation_threshold": 1
            },
            ErrorType.NETWORK_CONNECTIVITY: {
                "keywords": ["network error", "connection timeout", "dns", "unreachable"],
                "recovery_strategies": [
                    RecoveryStrategy.RETRY_WITH_DELAY,
                    RecoveryStrategy.ALTERNATIVE_APPROACH
                ],
                "escalation_threshold": 3
            },
            ErrorType.JAVASCRIPT_TIMEOUT: {
                "keywords": ["javascript timeout", "script error", "execution timeout"],
                "recovery_strategies": [
                    RecoveryStrategy.RETRY_WITH_DELAY,
                    RecoveryStrategy.ALTERNATIVE_APPROACH
                ],
                "escalation_threshold": 2
            },
            ErrorType.ANTI_BOT_PROTECTION: {
                "keywords": ["captcha", "bot protection", "cloudflare", "rate limit"],
                "recovery_strategies": [
                    RecoveryStrategy.HUMAN_ESCALATION,
                    RecoveryStrategy.ALTERNATIVE_APPROACH
                ],
                "escalation_threshold": 1
            },
            ErrorType.CONFIGURATION_ERROR: {
                "keywords": ["configuration", "nonetype", "attribute error", "import error"],
                "recovery_strategies": [
                    RecoveryStrategy.ALTERNATIVE_APPROACH,
                    RecoveryStrategy.HUMAN_ESCALATION
                ],
                "escalation_threshold": 1
            }
        }
    
    def classify_error(self, error_message: str, context: Dict = None) -> ErrorType:
        """
        Classify an error based on its message and context
        """
        error_message_lower = error_message.lower()
        
        # Check each error type pattern
        for error_type, pattern_info in self.error_patterns.items():
            for keyword in pattern_info["keywords"]:
                if keyword in error_message_lower:
                    return error_type
        
        # Check context for additional clues
        if context:
            if context.get("action") == "extract_content":
                return ErrorType.CONTENT_EXTRACTION_FAILURE
            elif context.get("action") in ["go_to_url", "navigate"]:
                return ErrorType.NAVIGATION_FAILURE
        
        return ErrorType.UNKNOWN_ERROR
    
    def get_recovery_strategy(self, error_type: ErrorType, attempt_count: int) -> RecoveryStrategy:
        """
        Get the appropriate recovery strategy based on error type and attempt count
        """
        pattern_info = self.error_patterns.get(error_type, {})
        strategies = pattern_info.get("recovery_strategies", [RecoveryStrategy.HUMAN_ESCALATION])
        
        # Return strategy based on attempt count
        if attempt_count < len(strategies):
            return strategies[attempt_count]
        else:
            # If we've exhausted specific strategies, escalate to human
            return RecoveryStrategy.HUMAN_ESCALATION
    
    def should_escalate(self, error_type: ErrorType, attempt_count: int) -> bool:
        """
        Determine if the error should be escalated to human intervention
        """
        pattern_info = self.error_patterns.get(error_type, {})
        threshold = pattern_info.get("escalation_threshold", 2)
        return attempt_count >= threshold
    
    def record_error(self, error_message: str, context: Dict = None) -> Dict:
        """
        Record an error and return recovery recommendations
        """
        error_type = self.classify_error(error_message, context)
        error_key = f"{error_type.value}:{context.get('action', 'unknown') if context else 'unknown'}"
        
        # Increment attempt count
        self.recovery_attempts[error_key] = self.recovery_attempts.get(error_key, 0) + 1
        attempt_count = self.recovery_attempts[error_key]
        
        # Get recovery strategy
        recovery_strategy = self.get_recovery_strategy(error_type, attempt_count - 1)
        should_escalate = self.should_escalate(error_type, attempt_count)
        
        # Record in history
        error_record = {
            "timestamp": time.time(),
            "error_message": error_message,
            "error_type": error_type.value,
            "context": context or {},
            "attempt_count": attempt_count,
            "recovery_strategy": recovery_strategy.value,
            "should_escalate": should_escalate
        }
        
        self.error_history.append(error_record)
        
        # Keep only last 50 errors
        if len(self.error_history) > 50:
            self.error_history = self.error_history[-50:]
        
        logger.info(f"Error classified as {error_type.value}, attempt {attempt_count}, strategy: {recovery_strategy.value}")
        
        return error_record
    
    def get_recovery_instructions(self, error_record: Dict) -> Dict:
        """
        Get detailed recovery instructions based on the error record
        """
        error_type = ErrorType(error_record["error_type"])
        recovery_strategy = RecoveryStrategy(error_record["recovery_strategy"])
        
        instructions = {
            "strategy": recovery_strategy.value,
            "should_escalate": error_record["should_escalate"],
            "instructions": "",
            "alternative_actions": [],
            "escalation_message": ""
        }
        
        # Generate specific instructions based on error type and strategy
        if error_type == ErrorType.CONTENT_EXTRACTION_FAILURE:
            if recovery_strategy == RecoveryStrategy.RETRY_WITH_DELAY:
                instructions["instructions"] = "Wait 5-10 seconds for dynamic content to load, then retry extraction"
                instructions["alternative_actions"] = ["wait", "extract_content"]
            elif recovery_strategy == RecoveryStrategy.ALTERNATIVE_APPROACH:
                instructions["instructions"] = "Try alternative extraction methods or take screenshot for analysis"
                instructions["alternative_actions"] = ["screenshot", "extract_basic_info", "analyze_page_structure"]
            elif recovery_strategy == RecoveryStrategy.FALLBACK_TOOL:
                instructions["instructions"] = "Use web search to find information about this website"
                instructions["alternative_actions"] = ["web_search", "find_alternative_sources"]
        
        elif error_type == ErrorType.NAVIGATION_FAILURE:
            if recovery_strategy == RecoveryStrategy.RETRY_WITH_DELAY:
                instructions["instructions"] = "Wait and retry navigation, possibly with different URL format"
                instructions["alternative_actions"] = ["wait", "try_www_prefix", "try_https"]
            elif recovery_strategy == RecoveryStrategy.ALTERNATIVE_APPROACH:
                instructions["instructions"] = "Search for the website or try alternative URLs"
                instructions["alternative_actions"] = ["web_search", "check_domain_status"]
        
        elif error_type == ErrorType.AUTHENTICATION_REQUIRED:
            instructions["instructions"] = "Website requires login or authentication"
            instructions["escalation_message"] = "This website requires user authentication. Please provide login credentials or access the site manually."
        
        elif error_type == ErrorType.ANTI_BOT_PROTECTION:
            instructions["instructions"] = "Website has anti-bot protection active"
            instructions["escalation_message"] = "This website has anti-bot protection (CAPTCHA, Cloudflare, etc.). Manual access may be required."
        
        # Add escalation message if needed
        if error_record["should_escalate"] and not instructions["escalation_message"]:
            instructions["escalation_message"] = f"Multiple attempts to resolve {error_type.value} have failed. Human intervention may be required."
        
        return instructions
    
    def get_error_summary(self) -> Dict:
        """
        Get a summary of recent errors and patterns
        """
        if not self.error_history:
            return {"total_errors": 0, "recent_patterns": []}
        
        # Analyze recent errors (last 10)
        recent_errors = self.error_history[-10:]
        error_types = {}
        
        for error in recent_errors:
            error_type = error["error_type"]
            error_types[error_type] = error_types.get(error_type, 0) + 1
        
        return {
            "total_errors": len(self.error_history),
            "recent_errors": len(recent_errors),
            "error_type_distribution": error_types,
            "most_common_error": max(error_types.items(), key=lambda x: x[1])[0] if error_types else None,
            "escalation_needed": any(error["should_escalate"] for error in recent_errors[-3:])
        }


class AdaptiveRecoverySystem:
    """
    Adaptive recovery system that learns from past failures and successes
    """
    
    def __init__(self):
        self.error_handler = IntelligentErrorHandler()
        self.success_patterns: Dict[str, int] = {}
        self.failure_patterns: Dict[str, int] = {}
        
    def handle_error(self, error_message: str, context: Dict = None) -> Dict:
        """
        Handle an error and return recovery recommendations
        """
        error_record = self.error_handler.record_error(error_message, context)
        recovery_instructions = self.error_handler.get_recovery_instructions(error_record)
        
        # Record failure pattern
        pattern_key = f"{error_record['error_type']}:{context.get('action', 'unknown') if context else 'unknown'}"
        self.failure_patterns[pattern_key] = self.failure_patterns.get(pattern_key, 0) + 1
        
        return {
            "error_record": error_record,
            "recovery_instructions": recovery_instructions,
            "error_summary": self.error_handler.get_error_summary()
        }
    
    def record_success(self, action: str, context: Dict = None) -> None:
        """
        Record a successful action to learn from success patterns
        """
        pattern_key = f"success:{action}:{context.get('url_domain', 'unknown') if context else 'unknown'}"
        self.success_patterns[pattern_key] = self.success_patterns.get(pattern_key, 0) + 1
    
    def get_success_probability(self, action: str, context: Dict = None) -> float:
        """
        Estimate success probability for an action based on historical data
        """
        pattern_key = f"success:{action}:{context.get('url_domain', 'unknown') if context else 'unknown'}"
        failure_key = f"content_extraction_failure:{action}"
        
        successes = self.success_patterns.get(pattern_key, 0)
        failures = self.failure_patterns.get(failure_key, 0)
        
        if successes + failures == 0:
            return 0.5  # No data, assume 50% probability
        
        return successes / (successes + failures)
    
    def should_attempt_action(self, action: str, context: Dict = None, threshold: float = 0.3) -> bool:
        """
        Determine if an action should be attempted based on success probability
        """
        probability = self.get_success_probability(action, context)
        return probability >= threshold
    
    def get_adaptive_recommendations(self, current_context: Dict) -> List[Dict]:
        """
        Get adaptive recommendations based on current context and historical patterns
        """
        recommendations = []
        
        # Analyze current situation
        action = current_context.get("action", "unknown")
        url_domain = current_context.get("url_domain", "unknown")
        
        # Check success probability for current action
        success_prob = self.get_success_probability(action, current_context)
        
        if success_prob < 0.3:
            recommendations.append({
                "type": "warning",
                "message": f"Low success probability ({success_prob:.1%}) for {action} on {url_domain}",
                "suggestion": "Consider alternative approaches"
            })
        
        # Suggest alternative actions based on success patterns
        alternative_actions = ["web_search", "extract_basic_info", "screenshot"]
        for alt_action in alternative_actions:
            alt_prob = self.get_success_probability(alt_action, current_context)
            if alt_prob > success_prob + 0.2:  # Significantly better probability
                recommendations.append({
                    "type": "suggestion",
                    "message": f"Consider using {alt_action} instead (success rate: {alt_prob:.1%})",
                    "action": alt_action
                })
        
        return recommendations

