import time
from abc import ABC, abstractmethod
from collections import deque
from contextlib import asynccontextmanager
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field, model_validator

from app.llm import LLM
from app.logger import logger
from app.sandbox.client import SANDBOX_CLIENT
from app.schema import ROLE_TYPE, AgentState, Memory, Message


class CircuitBreaker:
    """Circuit breaker pattern for handling repeated failures."""

    def __init__(self, failure_threshold: int = 3, recovery_timeout: int = 60):
        self.failure_threshold = failure_threshold
        self.recovery_timeout = recovery_timeout
        self.failure_count = 0
        self.last_failure_time = 0
        self.state = "CLOSED"  # CLOSED, OPEN, HALF_OPEN

    def call_failed(self):
        """Record a failure."""
        self.failure_count += 1
        self.last_failure_time = time.time()

        if self.failure_count >= self.failure_threshold:
            self.state = "OPEN"
            logger.warning(
                f"Circuit breaker opened after {self.failure_count} failures"
            )

    def call_succeeded(self):
        """Record a success."""
        self.failure_count = 0
        self.state = "CLOSED"

    def can_execute(self) -> bool:
        """Check if execution is allowed."""
        if self.state == "CLOSED":
            return True
        elif self.state == "OPEN":
            if time.time() - self.last_failure_time > self.recovery_timeout:
                self.state = "HALF_OPEN"
                logger.info("Circuit breaker entering half-open state")
                return True
            return False
        else:  # HALF_OPEN
            return True


class StuckStateDetector:
    """Advanced stuck state detection with multiple strategies."""

    def __init__(self, window_size: int = 5, similarity_threshold: float = 0.8):
        self.window_size = window_size
        self.similarity_threshold = similarity_threshold
        self.recent_responses = deque(maxlen=window_size)
        self.recent_actions = deque(maxlen=window_size)
        self.stuck_count = 0
        self.last_progress_time = time.time()

    def add_response(self, content: str, actions: List[str] = None):
        """Add a response for analysis."""
        self.recent_responses.append(content)
        if actions:
            self.recent_actions.append(tuple(actions))
        else:
            self.recent_actions.append(())

    def is_stuck(self) -> bool:
        """Detect if the agent is stuck using multiple strategies."""
        if len(self.recent_responses) < 2:
            return False

        # Strategy 1: Exact duplicate detection
        if self._has_exact_duplicates():
            self.stuck_count += 1
            logger.debug(f"Exact duplicate detected (count: {self.stuck_count})")
            return self.stuck_count >= 2

        # Strategy 2: Semantic similarity detection
        if self._has_semantic_similarity():
            self.stuck_count += 1
            logger.debug(f"Semantic similarity detected (count: {self.stuck_count})")
            return self.stuck_count >= 3

        # Strategy 3: Action repetition detection
        if self._has_action_repetition():
            self.stuck_count += 1
            logger.debug(f"Action repetition detected (count: {self.stuck_count})")
            return self.stuck_count >= 2

        # Strategy 4: Time-based stagnation
        if self._is_time_stagnant():
            logger.warning("Time-based stagnation detected")
            return True

        # Reset stuck count if no patterns detected
        self.stuck_count = max(0, self.stuck_count - 1)
        self.last_progress_time = time.time()
        return False

    def _has_exact_duplicates(self) -> bool:
        """Check for exact duplicate responses."""
        if len(self.recent_responses) < 2:
            return False

        last_response = self.recent_responses[-1]
        return any(
            response == last_response for response in list(self.recent_responses)[:-1]
        )

    def _has_semantic_similarity(self) -> bool:
        """Check for semantically similar responses."""
        if len(self.recent_responses) < 2:
            return False

        last_response = self.recent_responses[-1].lower()

        # Simple similarity check based on common words
        for response in list(self.recent_responses)[:-1]:
            response_lower = response.lower()

            # Calculate word overlap
            words1 = set(last_response.split())
            words2 = set(response_lower.split())

            if len(words1) == 0 or len(words2) == 0:
                continue

            overlap = len(words1.intersection(words2))
            similarity = overlap / max(len(words1), len(words2))

            if similarity > self.similarity_threshold:
                return True

        return False

    def _has_action_repetition(self) -> bool:
        """Check for repeated action patterns."""
        if len(self.recent_actions) < 2:
            return False

        last_actions = self.recent_actions[-1]
        return any(
            actions == last_actions for actions in list(self.recent_actions)[:-1]
        )

    def _is_time_stagnant(self) -> bool:
        """Check if too much time has passed without progress."""
        return time.time() - self.last_progress_time > 300  # 5 minutes

    def reset(self):
        """Reset the detector state."""
        self.recent_responses.clear()
        self.recent_actions.clear()
        self.stuck_count = 0
        self.last_progress_time = time.time()


class BaseAgent(BaseModel, ABC):
    """
    Abstract base class for managing agent state and execution with enhanced reliability.

    Provides foundational functionality for state transitions, memory management,
    and a step-based execution loop with circuit breaker and stuck state detection.
    """

    # Core attributes
    name: str = Field(..., description="Unique name of the agent")
    description: Optional[str] = Field(None, description="Optional agent description")

    # Prompts
    system_prompt: Optional[str] = Field(
        None, description="System-level instruction prompt"
    )
    next_step_prompt: Optional[str] = Field(
        None, description="Prompt for determining next action"
    )

    # Dependencies
    llm: LLM = Field(default_factory=LLM, description="Language model instance")
    memory: Memory = Field(default_factory=Memory, description="Agent's memory store")
    state: AgentState = Field(
        default=AgentState.IDLE, description="Current agent state"
    )

    # Execution control
    max_steps: int = Field(default=10, description="Maximum steps before termination")
    current_step: int = Field(default=0, description="Current step in execution")
    duplicate_threshold: int = Field(
        default=2, description="Threshold for duplicate detection"
    )

    # Enhanced reliability features
    circuit_breaker: CircuitBreaker = Field(default_factory=CircuitBreaker)
    stuck_detector: StuckStateDetector = Field(default_factory=StuckStateDetector)
    performance_metrics: Dict[str, Any] = Field(default_factory=dict)

    class Config:
        arbitrary_types_allowed = True
        extra = "allow"

    @model_validator(mode="after")
    def initialize_agent(self) -> "BaseAgent":
        """Initialize agent with enhanced monitoring."""
        logger.info(
            f"Initializing {self.name} agent with enhanced reliability features"
        )
        self.performance_metrics = {
            "start_time": time.time(),
            "total_steps": 0,
            "successful_steps": 0,
            "failed_steps": 0,
            "stuck_recoveries": 0,
        }
        return self

    @asynccontextmanager
    async def state_context(self, new_state: AgentState):
        """Context manager for safe state transitions."""
        old_state = self.state
        self.state = new_state
        logger.debug(f"{self.name} state: {old_state} -> {new_state}")
        try:
            yield
        finally:
            if self.state == new_state:  # Only revert if state hasn't changed
                self.state = old_state
                logger.debug(f"{self.name} state reverted: {new_state} -> {old_state}")

    def update_memory(
        self,
        role: ROLE_TYPE,
        content: str,
        base64_image: Optional[str] = None,
        **kwargs,
    ):
        """Add a message to the agent's memory with validation."""
        message_map = {
            "user": Message.user_message,
            "system": Message.system_message,
            "assistant": Message.assistant_message,
            "tool": lambda content, **kw: Message.tool_message(content, **kw),
        }

        if role not in message_map:
            raise ValueError(f"Unsupported message role: {role}")

        kwargs = {"base64_image": base64_image, **(kwargs if role == "tool" else {})}
        self.memory.add_message(message_map[role](content, **kwargs))

    async def run(self, request: Optional[str] = None) -> str:
        """Execute the agent's main loop with enhanced error handling and monitoring."""
        if self.state != AgentState.IDLE:
            raise RuntimeError(f"Cannot run agent from state: {self.state}")

        if request:
            self.update_memory("user", request)

        results: List[str] = []
        start_time = time.time()

        async with self.state_context(AgentState.RUNNING):
            while (
                self.current_step < self.max_steps
                and self.state != AgentState.FINISHED
                and self.circuit_breaker.can_execute()
            ):
                self.current_step += 1
                self.performance_metrics["total_steps"] += 1

                logger.info(f"Executing step {self.current_step}/{self.max_steps}")

                try:
                    step_start = time.time()
                    step_result = await self.step()
                    step_duration = time.time() - step_start

                    # Record successful step
                    self.performance_metrics["successful_steps"] += 1
                    self.circuit_breaker.call_succeeded()

                    # Add to stuck detector
                    self.stuck_detector.add_response(step_result)

                    # Check for stuck state
                    if self.stuck_detector.is_stuck():
                        recovery_success = await self.handle_stuck_state_advanced()
                        if recovery_success:
                            self.performance_metrics["stuck_recoveries"] += 1
                        else:
                            logger.error(
                                "Failed to recover from stuck state, terminating"
                            )
                            break

                    results.append(f"Step {self.current_step}: {step_result}")

                    # Log performance if slow
                    if step_duration > 30:
                        logger.warning(
                            f"Step {self.current_step} took {step_duration:.1f}s"
                        )

                except Exception as e:
                    logger.error(f"Step {self.current_step} failed: {e}")
                    self.performance_metrics["failed_steps"] += 1
                    self.circuit_breaker.call_failed()

                    # Add error to results
                    results.append(f"Step {self.current_step}: Error - {str(e)}")

                    # Break if circuit breaker opens
                    if not self.circuit_breaker.can_execute():
                        logger.error("Circuit breaker opened, terminating execution")
                        break

            # Handle termination reasons
            if self.current_step >= self.max_steps:
                self.current_step = 0
                self.state = AgentState.IDLE
                results.append(f"Terminated: Reached max steps ({self.max_steps})")
            elif not self.circuit_breaker.can_execute():
                results.append(
                    "Terminated: Circuit breaker opened due to repeated failures"
                )

        # Log performance summary
        total_duration = time.time() - start_time
        self._log_performance_summary(total_duration)

        await SANDBOX_CLIENT.cleanup()
        return "\n".join(results) if results else "No steps executed"

    async def handle_stuck_state_advanced(self) -> bool:
        """Advanced stuck state handling with multiple recovery strategies."""
        logger.warning(f"Agent {self.name} detected stuck state, attempting recovery")

        # Strategy 1: Clear recent memory
        if len(self.memory.messages) > 10:
            logger.info("Clearing recent memory to break stuck pattern")
            self.memory.messages = self.memory.messages[:-3]  # Remove last 3 messages

        # Strategy 2: Add randomization prompt
        randomization_prompts = [
            "Try a completely different approach to solve this problem.",
            "Consider alternative methods you haven't tried yet.",
            "Step back and reassess the situation from a new perspective.",
            "Use a different strategy or tool to make progress.",
            "Break down the problem into smaller, different steps.",
        ]

        import random

        random_prompt = random.choice(randomization_prompts)
        self.next_step_prompt = (
            f"{random_prompt}\n\nOriginal task: {self.next_step_prompt}"
        )

        # Strategy 3: Reset stuck detector
        self.stuck_detector.reset()

        logger.info(f"Applied recovery strategy: {random_prompt}")
        return True

    def handle_stuck_state(self):
        """Legacy stuck state handler for backward compatibility."""
        stuck_prompt = (
            "Observed duplicate responses. Consider new strategies and "
            "avoid repeating ineffective paths already attempted."
        )
        self.next_step_prompt = f"{stuck_prompt}\n{self.next_step_prompt}"
        logger.warning(f"Agent detected stuck state. Added prompt: {stuck_prompt}")

    def is_stuck(self) -> bool:
        """Legacy stuck detection for backward compatibility."""
        return self.stuck_detector.is_stuck()

    def _log_performance_summary(self, duration: float):
        """Log performance summary."""
        metrics = self.performance_metrics
        success_rate = (
            metrics["successful_steps"] / max(metrics["total_steps"], 1) * 100
        )

        logger.info(
            f"Agent {self.name} performance summary: "
            f"Duration: {duration:.1f}s, "
            f"Steps: {metrics['total_steps']}, "
            f"Success rate: {success_rate:.1f}%, "
            f"Stuck recoveries: {metrics['stuck_recoveries']}"
        )

    @abstractmethod
    async def step(self) -> str:
        """Execute a single step in the agent's workflow."""

    @property
    def messages(self) -> List[Message]:
        """Retrieve a list of messages from the agent's memory."""
        return self.memory.messages

    @messages.setter
    def messages(self, value: List[Message]):
        """Set the list of messages in the agent's memory."""
        self.memory.messages = value
