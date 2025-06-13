"""
Enhanced Memory System for ParManusAI with Task Planning Integration
"""

import json
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

from app.logger import logger
from app.schema import Message
from app.memory import Memory


class TaskContext:
    """Represents the context of a specific task"""
    
    def __init__(self, task_id: str, goal: str, created_at: float = None):
        self.task_id = task_id
        self.goal = goal
        self.created_at = created_at or time.time()
        self.phases_completed = 0
        self.current_phase = 0
        self.steps_completed = 0
        self.current_step = 0
        self.status = "active"
        self.key_findings = []
        self.challenges_encountered = []
        self.adaptations_made = []
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization"""
        return {
            "task_id": self.task_id,
            "goal": self.goal,
            "created_at": self.created_at,
            "phases_completed": self.phases_completed,
            "current_phase": self.current_phase,
            "steps_completed": self.steps_completed,
            "current_step": self.current_step,
            "status": self.status,
            "key_findings": self.key_findings,
            "challenges_encountered": self.challenges_encountered,
            "adaptations_made": self.adaptations_made
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "TaskContext":
        """Create from dictionary"""
        context = cls(data["task_id"], data["goal"], data["created_at"])
        context.phases_completed = data.get("phases_completed", 0)
        context.current_phase = data.get("current_phase", 0)
        context.steps_completed = data.get("steps_completed", 0)
        context.current_step = data.get("current_step", 0)
        context.status = data.get("status", "active")
        context.key_findings = data.get("key_findings", [])
        context.challenges_encountered = data.get("challenges_encountered", [])
        context.adaptations_made = data.get("adaptations_made", [])
        return context


class EnhancedMemory(Memory):
    """Enhanced memory system with task planning and progress tracking"""
    
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        
        # Task-specific memory
        self.current_task: Optional[TaskContext] = None
        self.task_history: List[TaskContext] = []
        
        # Enhanced context tracking
        self.important_messages: List[Message] = []
        self.tool_usage_history: List[Dict[str, Any]] = []
        self.error_patterns: List[Dict[str, Any]] = []
        
        # Memory optimization
        self.max_context_messages = 50
        self.max_tool_history = 100
        
    def start_new_task(self, goal: str, task_id: str = None) -> TaskContext:
        """Start tracking a new task"""
        if task_id is None:
            task_id = f"task_{int(time.time())}"
        
        # Archive current task if exists
        if self.current_task:
            self.current_task.status = "completed"
            self.task_history.append(self.current_task)
        
        # Create new task context
        self.current_task = TaskContext(task_id, goal)
        logger.info(f"Started new task: {task_id} - {goal}")
        
        return self.current_task
    
    def update_task_progress(self, phase: int = None, step: int = None, 
                           finding: str = None, challenge: str = None, 
                           adaptation: str = None):
        """Update current task progress"""
        if not self.current_task:
            return
        
        if phase is not None:
            self.current_task.current_phase = phase
        
        if step is not None:
            self.current_task.current_step = step
        
        if finding:
            self.current_task.key_findings.append({
                "timestamp": time.time(),
                "finding": finding
            })
        
        if challenge:
            self.current_task.challenges_encountered.append({
                "timestamp": time.time(),
                "challenge": challenge
            })
        
        if adaptation:
            self.current_task.adaptations_made.append({
                "timestamp": time.time(),
                "adaptation": adaptation
            })
    
    def mark_message_important(self, message: Message, reason: str = ""):
        """Mark a message as important for long-term retention"""
        # Add metadata to track importance
        important_msg = Message(
            role=message.role,
            content=message.content,
            metadata={
                "importance": "high",
                "reason": reason,
                "timestamp": time.time(),
                "task_id": self.current_task.task_id if self.current_task else None
            }
        )
        
        self.important_messages.append(important_msg)
        logger.debug(f"Marked message as important: {reason}")
    
    def track_tool_usage(self, tool_name: str, action: str, result: str, 
                        success: bool = True, execution_time: float = 0):
        """Track tool usage for pattern analysis"""
        usage_record = {
            "timestamp": time.time(),
            "tool_name": tool_name,
            "action": action,
            "result": result[:200],  # Truncate long results
            "success": success,
            "execution_time": execution_time,
            "task_id": self.current_task.task_id if self.current_task else None
        }
        
        self.tool_usage_history.append(usage_record)
        
        # Keep only recent history
        if len(self.tool_usage_history) > self.max_tool_history:
            self.tool_usage_history = self.tool_usage_history[-self.max_tool_history:]
        
        # Track error patterns
        if not success:
            self._track_error_pattern(tool_name, action, result)
    
    def _track_error_pattern(self, tool_name: str, action: str, error: str):
        """Track error patterns for better recovery"""
        error_pattern = {
            "timestamp": time.time(),
            "tool_name": tool_name,
            "action": action,
            "error": error[:200],
            "task_id": self.current_task.task_id if self.current_task else None
        }
        
        self.error_patterns.append(error_pattern)
        
        # Keep only recent errors
        if len(self.error_patterns) > 50:
            self.error_patterns = self.error_patterns[-50:]
    
    def get_enhanced_context(self, max_tokens: int = 4000, 
                           include_task_context: bool = True,
                           include_important_messages: bool = True) -> List[Message]:
        """Get enhanced context with task-aware message selection"""
        context_messages = []
        
        # Always include system messages
        system_messages = [msg for msg in self.messages if msg.role == "system"]
        context_messages.extend(system_messages)
        
        # Include task context if requested
        if include_task_context and self.current_task:
            task_context_msg = self._create_task_context_message()
            context_messages.append(task_context_msg)
        
        # Include important messages if requested
        if include_important_messages:
            # Add recent important messages
            recent_important = [msg for msg in self.important_messages[-10:]]
            context_messages.extend(recent_important)
        
        # Calculate remaining token budget
        used_chars = sum(len(msg.content or "") for msg in context_messages)
        remaining_chars = (max_tokens * 4) - used_chars
        
        # Add recent messages within remaining budget
        for message in reversed(self.messages):
            if message.role == "system":
                continue  # Already included
            
            message_chars = len(message.content or "")
            if message_chars > remaining_chars:
                break
            
            context_messages.insert(-len(recent_important) if recent_important else len(context_messages), message)
            remaining_chars -= message_chars
        
        return context_messages
    
    def _create_task_context_message(self) -> Message:
        """Create a message with current task context"""
        if not self.current_task:
            return Message.system_message("No active task context")
        
        context_content = f"""
CURRENT TASK CONTEXT:
Goal: {self.current_task.goal}
Status: {self.current_task.status}
Current Phase: {self.current_task.current_phase + 1}
Current Step: {self.current_task.current_step + 1}
"""
        
        if self.current_task.key_findings:
            context_content += f"\nKey Findings:\n"
            for finding in self.current_task.key_findings[-3:]:  # Last 3 findings
                context_content += f"- {finding['finding']}\n"
        
        if self.current_task.challenges_encountered:
            context_content += f"\nRecent Challenges:\n"
            for challenge in self.current_task.challenges_encountered[-2:]:  # Last 2 challenges
                context_content += f"- {challenge['challenge']}\n"
        
        return Message.system_message(context_content)
    
    def get_tool_usage_patterns(self, tool_name: str = None, 
                               recent_only: bool = True) -> List[Dict[str, Any]]:
        """Get tool usage patterns for analysis"""
        patterns = self.tool_usage_history
        
        if tool_name:
            patterns = [p for p in patterns if p["tool_name"] == tool_name]
        
        if recent_only:
            # Only patterns from last hour
            cutoff_time = time.time() - 3600
            patterns = [p for p in patterns if p["timestamp"] > cutoff_time]
        
        return patterns
    
    def get_error_patterns(self, tool_name: str = None) -> List[Dict[str, Any]]:
        """Get error patterns for debugging"""
        patterns = self.error_patterns
        
        if tool_name:
            patterns = [p for p in patterns if p["tool_name"] == tool_name]
        
        return patterns
    
    def compress_memory_intelligently(self):
        """Intelligent memory compression that preserves important context"""
        if len(self.messages) < self.max_context_messages:
            return
        
        # Identify messages to keep
        keep_messages = []
        
        # Always keep system messages
        keep_messages.extend([msg for msg in self.messages if msg.role == "system"])
        
        # Keep important messages
        keep_messages.extend(self.important_messages)
        
        # Keep recent messages (last 20)
        keep_messages.extend(self.messages[-20:])
        
        # Keep messages with tool calls that were successful
        successful_tool_messages = [
            msg for msg in self.messages 
            if msg.role == "assistant" and "tool" in (msg.content or "").lower()
        ][-10:]  # Last 10 successful tool messages
        
        keep_messages.extend(successful_tool_messages)
        
        # Remove duplicates while preserving order
        seen = set()
        compressed_messages = []
        for msg in keep_messages:
            msg_id = id(msg)
            if msg_id not in seen:
                seen.add(msg_id)
                compressed_messages.append(msg)
        
        # Update messages
        old_count = len(self.messages)
        self.messages = compressed_messages
        new_count = len(self.messages)
        
        logger.info(f"Compressed memory: {old_count} -> {new_count} messages")
    
    def save_enhanced_session(self, filename: str = None):
        """Save session with enhanced context"""
        try:
            save_file = self.session_dir / filename if filename else self.session_file
            
            session_data = {
                "timestamp": datetime.now().isoformat(),
                "initial_prompt": self.initial_prompt,
                "messages": [self._message_to_dict(msg) for msg in self.messages],
                "current_task": self.current_task.to_dict() if self.current_task else None,
                "task_history": [task.to_dict() for task in self.task_history],
                "important_messages": [self._message_to_dict(msg) for msg in self.important_messages],
                "tool_usage_history": self.tool_usage_history[-50:],  # Last 50 entries
                "error_patterns": self.error_patterns[-20:]  # Last 20 errors
            }
            
            with open(save_file, "w", encoding="utf-8") as f:
                json.dump(session_data, f, indent=2, ensure_ascii=False)
            
            logger.info(f"Enhanced session saved to {save_file}")
            
        except Exception as e:
            logger.error(f"Failed to save enhanced session: {e}")
    
    def load_enhanced_session(self, filename: str = None) -> bool:
        """Load session with enhanced context"""
        try:
            load_file = self.session_dir / filename if filename else self.session_file
            
            if not load_file.exists():
                logger.info("No previous enhanced session found")
                return False
            
            with open(load_file, "r", encoding="utf-8") as f:
                session_data = json.load(f)
            
            # Load basic session data
            self.initial_prompt = session_data.get("initial_prompt")
            self.messages = [
                self._dict_to_message(msg_dict)
                for msg_dict in session_data.get("messages", [])
            ]
            
            # Load enhanced data
            if session_data.get("current_task"):
                self.current_task = TaskContext.from_dict(session_data["current_task"])
            
            self.task_history = [
                TaskContext.from_dict(task_dict)
                for task_dict in session_data.get("task_history", [])
            ]
            
            self.important_messages = [
                self._dict_to_message(msg_dict)
                for msg_dict in session_data.get("important_messages", [])
            ]
            
            self.tool_usage_history = session_data.get("tool_usage_history", [])
            self.error_patterns = session_data.get("error_patterns", [])
            
            logger.info(f"Enhanced session loaded from {load_file}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to load enhanced session: {e}")
            return False

