import asyncio
import json
import os
import time
from typing import Dict, List, Optional

from app.exceptions import AgentTaskComplete
from app.logger import logger


class ManusUtils:
    def __init__(self, agent):
        self.agent = agent

    async def _validate_current_position(self) -> bool:
        """Validate current phase and step indices"""
        if not self.agent.current_plan or "phases" not in self.agent.current_plan:
            return False

        # Check phase bounds
        if self.agent.current_phase < 0 or self.agent.current_phase >= len(
            self.agent.current_plan["phases"]
        ):
            logger.error(f"Invalid phase index: {self.agent.current_phase}")
            return False

        current_phase = self.agent.current_plan["phases"][self.agent.current_phase]
        if "steps" not in current_phase:
            logger.error("Current phase has no steps")
            return False

        # Check step bounds
        if self.agent.current_step < 0 or self.agent.current_step >= len(
            current_phase["steps"]
        ):
            logger.error(f"Invalid step index: {self.agent.current_step}")
            return False

        return True

    async def _get_current_phase(self) -> Optional[Dict]:
        """Get the current phase from the plan"""
        if not self.agent.current_plan or "phases" not in self.agent.current_plan:
            logger.error("No valid plan exists")
            return None

        try:
            if self.agent.current_phase < 0 or self.agent.current_phase >= len(
                self.agent.current_plan["phases"]
            ):
                logger.error(f"Phase index {self.agent.current_phase} out of range")
                return None
            return self.agent.current_plan["phases"][self.agent.current_phase]
        except (IndexError, KeyError) as e:
            logger.error(f"Error getting current phase: {str(e)}")
            return None

    async def _get_current_step(self) -> Optional[str]:
        """Get the current step from the current phase"""
        current_phase = await self._get_current_phase()
        if not current_phase or "steps" not in current_phase:
            return None

        try:
            if self.agent.current_step < 0 or self.agent.current_step >= len(
                current_phase["steps"]
            ):
                logger.error(f"Step index {self.agent.current_step} out of range")
                return None
            return current_phase["steps"][self.agent.current_step]
        except (IndexError, KeyError) as e:
            logger.error(f"Error getting current step: {str(e)}")
            return None

    async def progress_to_next_step(self) -> bool:
        """Progress to the next step in the current phase or next phase"""
        if not self.agent.current_plan or "phases" not in self.agent.current_plan:
            logger.error("No valid plan exists for progression")
            return False

        current_phase = await self._get_current_phase()
        if not current_phase or "steps" not in current_phase:
            logger.error("No valid phase for progression")
            return False

        # Calculate next step
        next_step = self.agent.current_step + 1

        # If we've completed all steps in the current phase
        if next_step >= len(current_phase["steps"]):
            return await self.progress_to_next_phase()

        # Move to next step in current phase
        self.agent.current_step = next_step
        logger.info(
            f"Progressed to step {self.agent.current_step} in phase {self.agent.current_phase}"
        )
        await self.agent.update_todo_progress()
        return True

    async def progress_to_next_phase(self) -> bool:
        """Progress to the next phase and reset step counter"""
        if not self.agent.current_plan or "phases" not in self.agent.current_plan:
            logger.error("No valid plan exists for phase progression")
            return False

        # Calculate next phase
        next_phase = self.agent.current_phase + 1

        # Check if we've completed all phases
        if next_phase >= len(self.agent.current_plan["phases"]):
            logger.info("All phases complete!")
            raise AgentTaskComplete("All phases of the plan have been completed")

        # Move to next phase
        self.agent.current_phase = next_phase
        self.agent.current_step = 0  # Reset step counter for new phase
        logger.info(
            f"Progressed to phase {self.agent.current_phase}, step {self.agent.current_step}"
        )
        await self.agent.update_todo_progress()
        return True
