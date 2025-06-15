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

        # Check if we can move to next step in current phase
        if self.agent.current_step + 1 < len(current_phase["steps"]):
            self.agent.current_step += 1
            logger.info(
                f"Progressed to step {self.agent.current_step} in phase {self.agent.current_phase}"
            )
            await self.agent.update_todo_progress()
            return True
        else:
            # Move to next phase
            return await self.progress_to_next_phase()

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

    async def recover_from_invalid_position(self) -> bool:
        """Recover from invalid position by resetting to valid indices"""
        try:
            if not self.agent.current_plan or "phases" not in self.agent.current_plan:
                logger.error("Cannot recover: no valid plan exists")
                return False
            
            # Reset phase if out of bounds
            if self.agent.current_phase >= len(self.agent.current_plan["phases"]):
                self.agent.current_phase = len(self.agent.current_plan["phases"]) - 1
            elif self.agent.current_phase < 0:
                self.agent.current_phase = 0
            
            # Reset step if out of bounds
            current_phase = self.agent.current_plan["phases"][self.agent.current_phase]
            if "steps" in current_phase:
                if self.agent.current_step >= len(current_phase["steps"]):
                    # If we're past the last step, try to move to next phase
                    if self.agent.current_phase + 1 < len(self.agent.current_plan["phases"]):
                        logger.info("Moving to next phase during recovery")
                        self.agent.current_phase += 1
                        self.agent.current_step = 0
                    else:
                        # We're at the last phase, set to last valid step
                        self.agent.current_step = len(current_phase["steps"]) - 1
                elif self.agent.current_step < 0:
                    self.agent.current_step = 0
            else:
                self.agent.current_step = 0
            
            logger.info(f"Position recovered to phase {self.agent.current_phase}, step {self.agent.current_step}")
            return True
            
        except Exception as e:
            logger.error(f"Error during position recovery: {str(e)}")
            return False

    async def sync_with_base_framework(self) -> bool:
        """Synchronize our step tracking with the base framework's step counter"""
        try:
            # This method helps coordinate between our plan-based step tracking
            # and the base framework's automatic step incrementing
            if not self.agent.current_plan or "phases" not in self.agent.current_plan:
                return False
            
            current_phase = await self._get_current_phase()
            if not current_phase or "steps" not in current_phase:
                return False
            
            # If we've completed all steps in current phase, signal completion
            if self.agent.current_step >= len(current_phase["steps"]):
                logger.info("All steps in current phase completed")
                return await self.progress_to_next_phase()
            
            return True
            
        except Exception as e:
            logger.error(f"Error syncing with base framework: {str(e)}")
            return False

