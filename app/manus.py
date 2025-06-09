from typing import Annotated, Optional, Union

from pydantic import AfterValidator, BaseModel, Field, model_validator

from .gpu_manager import CUDAGPUManager
from .llm_optimized import LLMOptimized


def validate_llm_instance(v: LLMOptimized) -> LLMOptimized:
    """Validate that v is an instance of LLMOptimized."""
    if not isinstance(v, LLMOptimized):
        raise ValueError("LLM must be an instance of LLMOptimized")
    return v


class Manus(BaseModel):
    """Main Manus AI class."""

    llm: Annotated[LLMOptimized, AfterValidator(validate_llm_instance)]
    gpu_manager: Optional[CUDAGPUManager] = Field(default_factory=CUDAGPUManager)

    class Config:
        arbitrary_types_allowed = True

    @model_validator(mode="after")
    def validate_llm(self) -> "Manus":
        """Validate LLM instance after model creation and share GPU manager."""
        # Ensure GPU manager is passed to LLM
        if self.gpu_manager and not self.llm.gpu_manager:
            self.llm.gpu_manager = self.gpu_manager
        return self
