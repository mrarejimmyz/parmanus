from pydantic import Field

from app.agent.toolcall import ToolCallAgent
from app.config import config
from app.prompt.visualization import NEXT_STEP_PROMPT, SYSTEM_PROMPT
from app.tool import Terminate, ToolCollection
from app.tool.chart_visualization.chart_visualization import ChartVisualization
from app.tool.chart_visualization.normal_python_execute import NormalPythonExecute
from app.tool.chart_visualization.chart_prepare import (
    VisualizationPrepare,
)


class DataAnalysis(ToolCallAgent):
    """
    A data analysis agent that uses planning to solve various data analysis tasks.

    This agent extends BrowserAgent with a comprehensive set of tools and capabilities,
    including Python execution, web browsing, chart visualization.
    """

    name: str = "DataAnalysis"
    description: str = (
        "An analytical agent that utilizes multiple tools to solve diverse data tasks"
    )

    system_prompt: str = SYSTEM_PROMPT.format(directory=config.workspace_root)
    next_step_prompt: str = NEXT_STEP_PROMPT

    max_observe: int = 10000
    max_steps: int = 20

    # Add general-purpose tools to the tool collection
    available_tools: ToolCollection = Field(
        default_factory=lambda: ToolCollection(
            NormalPythonExecute(),
            VisualizationPrepare(),
            ChartVisualization(),
            Terminate(),
        )
    )
