import subprocess
import json
import threading
import base64
import pandas as pd
import aiofiles
import os
from typing import Any, Dict, Hashable
from pydantic import Field, model_validator

from app.llm import LLM
from app.tool.base import BaseTool
from app.tool.chart_visualization.utils import (
    run_code_in_thread,
    extract_executable_code,
)


class ChartVisualization(BaseTool):
    name: str = "generate_data_visualization"
    description: str = """Visualize a statistical chart using csv data and chart description. The tool accepts code to generate csv data and description of the chart, and output a chart in png or html.
Note: Each tool call generates a single chart.
"""
    parameters: dict = {
        "type": "object",
        "properties": {
            "code": {
                "type": "string",
                "description": """Python code template EXCLUSIVELY for CSV generation. MUST CONTAIN:
1. Data loading logic (handle dataframe/dict/file/url/json)
2. Data processing (cleaning/transformation)
3. CSV saving with path print (Only csv path): print(csv_path)""",
            },
            "chart_description": {
                "type": "string",
                "description": "The chart title or description should be concise and clear. Examples: 'Product sales distribution', 'Monthly revenue trend'.",
            },
            "output_type": {
                "description": "Rendering format (html=interactive)",
                "type": "string",
                "default": "html",
                "enum": ["png", "html"],
            },
        },
        "required": ["code", "chart_description"],
    }
    llm: LLM = Field(default_factory=LLM, description="Language model instance")

    @model_validator(mode="after")
    def initialize_llm(self):
        """Initialize llm with default settings if not provided."""
        if self.llm is None or not isinstance(self.llm, LLM):
            self.llm = LLM(config_name=self.name.lower())
        return self

    async def execute(self, code: str, chart_description: str, output_type: str) -> str:
        code_result = await self.execute_code(code=code)
        if "success" in code_result and code_result["success"] is False:
            return code_result
        if code_result["observation"].startswith("Error"):
            return {"observation": code_result["observation"], "success": False}

        try:
            data_path = (
                code_result["observation"].replace("\n", "").replace("\r", "").strip()
            )
            if not data_path.endswith(".csv"):
                return {
                    "observation": "Error: Code should ONLY output CSV data path",
                    "success": False,
                }
            df = pd.read_csv(data_path)
            df = df.astype(object)
            df = df.where(pd.notnull(df), None)
            data_dict_list = df.to_json(orient="records", force_ascii=False)
            result = await self.invoke_vmind(
                data_dict_list, chart_description, output_type
            )
            if "error" in result:
                return {
                    "observation": f"Error: {result["error"]}",
                    "success": False,
                }
            chart_file_path = data_path.replace(".csv", f".{output_type}")
            while os.path.exists(chart_file_path):
                chart_file_path = chart_file_path.replace(
                    f".{output_type}", f"_new.{output_type}"
                )
            if output_type == "png":
                byte_data = base64.b64decode(result["res"])
                async with aiofiles.open(chart_file_path, "wb") as file:
                    await file.write(byte_data)
            else:
                async with aiofiles.open(
                    chart_file_path, "w", encoding="utf-8"
                ) as file:
                    await file.write(result["res"])
            return {"observation": f"chart successfully saved to {chart_file_path}"}
        except Exception as e:
            return {
                "observation": f"Error: {e}",
                "success": False,
            }

    async def execute_code(
        self,
        code: str,
        timeout: int = 5,
    ) -> Dict:
        """
        Executes the provided Python code with a timeout.

        Args:
            code (str): The Python code to execute.
            timeout (int): Execution timeout in seconds.

        Returns:
            Dict: Contains 'output' with execution output or error message and 'success' status.
        """
        result = {"observation": ""}
        be_extracted_code = extract_executable_code(code)

        thread = threading.Thread(
            target=run_code_in_thread, args=(be_extracted_code, result)
        )
        thread.start()
        thread.join(timeout)

        if thread.is_alive():
            return {
                "observation": f"Execution timeout after {timeout} seconds",
                "success": False,
            }

        return result

    async def invoke_vmind(
        self,
        dict_data: list[dict[Hashable, Any]],
        chart_description: str,
        output_type: str,
    ):
        llm_config = {
            "base_url": self.llm.base_url,
            "model": self.llm.model,
            "api_key": self.llm.api_key,
        }
        vmind_params = {
            "llm_config": llm_config,
            "user_prompt": chart_description,
            "dataset": dict_data,
            "output_type": output_type,
        }
        process = subprocess.run(
            ["npx", "ts-node", "src/chartVisualize.ts"],
            input=json.dumps(vmind_params),
            capture_output=True,
            text=True,
            encoding="utf-8",
            cwd=os.path.dirname(__file__),
        )
        if process.returncode == 0:
            return json.loads(process.stdout)
        else:
            return {"error": f"Node.js Error: {process.stderr}"}
