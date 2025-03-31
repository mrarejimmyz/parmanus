import json
import asyncio
import pandas as pd
import os
from typing import Any, Hashable
from pydantic import Field, model_validator

from app.llm import LLM
from app.tool.base import BaseTool
from app.logger import logger
from app.config import config


class ChartVisualization(BaseTool):
    name: str = "data_visualization"
    description: str = (
        """Visualize statistical chart with JSON info from visualization_preparation tool. Outputs: 1) Charts (png/html) 2) Charts Insights (.md)(Optional)."""
    )
    parameters: dict = {
        "type": "object",
        "properties": {
            "json_path": {
                "type": "string",
                "description": """file path of json info with ".json" in the end""",
            },
            "output_type": {
                "description": "Rendering format (html=interactive)",
                "type": "string",
                "default": "html",
                "enum": ["png", "html"],
            },
        },
        "required": ["code"],
    }
    llm: LLM = Field(default_factory=LLM, description="Language model instance")

    @model_validator(mode="after")
    def initialize_llm(self):
        """Initialize llm with default settings if not provided."""
        if self.llm is None or not isinstance(self.llm, LLM):
            self.llm = LLM(config_name=self.name.lower())
        return self

    def get_csv_path(self, json_info: list[dict[str, str]]) -> list[str]:
        res = []
        for item in json_info:
            if os.path.exists(item["csvFilePath"]):
                res.append(item["csvFilePath"])
            elif os.path.exists(
                os.path.join(f"{config.workspace_root}", item["csvFilePath"])
            ):
                res.append(
                    os.path.join(f"{config.workspace_root}", item["csvFilePath"])
                )
            else:
                raise Exception(f"No such file or directory: {item["csvFilePath"]}")
        return res

    def success_output_template(self, result: list[dict[str, str]]) -> str:
        content = ""
        if len(result) == 0:
            return "Is EMPTY!"
        for item in result:
            content += f"""## {item["title"]}\nChart saved in: {item["chart_path"]}"""
            if "insight_path" in item and item["insight_path"]:
                content += f"""\nChart insights saved in {item["insight_path"]}\n"""
            else:
                content += "\n"
        return f"Chart Generated Successful! Detail is below:\n{content}"

    async def execute(self, json_path: str, output_type: str) -> str:
        logger.info(f"📈 Chart Generation with json path: {json_path} ")
        try:
            with open(json_path, "r", encoding="utf-8") as file:
                json_info = json.load(file)
            data_list = []
            csv_file_path = self.get_csv_path(json_info)
            for index, item in enumerate(json_info):
                df = pd.read_csv(csv_file_path[index], encoding="utf-8")
                df = df.astype(object)
                df = df.where(pd.notnull(df), None)
                data_dict_list = df.to_json(orient="records", force_ascii=False)

                data_list.append(
                    {
                        "file_name": os.path.basename(csv_file_path[index]).replace(
                            ".csv", ""
                        ),
                        "dict_data": data_dict_list,
                        "chartTitle": item["chartTitle"],
                    }
                )
            tasks = [
                self.invoke_vmind(
                    item["dict_data"],
                    item["chartTitle"],
                    item["file_name"],
                    output_type,
                )
                for item in data_list
            ]

            results = await asyncio.gather(*tasks)
            error_list = []
            success_list = []
            for index, result in enumerate(results):
                csv_path = csv_file_path[index]
                if "error" in result and "chart_path" not in result:
                    error_list.append(f"Error in {csv_path}: {result["error"]}")
                else:
                    success_list.append(
                        {
                            **result,
                            "title": json_info[index]["chartTitle"],
                        }
                    )
            if len(error_list) > 0:
                return {
                    "observation": f"# Error chart generated{'\n'.join(error_list)}\n{self.success_output_template(success_list)}",
                    "success": False,
                }
            else:
                return {"observation": f"{self.success_output_template(success_list)}"}
        except Exception as e:
            return {
                "observation": f"Error: {e}",
                "success": False,
            }

    async def invoke_vmind(
        self,
        dict_data: list[dict[Hashable, Any]],
        chart_description: str,
        file_name: str,
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
            "file_name": file_name,
            "directory": str(config.workspace_root),
        }
        # build async sub process
        process = await asyncio.create_subprocess_exec(
            "npx",
            "ts-node",
            "src/chartVisualize.ts",
            stdin=asyncio.subprocess.PIPE,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
            cwd=os.path.dirname(__file__),
        )
        input_json = json.dumps(vmind_params, ensure_ascii=False).encode("utf-8")
        try:
            stdout, stderr = await process.communicate(input_json)
            stdout_str = stdout.decode("utf-8")
            stderr_str = stderr.decode("utf-8")
            if process.returncode == 0:
                return json.loads(stdout_str)
            else:
                return {"error": f"Node.js Error: {stderr_str}"}
        except Exception as e:
            return {"error": f"Subprocess Error: {str(e)}"}
