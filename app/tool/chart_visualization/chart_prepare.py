from app.tool.chart_visualization.normal_python_execute import NormalPythonExecute


class VisualizationPrepare(NormalPythonExecute):
    """A tool for Chart Generation Preparation"""

    name: str = "visualization_preparation"
    description: str = (
        "Using Python code to Generates structured visualization datasets with metadata. Outputs: 1) Cleaned CSV data files 2) JSON info with csv path and visualization description."
    )
    parameters: dict = {
        "type": "object",
        "properties": {
            "code": {
                "type": "string",
                "description": """Python code template EXCLUSIVELY for visualization prepare. Must Contains:
1. Data loading logic (handle dataframe/dict/file/url/json/web crawler)
2. Csv Data and chart description generate
2.1 Csv data (The data you want to visulazation, cleaning / transform from origin data, saved in .csv)
2.2 Chart description of csv data (The chart title or description should be concise and clear. Examples: 'Product sales distribution', 'Monthly revenue trend'.)
3. Save information in json file.( format: {"csvFilePath": string, "chartTitle": string}[] encoding='utf-8')
3. Json file saving with path print: print(json_path)
# Note
You can generate one or multiple csv data with different visualization needs.
""",
            },
        },
        "required": ["code"],
    }

    async def execute(self, code: str, timeout=5):
        """
        Executes the provided Python code with a timeout.

        Args:
            code (str): The Python code to execute.
            analysis_content (str): The analysis content of current task.
            timeout (int): Execution timeout in seconds.

        Returns:
            Dict: Contains 'output' with execution output or error message and 'success' status.
        """
        return await super().execute(code, timeout)
