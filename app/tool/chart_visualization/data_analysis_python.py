from app.tool.chart_visualization.normal_python_execute import NormalPythonExecute


class DataAnalysisPythonExecute(NormalPythonExecute):
    """A tool for executing Python code in data analysis task with timeout and safety restrictions."""

    name: str = "data_analysis_python_execute"
    description: str = (
        "Executes Python code string in data analysis task, save data table in csv file. Note: Only print outputs are visible, function return values are not captured. Use print statements to see results."
    )
    parameters: dict = {
        "type": "object",
        "properties": {
            "code": {
                "type": "string",
                "description": """Python code template EXCLUSIVELY for data analysis. Must Contains:
1. Data loading logic (handle dataframe/dict/file/url/json/web crawler)
2. Data analysis (cleaning/transformation)
3. CSV saving with path print: print(csv_path)
""",
            },
            "analysis_content": {
                "type": "string",
                "description": "Your analysis of current task, ensure your analysis is concise, clear, and easy to understand.",
            },
        },
        "required": ["code"],
    }

    async def execute(self, code: str, analysis_content: str, timeout=5):
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
