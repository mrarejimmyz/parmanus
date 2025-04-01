from app.tool.python_execute import PythonExecute


class NormalPythonExecute(PythonExecute):
    """A tool for executing Python code with timeout and safety restrictions."""

    name: str = "common_python_execute"
    description: str = (
        """Executes Python code strings to tasks such as data process and data report"""
    )
    parameters: dict = {
        "type": "object",
        "properties": {
            "code": {
                "type": "string",
                "description": """The Python code to execute. Note:
1. Only outputs from print() are visible; function return values are not captured. Use print() statements to display results
2. Do data process (cleaning / transform) saved in *.csv
3. Generate a data analysis report in html""",
            },
            "code_type": {
                "description": "code type",
                "type": "string",
                "default": "process",
                "enum": ["process", "report", "others"],
            },
        },
        "required": ["code"],
    }

    async def execute(self, code: str, code_type: str | None = None, timeout=5):
        return await super().execute(code, timeout)
