from app.tool.python_execute import PythonExecute


class NormalPythonExecute(PythonExecute):
    """A tool for executing Python code with timeout and safety restrictions."""

    name: str = "common_python_execute"
    description: str = (
        "Execute Python code for in-depth data analysis without direct visualization. "
        "The code should generate a comprehensive text-based report containing dataset overview, "
        "column details, basic statistics, derived metrics, day-of-week comparisons, outliers, and key insights. "
        "Use print() for all outputs so the analysis (including sections like 'Dataset Overview' or 'Preprocessing Results') "
        "is clearly visible, and save any final report or processed files to config.workspace. "
        "Include try-except blocks for error handling, and provide the code as a single string with '\\n' for line breaks."
    )
    parameters: dict = {
        "type": "object",
        "properties": {
            "code": {
                "type": "string",
            },
        },
        "required": ["code"],
    }
