import sys
from io import StringIO

from app.tool.python_execute import PythonExecute
from app.tool.chart_visualization.utils import (
    extract_executable_code,
)


class NormalPythonExecute(PythonExecute):
    """A tool for executing Python code with timeout and safety restrictions."""

    name: str = "common_python_execute"
    description: str = (
        """Executes Python code strings to do data analysis. Note:
1. Only outputs from print() are visible; function return values are not captured. Use print() statements to display results
2. Do data analysis (cleaning / transform) saved in *.csv
3. Generate a data analysis report in *.md"""
    )
    parameters: dict = {
        "type": "object",
        "properties": {
            "code": {
                "type": "string",
                "description": "The Python code to execute.",
            },
        },
        "required": ["code"],
    }

    def _run_code(self, code: str, result_dict: dict, safe_globals: dict) -> None:
        original_stdout = sys.stdout
        be_extracted_code = extract_executable_code(code)  # ignore_security_alert RCE
        try:
            output_buffer = StringIO()
            sys.stdout = output_buffer
            exec(  # ignore_security_alert RCE
                be_extracted_code, safe_globals, safe_globals
            )  # ignore_security_alert RCE
            result_dict["observation"] = output_buffer.getvalue()
            result_dict["success"] = True
        except Exception as e:
            result_dict["observation"] = str(e)
            result_dict["success"] = False
        finally:
            sys.stdout = original_stdout
