import sys
from io import StringIO

from app.tool.chart_visualization.utils import extract_executable_code
from app.tool.python_execute import PythonExecute


class NormalPythonExecute(PythonExecute):
    """A tool for executing Python code with timeout and safety restrictions."""

    name: str = "common_python_execute"
    description: str = (
        """
            Execute Python code for data analysis tasks without visualization. Important notes:

            1. Output: Only print() statements are visible. Use print() for all outputs.
            2. Data Processing: Load, clean, and transform data. Save results as CSV files if needed.
            3. Analysis: Perform statistical analysis, aggregations, and data exploration.
            4. Code Format: Provide code as a single string, use '\\n' for line breaks.
            5. File Paths: Use './data/' for relative paths to data files.
            6. Error Handling: Include try-except blocks for robust error management.
            7. No Visualization: This tool is for data analysis only, not for creating charts or plots.
            8. Analysis Results: Generate a comprehensive analysis report and save it in the './data/' directory.

            The analysis report should include:
            - Dataset overview (rows, columns, data types)
            - Basic statistics (averages, maximums, minimums for key metrics)
            - Initial observations and insights
            - Any patterns or trends identified in the data

    """
    )
    parameters: dict = {
        "type": "object",
        "properties": {
            "code": {
                "type": "string",
                "default": "html",
                "enum": ["process", "report", "others"],
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
