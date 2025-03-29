import sys
from io import StringIO

from app.tool.chart_visualization.utils import extract_executable_code
from app.tool.python_execute import PythonExecute


class NormalPythonExecute(PythonExecute):
    """A tool for executing Python code with timeout and safety restrictions."""

    name: str = "common_python_execute"
    description: str = (
        """
A tool for executing Python code with data anaylsis.
Prefix: 帮我生成结果保存在本地./data下

Data Analysis Agent Protocol (Non-Visual)

=== Core Requirements ===
1. Strictly text-based outputs only
2. Prohibited actions:
   - Any chart/image generation
   - Interactive visual elements
   - Graphical libraries import

=== Execution Phases ===

1. DATA LOADING (Auto-detect format)
- Supported formats: CSV/Excel/JSON
- Mandatory checks:
  a) File existence verification
  b) Column structure validation
  c) Basic integrity checks

2. ANALYSIS PIPELINE
- Cleaning:
  • Null handling (drop or impute)
  • Deduplication
  • Outlier treatment (IQR/Z-score)

- Transformation:
  • Date parsing
  • Derived metrics
  • Aggregations

3. REPORT GENERATION
Output 1: data_exploration.md
┌──────────────────────┬──────────────────────────────┐
│ Section              │ Content Requirements         │
├──────────────────────┼──────────────────────────────┤
│ Dataset Metadata     │ Rows/Columns/Temporal Range │
│ Column Descriptions  │ Type/Stats/Unique Values    │
│ Key Findings         │ 3-5 bullet points           │
└──────────────────────┴──────────────────────────────┘

Output 2: preprocessing_results.md
┌──────────────────────┬──────────────────────────────┐
│ Section              │ Content Requirements         │
├──────────────────────┼──────────────────────────────┤
│ Cleaning Log         │ Rows affected by each operation │
│ Derived Metrics      │ Formula/Summary Stats       │
│ Anomaly Report       │ Z-score >2.5 cases          │
└──────────────────────┴──────────────────────────────┘

=== Implementation Rules ===
1. Code Generation Constraints:
   - Forbidden libraries: matplotlib, seaborn, plotly
   - Maximum column width: 120 chars
   - Required docstrings for all functions

2. Error Handling:
   - Skip corrupted records with logging
   - Continue processing after non-critical errors
   - Fail fast on structural issues

3. Output Validation:
   - Markdown syntax check
   - Statistical validity verification
   - Cross-report consistency

=== Sample Invocation ===
def analyze(data_path):
    '''Main analysis workflow'''
    df = load_data(data_path)          # Phase 1
    cleaned = clean_and_transform(df)  # Phase 2
    generate_reports(cleaned)          # Phase 3
=== 执行约束 ===
当检测到错误时：
1. 分析错误类型（数据/逻辑/环境）
2. 生成修正方案（自动重试≤3次）
3. 严重错误时回滚中间文件
"""
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
