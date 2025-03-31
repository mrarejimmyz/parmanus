from app.tool.python_execute import PythonExecute


class VisualizationPrepare(PythonExecute):
    """A tool for Chart Generation Preparation"""

    name: str = "visualization_preparation"
    description: str = (
        "Using Python code to Generates metadata of data_visualization tool. Outputs: 1) Cleaned CSV data files 2) JSON info with csv path and visualization description."
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
3. Save information in json file.( format: {"csvFilePath": string, "chartTitle": string}[])
4. Json file saving with path print: print(json_path)
# Note
1. You can generate one or multiple csv data with different visualization needs.
2. Make each chart data esay, clean and different.
3. save/read in utf-8
""",
            },
        },
        "required": ["code"],
    }
