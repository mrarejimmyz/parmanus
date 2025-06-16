#!/usr/bin/env python3

import asyncio
import os
import sys
from pathlib import Path

# Add the app directory to Python path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "app"))

from app.llm import LLM
from app.schema import Message


async def create_air_india_report():
    print("Creating Air India crash report...")

    try:
        llm = LLM()

        reasoning_prompt = """
Create a professional news analysis report about the Air India crash incident.

INSTRUCTIONS:
1. Create a comprehensive news analysis report
2. Include executive summary, incident details, analysis, and implications
3. Use proper markdown formatting with headers and sections
4. Make the content substantial and professional (minimum 800 words)
5. Include sections for: Executive Summary, Incident Overview, Analysis, Impact Assessment, and Conclusion
6. Use professional news reporting tone

Generate the complete Air India crash report in markdown format:
"""

        messages = [Message(role="user", content=reasoning_prompt)]
        content = await llm.ask(messages)

        if content and len(content) > 500:
            workspace_path = Path("workspace")
            workspace_path.mkdir(exist_ok=True)

            report_file = workspace_path / "air_india_crash_report.md"
            with open(report_file, "w", encoding="utf-8") as f:
                f.write(content)

            print(f"SUCCESS: Air India crash report created: {report_file}")
            print(f"Content length: {len(content)} characters")

            return True
        else:
            print(f"ERROR: LLM response insufficient")
            return False

    except Exception as e:
        print(f"ERROR: Failed to create report: {e}")
        return False


async def main():
    print("Air India Crash Report Generation")
    print("=" * 40)

    success = await create_air_india_report()

    if success:
        print("SUCCESS: Air India crash report generated!")
    else:
        print("FAILED: Could not generate report")


if __name__ == "__main__":
    asyncio.run(main())
