"""Debug script for tool call parsing in ParManusAI."""

import sys
import json
import re
from pathlib import Path

# Add parent directory to Python path
sys.path.append(str(Path(__file__).parent.absolute()))

# Import the necessary modules
try:
    from app.llm_tool_patch import _parse_tool_calls, _format_tool_call
    from app.logger import logger
    print("Successfully imported modules from app.llm_tool_patch")
except Exception as e:
    print(f"Failed to import modules: {e}")
    sys.exit(1)

# Set up basic logging
import logging
logging.basicConfig(level=logging.DEBUG)

# Sample completion text that would typically be returned from the LLM
sample_texts = [
    # Standard JSON format
    '{"name": "search", "arguments": {"query": "how to build a website"}}',

    # Function format with name inside function
    '{"function": {"name": "search", "arguments": {"query": "how to build a website"}}}',

    # XML format
    '<function_calls><invoke name="search"><parameter name="query">how to build a website</parameter></invoke></function_calls>',

    # Natural language format
    'I will use `search` with {"query": "how to build a website"}',

    # JSON code block
    '```json\n{"name": "search", "arguments": {"query": "how to build a website"}}\n```',

    # XML format with antml prefix
    '<function_calls>\n<invoke name="search">\n<parameter name="query">how to build a website
