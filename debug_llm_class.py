#!/usr/bin/env python3
"""
Debug which LLM class is being used
"""

import os
import sys

# Add the app directory to Python path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "app"))

# Import and apply patches first
from app.llm_tool_patch_fix import patch_llm_class

patch_llm_class()

from app.llm import LLM


def debug_llm_class():
    """Debug the LLM class"""

    print(f"🔍 LLM class: {LLM}")
    print(f"🔍 LLM class name: {LLM.__name__}")
    print(f"🔍 LLM module: {LLM.__module__}")
    print(f"🔍 LLM MRO: {LLM.__mro__}")

    # Create instance
    llm = LLM()
    print(f"🔍 LLM instance type: {type(llm)}")
    print(f"🔍 LLM instance class: {llm.__class__}")

    # Check attributes
    print(f"🔍 Has text_model property: {hasattr(llm, 'text_model')}")
    print(f"🔍 Has ask_tool method: {hasattr(llm, 'ask_tool')}")
    print(f"🔍 Has _text_model_key: {hasattr(llm, '_text_model_key')}")

    # Check if it's optimized
    print(
        f"🔍 Instance attributes: {[attr for attr in dir(llm) if not attr.startswith('__')][:10]}"
    )


if __name__ == "__main__":
    debug_llm_class()
