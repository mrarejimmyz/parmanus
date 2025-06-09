#!/usr/bin/env python3
"""
Test if patches are being applied correctly
"""

import os
import sys

# Add the app directory to Python path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "app"))


def test_patches():
    """Test if patches are applied"""

    print("🔧 Testing patch application...")

    # Import and apply patches manually
    try:
        from app.llm_tool_patch_fix import patch_llm_class

        print("✅ Successfully imported patch_llm_class")

        # Apply patches
        patch_llm_class()
        print("✅ Patches applied successfully")

    except Exception as e:
        print(f"❌ Error applying patches: {e}")
        import traceback

        traceback.print_exc()
        return False

    # Test if LLM has the methods
    try:
        from app.llm import LLM

        # Create LLM instance
        llm = LLM()
        print(f"✅ LLM instance created: {type(llm)}")

        # Check if ask_tool method exists
        if hasattr(llm, "ask_tool"):
            print("✅ ask_tool method found")
        else:
            print("❌ ask_tool method NOT found")
            return False

        # Check if _parse_tool_calls method exists
        if hasattr(llm, "_parse_tool_calls"):
            print("✅ _parse_tool_calls method found")
        else:
            print("❌ _parse_tool_calls method NOT found")
            return False

        print("🎉 All patches applied successfully!")
        return True

    except Exception as e:
        print(f"❌ Error testing LLM: {e}")
        import traceback

        traceback.print_exc()
        return False


if __name__ == "__main__":
    test_patches()
