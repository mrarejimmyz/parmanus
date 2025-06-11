#!/usr/bin/env python3
"""
Simple direct test of computer control tool availability and action names
"""

import sys
from pathlib import Path

# Add the project root to the Python path
sys.path.insert(0, str(Path(__file__).parent))

from app.tool.computer_control import ComputerControlTool


def test_action_names_enum():
    """Test that all expected action names are in the enum"""
    print("🔍 Testing Computer Control Action Names Enum")
    print("=" * 50)

    tool = ComputerControlTool()
    expected_actions = [
        "screenshot",
        "screenshot_region",
        "mouse_click",
        "mouse_move",
        "mouse_drag",
        "mouse_scroll",
        "type_text",
        "send_keys",
        "key_combination",
        "launch_app",
        "close_app",
        "list_processes",
        "kill_process",
        "list_windows",
        "focus_window",
        "move_window",
        "resize_window",
        "minimize_window",
        "maximize_window",
        "close_window",
        "find_ui_element",
        "click_ui_element",
        "get_clipboard",
        "set_clipboard",
        "execute_command",
        "get_system_info",
        "get_mouse_position",
        "get_screen_info",
        "wait",
    ]

    # Get the actual enum from parameters
    actual_actions = tool.parameters["properties"]["action"]["enum"]

    print(f"📋 Expected actions: {len(expected_actions)}")
    print(f"🔧 Actual actions: {len(actual_actions)}")

    missing = set(expected_actions) - set(actual_actions)
    extra = set(actual_actions) - set(expected_actions)

    if missing:
        print(f"❌ Missing actions: {missing}")

    if extra:
        print(f"➕ Extra actions: {extra}")

    if not missing and not extra:
        print("✅ All action names match perfectly!")
        return True
    else:
        print("❌ Action name mismatch detected!")
        return False


def main():
    """Main test function"""
    print("🧪 Computer Control Action Names Validation")
    print("=" * 60)

    success = test_action_names_enum()

    print("\n" + "=" * 60)
    if success:
        print("🎉 ACTION NAMES VALIDATION PASSED!")
        print("✅ All expected action names are correctly defined")
        return 0
    else:
        print("💔 ACTION NAMES VALIDATION FAILED!")
        print("❌ Check the computer control tool action enum")
        return 1


if __name__ == "__main__":
    sys.exit(main())
