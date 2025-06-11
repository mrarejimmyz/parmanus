#!/usr/bin/env python3
"""
Test script to verify computer control tool with correct action names
Tests all available actions to ensure they work properly
"""

import asyncio
import sys
import time
from pathlib import Path

# Add the project root to the Python path
sys.path.insert(0, str(Path(__file__).parent))

from app.tool.computer_control import ComputerControlTool


async def test_computer_control_actions():
    """Test computer control tool with correct action names"""
    print("🧪 Testing Computer Control Tool Actions")
    print("=" * 50)

    tool = ComputerControlTool()

    # Test cases with correct action names
    test_cases = [
        {
            "name": "Take Screenshot",
            "action": "screenshot",
            "params": {"action": "screenshot"},
        },
        {
            "name": "Get Mouse Position",
            "action": "get_mouse_position",
            "params": {"action": "get_mouse_position"},
        },
        {
            "name": "Get System Info",
            "action": "get_system_info",
            "params": {"action": "get_system_info"},
        },
        {
            "name": "Get Screen Info",
            "action": "get_screen_info",
            "params": {"action": "get_screen_info"},
        },
        {
            "name": "List Processes",
            "action": "list_processes",
            "params": {"action": "list_processes"},
        },
        {
            "name": "Get Clipboard",
            "action": "get_clipboard",
            "params": {"action": "get_clipboard"},
        },
        {
            "name": "Set Clipboard",
            "action": "set_clipboard",
            "params": {"action": "set_clipboard", "text": "Test clipboard content"},
        },
        {
            "name": "List Windows",
            "action": "list_windows",
            "params": {"action": "list_windows"},
        },
    ]

    results = []

    for test_case in test_cases:
        print(f"\n🔧 Testing: {test_case['name']}")
        print(f"   Action: {test_case['action']}")

        try:
            result = await tool.execute(**test_case["params"])

            if not result.error:  # Success means no error
                print(f"   ✅ SUCCESS")
                if hasattr(result, "output") and result.output:
                    # Truncate long outputs
                    output_preview = str(result.output)[:200]
                    if len(str(result.output)) > 200:
                        output_preview += "..."
                    print(f"   📄 Output: {output_preview}")
                results.append({"test": test_case["name"], "status": "PASS"})
            else:
                print(f"   ❌ FAILED: {result.error}")
                results.append(
                    {"test": test_case["name"], "status": "FAIL", "error": result.error}
                )

        except Exception as e:
            print(f"   💥 EXCEPTION: {str(e)}")
            results.append(
                {"test": test_case["name"], "status": "ERROR", "error": str(e)}
            )

        # Small delay between tests
        await asyncio.sleep(0.5)

    print("\n" + "=" * 50)
    print("📊 TEST RESULTS SUMMARY")
    print("=" * 50)

    passed = sum(1 for r in results if r["status"] == "PASS")
    failed = sum(1 for r in results if r["status"] in ["FAIL", "ERROR"])

    print(f"✅ PASSED: {passed}")
    print(f"❌ FAILED: {failed}")
    print(f"📈 SUCCESS RATE: {(passed / len(results)) * 100:.1f}%")

    print("\nDetailed Results:")
    for result in results:
        status_icon = "✅" if result["status"] == "PASS" else "❌"
        print(f"  {status_icon} {result['test']}: {result['status']}")
        if "error" in result:
            print(f"      Error: {result['error']}")

    return passed == len(results)


async def test_action_name_validation():
    """Test that invalid action names are properly rejected"""
    print("\n🔒 Testing Invalid Action Name Rejection")
    print("=" * 50)

    tool = ComputerControlTool()

    invalid_actions = [
        "capture_screenshot",  # Wrong name - should be "screenshot"
        "take_screenshot",  # Wrong name - should be "screenshot"
        "click_mouse",  # Wrong name - should be "mouse_click"
        "move_mouse",  # Wrong name - should be "mouse_move"
        "invalid_action",  # Completely invalid
    ]

    for invalid_action in invalid_actions:
        print(f"\n🚫 Testing invalid action: {invalid_action}")
        try:
            result = await tool.execute(action=invalid_action)
            if result.error:  # Failed means there's an error
                print(f"   ✅ Correctly rejected invalid action")
            else:
                print(f"   ❌ FAILED: Invalid action was accepted!")
        except Exception as e:
            print(f"   ✅ Correctly raised exception: {str(e)}")


async def main():
    """Main test function"""
    print("🚀 Starting Computer Control Action Names Test")
    print("=" * 60)

    # Test valid actions
    valid_tests_passed = await test_computer_control_actions()

    # Test invalid action rejection
    await test_action_name_validation()

    print("\n" + "=" * 60)
    if valid_tests_passed:
        print("🎉 ALL TESTS PASSED! Computer control tool is working correctly.")
        print("✅ Action names are properly implemented and recognized.")
        return 0
    else:
        print(
            "💔 SOME TESTS FAILED! Please check the computer control tool implementation."
        )
        return 1


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
