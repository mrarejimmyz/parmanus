#!/usr/bin/env python3
"""
ParManus AI - Getting Started Demo
Demonstrates basic computer control capabilities
"""

import asyncio
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

from app.tool.computer_control import ComputerControlTool


async def demo_computer_control():
    """Demonstrate basic computer control features"""
    print("🤖 ParManus AI - Computer Control Demo")
    print("=" * 50)

    tool = ComputerControlTool()

    print("1. 📸 Taking a screenshot...")
    result = await tool.execute(action="screenshot")
    if not result.error:
        print("   ✅ Screenshot captured successfully!")
        print(f"   📄 {result.output}")
    else:
        print(f"   ❌ Error: {result.error}")

    print("\n2. 🖱️ Getting mouse position...")
    result = await tool.execute(action="get_mouse_position")
    if not result.error:
        print("   ✅ Mouse position retrieved!")
        print(f"   📍 {result.output}")
    else:
        print(f"   ❌ Error: {result.error}")

    print("\n3. 💻 Getting system information...")
    result = await tool.execute(action="get_system_info")
    if not result.error:
        print("   ✅ System info retrieved!")
        print(f"   📊 {result.output[:200]}...")
    else:
        print(f"   ❌ Error: {result.error}")

    print("\n4. 📋 Testing clipboard operations...")
    # Set clipboard
    result = await tool.execute(action="set_clipboard", text="Hello from ParManus AI!")
    if not result.error:
        print("   ✅ Text copied to clipboard!")

    # Get clipboard
    result = await tool.execute(action="get_clipboard")
    if not result.error:
        print("   ✅ Clipboard content retrieved!")
        print(f"   📝 {result.output}")
    else:
        print(f"   ❌ Error: {result.error}")

    print("\n5. 🪟 Listing open windows...")
    result = await tool.execute(action="list_windows")
    if not result.error:
        print("   ✅ Windows listed successfully!")
        print(f"   🏠 {result.output[:300]}...")
    else:
        print(f"   ❌ Error: {result.error}")

    print("\n" + "=" * 50)
    print("🎉 Demo completed! ParManus AI computer control is working.")
    print("\n💡 Try these commands:")
    print("   python main.py --prompt 'Take a screenshot and tell me what you see'")
    print("   python main.py --prompt 'Open calculator and compute 25 * 47'")
    print("   python main.py --prompt 'Show me my system information'")
    print("\n📚 For more information, see INSTALLATION_AND_USAGE_GUIDE.md")


async def main():
    """Main demo function"""
    try:
        await demo_computer_control()
    except KeyboardInterrupt:
        print("\n⚠️ Demo interrupted by user")
    except Exception as e:
        print(f"\n❌ Demo failed: {str(e)}")
        print("💡 Make sure you've installed all dependencies:")
        print(
            "   pip install pyautogui pygetwindow pyperclip opencv-python screeninfo mss"
        )


if __name__ == "__main__":
    asyncio.run(main())
