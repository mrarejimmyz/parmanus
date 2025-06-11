#!/usr/bin/env python3
"""
Integration test for Manus agent with computer control capabilities
Tests that the agent can correctly use computer control actions
"""

import asyncio
import sys
from pathlib import Path

# Add the project root to the Python path
sys.path.insert(0, str(Path(__file__).parent))

from app.agent.manus import Manus
from app.config import get_config


async def test_manus_computer_control():
    """Test Manus agent with computer control capabilities"""
    print("🤖 Testing Manus Agent with Computer Control")
    print("=" * 50)

    try:
        # Load config
        config = get_config()

        # Initialize Manus agent
        print("📚 Initializing Manus agent...")
        agent = Manus()

        # Test simple computer control task
        print("\n🖥️ Testing computer control action...")
        test_prompt = "Take a screenshot and tell me the screen dimensions"

        print(f"📝 Prompt: {test_prompt}")
        print("🔄 Processing...")

        # Run the agent
        response = await agent.run(test_prompt)

        print(f"\n🤖 Agent Response:")
        print(f"{response}")

        print("\n✅ Test completed successfully!")
        return True

    except Exception as e:
        print(f"\n❌ Test failed with error: {str(e)}")
        import traceback

        traceback.print_exc()
        return False


async def main():
    """Main test function"""
    print("🚀 Starting Manus Computer Control Integration Test")
    print("=" * 60)

    success = await test_manus_computer_control()

    print("\n" + "=" * 60)
    if success:
        print("🎉 INTEGRATION TEST PASSED!")
        print("✅ Manus agent can successfully use computer control capabilities")
        return 0
    else:
        print("💔 INTEGRATION TEST FAILED!")
        print("❌ Check the configuration and tool integration")
        return 1


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
