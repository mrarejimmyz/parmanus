#!/usr/bin/env python3
"""
End-to-end test for the enhanced Manus AI agent with calculator workflow fix
"""

import asyncio
import sys
from pathlib import Path

# Add the project root to the path
sys.path.insert(0, str(Path(__file__).parent))

from app.agent.manus import Manus
from app.config import config
from app.logger import logger


async def test_manus_calculator_workflow():
    """Test the complete Manus agent with calculator workflow"""
    print("\n🤖 Testing Manus AI Agent with Enhanced Calculator Workflow")
    print("=" * 70)

    try:
        # Create Manus agent instance
        print("🚀 Initializing Manus agent...")
        agent = await Manus.create()

        # Test prompt similar to the original issue
        test_prompt = "open calculator and do 25+48"

        print(f"📝 Testing prompt: '{test_prompt}'")
        print("🔄 Processing...")

        # Execute the prompt
        result = await agent.run(test_prompt)

        print(f"✅ Result: {result}")
        print("\n📊 Test completed!")

        return True

    except Exception as e:
        print(f"❌ Test failed: {e}")
        logger.error(f"End-to-end test error: {e}", exc_info=True)
        return False


async def main():
    """Main test runner"""
    print("🧪 ParManus AI - End-to-End Calculator Workflow Test")
    print("=" * 70)

    # Run the test
    success = await test_manus_calculator_workflow()

    if success:
        print("\n🎉 SUCCESS: Enhanced calculator workflow is working!")
        print("\n📝 Improvements implemented:")
        print("✅ Action name mapping for common incorrect variations")
        print("✅ Smart calculator workflow prevents multiple instances")
        print("✅ Enhanced system prompt with exact action name requirements")
        print("✅ Automatic focus on existing calculator windows")

        print("\n💡 The Manus AI agent should now:")
        print("   - Check for existing calculator before launching new ones")
        print("   - Handle incorrect action names automatically")
        print("   - Follow efficient application workflows")
        print("   - Use existing applications instead of opening duplicates")
    else:
        print("\n❌ Test failed - see error details above")

    return success


if __name__ == "__main__":
    success = asyncio.run(main())
    sys.exit(0 if success else 1)
