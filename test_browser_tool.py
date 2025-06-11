#!/usr/bin/env python3
"""
Test script for the browser tool functionality
"""
import asyncio
import os
import sys

# Add the project root to the Python path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from app.agent.router import AgentRouter
from app.config import config
from app.logger import logger


async def test_browser_functionality():
    """Test the browser tool with a comprehensive search task."""
    try:
        # Initialize the agent router (no initialize method needed)
        router = AgentRouter()

        # Test prompts to validate different aspects
        test_prompts = [
            "Test simple web navigation by going to google.com",
            "Search for news about Python programming language and extract key information",
            "Navigate to openai.com and tell me about their latest updates",
        ]

        for i, prompt in enumerate(test_prompts, 1):
            print(f"\n{'='*60}")
            print(f"TEST {i}: {prompt}")
            print("=" * 60)

            try:
                result = await router.route(prompt)
                print(f"✅ Test {i} completed successfully!")
                print(f"Result preview: {str(result)[:200]}...")

            except Exception as e:
                print(f"❌ Test {i} failed: {str(e)}")
                logger.error(f"Test {i} error: {e}")

            # Small delay between tests
            await asyncio.sleep(2)

    except Exception as e:
        print(f"❌ Fatal error during testing: {e}")
        logger.error(f"Fatal test error: {e}")
        return False

    return True


async def test_individual_browser_actions():
    """Test individual browser actions to identify specific issues."""
    try:
        from app.tool.browser_use_tool import BrowserUseTool

        browser_tool = BrowserUseTool()

        # Test individual actions
        actions_to_test = [
            {"action": "go_to_url", "url": "https://www.google.com"},
            {
                "action": "extract_content",
                "goal": "Get the page title and main search box information",
            },
            {"action": "web_search", "query": "Python programming news"},
        ]

        for i, action_params in enumerate(actions_to_test, 1):
            print(f"\n🔧 Testing browser action {i}: {action_params['action']}")

            try:
                result = await browser_tool.execute(**action_params)

                if result.error:
                    print(f"❌ Action failed: {result.error}")
                else:
                    print(f"✅ Action succeeded: {str(result.output)[:100]}...")

            except Exception as e:
                print(f"❌ Exception during action: {e}")
                logger.error(f"Browser action error: {e}")

        # Cleanup
        await browser_tool.cleanup()

    except Exception as e:
        print(f"❌ Error testing individual browser actions: {e}")
        logger.error(f"Individual browser action test error: {e}")


if __name__ == "__main__":
    print("🚀 Starting comprehensive ParManus AI browser tool testing...")
    print(f"Config loaded: {config}")

    # Run the tests
    asyncio.run(test_individual_browser_actions())
    print("\n" + "=" * 60)
    asyncio.run(test_browser_functionality())

    print("\n🏁 Testing completed!")
