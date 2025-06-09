#!/usr/bin/env python3
"""Test the Manus agent directly to isolate the tool call issue."""

import asyncio
import sys
from pathlib import Path

# Add the main directory to Python path
root_dir = Path(__file__).parent.absolute()
sys.path.insert(0, str(root_dir))


async def test_manus_direct():
    """Test Manus agent directly with a simple request."""
    print("Creating Manus agent...")

    try:
        from app.agent.manus import Manus

        # Create agent instance
        agent = await Manus.create()
        print("Manus agent created successfully")

        # Test with a simple request
        print("Testing with simple web page request...")
        result = await agent.run(
            "Create a simple HTML web page with a title and hello world message"
        )

        print(f"Result: {result}")

    except Exception as e:
        print(f"Error: {e}")
        import traceback

        traceback.print_exc()

    finally:
        # Clean up
        if "agent" in locals():
            await agent.cleanup()


if __name__ == "__main__":
    asyncio.run(test_manus_direct())
