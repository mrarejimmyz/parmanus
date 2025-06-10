#!/usr/bin/env python3
"""
Debug script to test tool loading and schema generation
"""
import asyncio
import sys


# Add the project root to the path
sys.path.insert(0, "/home/ubuntu/ParManusAI")

from app.agent.manus import Manus
from app.config import get_config


async def debug_tools():
    """Debug tool loading and schema generation"""
    print("🔍 Debugging ParManusAI Tool System")
    print("=" * 50)

    # Load config
    get_config("/home/ubuntu/ParManusAI/config/gpu_optimized.toml")
    print(f"✅ Config loaded")

    # Create Manus agent
    agent = await Manus.create()
    print(f"✅ Manus agent created")

    # Check available tools
    print(f"\n🛠️ Available Tools ({len(agent.available_tools.tools)}):")
    for i, tool in enumerate(agent.available_tools.tools, 1):
        print(f"  {i}. {tool.name}: {tool.description}")

    # Check tool parameters
    print(f"\n📋 Tool Parameters Schema:")
    tool_params = agent.available_tools.to_params()
    for i, param in enumerate(tool_params, 1):
        print(f"  {i}. {param['function']['name']}")
        print(f"     Description: {param['function']['description']}")
        if param["function"].get("parameters"):
            print(
                f"     Parameters: {list(param['function']['parameters'].get('properties', {}).keys())}"
            )
        else:
            print(f"     Parameters: None")
        print()

    # Test a simple prompt
    print(f"🧪 Testing simple prompt...")
    agent.memory.add_message(
        {"role": "user", "content": "List the available tools you have access to"}
    )

    try:
        result = await agent.think()
        print(f"✅ Think result: {result}")

        if agent.tool_calls:
            print(f"🔧 Tool calls made: {[tc.function.name for tc in agent.tool_calls]}")
        else:
            print(f"❌ No tool calls made")

        # Check the last message
        if agent.messages:
            last_msg = agent.messages[-1]
            print(f"📝 Last message content: {last_msg.content[:200]}...")

    except Exception as e:
        print(f"❌ Error during think: {e}")
        import traceback

        traceback.print_exc()

    # Cleanup
    await agent.cleanup()
    print(f"\n✅ Debug complete")


if __name__ == "__main__":
    asyncio.run(debug_tools())
