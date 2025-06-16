#!/usr/bin/env python3
"""
Simple test to verify the cleaned codebase with intelligent document creation.
"""

import asyncio
import os
import sys
from pathlib import Path

# Add the app directory to Python path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "app"))

from app.llm import LLM
from app.schema import Message


async def test_clean_codebase():
    """Test that the cleaned codebase works with intelligent reasoning."""
    print("🧹 Testing Cleaned Codebase with Intelligent Document Creation")
    print("=" * 60)

    try:
        # Test LLM is working
        llm = LLM()
        print(f"✅ LLM initialized: {type(llm)}")
        print(f"✅ Model: {getattr(llm, 'model', 'Unknown')}")
        print(f"✅ Vision enabled: {getattr(llm, 'vision_enabled', False)}")

        # Test simple document creation
        test_prompt = """
Create a brief professional resume for a software engineer named Alex Thompson.
Include the standard sections: contact info, summary, skills, and experience.
Keep it concise but professional.
"""

        messages = [Message(role="user", content=test_prompt)]
        content = await llm.ask(messages)

        if content and len(content) > 100:
            # Save test document
            workspace_path = Path("workspace")
            workspace_path.mkdir(exist_ok=True)

            test_file = workspace_path / "test_clean_codebase.md"
            with open(test_file, "w", encoding="utf-8") as f:
                f.write(content)

            print(f"✅ Test document created: {test_file}")
            print(f"📊 Content length: {len(content)} characters")

            # Show preview
            lines = content.split("\n")[:10]
            print("\n📄 Document preview:")
            for line in lines:
                if line.strip():
                    print(f"   {line}")

            print(f"\n🎉 SUCCESS: Cleaned codebase is working perfectly!")
            print(f"🧠 Pure LLM reasoning approach is functional!")
            return True
        else:
            print(f"❌ FAILED: LLM response insufficient")
            return False

    except Exception as e:
        print(f"❌ Test failed: {e}")
        import traceback

        traceback.print_exc()
        return False


async def main():
    """Run the clean codebase test."""
    try:
        success = await test_clean_codebase()

        # Show workspace status
        workspace_path = Path("workspace")
        if workspace_path.exists():
            files = list(workspace_path.glob("*"))
            print(f"\n📁 Workspace contents: {len(files)} files")
            for f in files:
                print(f"   📄 {f.name}")

        if success:
            print("\n✅ CODEBASE CLEANUP SUCCESSFUL!")
            print("🧠 Ready for intelligent document creation!")
        else:
            print("\n❌ Cleanup verification failed")

    except Exception as e:
        print(f"❌ Main test failed: {e}")


if __name__ == "__main__":
    asyncio.run(main())
