#!/usr/bin/env python3
"""Debug the empty completion issue."""

import asyncio
import sys
from pathlib import Path

# Add the main directory to Python path
root_dir = Path(__file__).parent.absolute()
sys.path.insert(0, str(root_dir))

from app.config import config
from app.llm_optimized import LLMOptimized
from app.tool import Bash, StrReplaceEditor, Terminate, ToolCollection


async def debug_empty_completion():
    """Debug why completion is empty."""
    print("=== Debugging Empty Completion Issue ===")

    try:
        # Initialize LLM
        llm = LLMOptimized(config.llm)
        print("✅ LLM initialized")

        # Create simple tools
        tool_collection = ToolCollection(Bash(), StrReplaceEditor(), Terminate())
        tools = tool_collection.to_params()
        print(f"✅ Tools created: {[t['function']['name'] for t in tools]}")

        # Test 1: Simple completion without tools first
        print("\n--- Test 1: Basic completion without tools ---")
        try:
            messages = [{"role": "user", "content": "Hello! Please say hi."}]
            basic_response = await llm.ask(messages, timeout=30)
            print(f"Basic response length: {len(basic_response)}")
            print(f"Basic response: {basic_response[:200]}...")

            if not basic_response or len(basic_response.strip()) < 5:
                print("❌ ISSUE: Basic completion is also empty!")
                return False
            else:
                print("✅ Basic completion works")

        except Exception as e:
            print(f"❌ Basic completion failed: {e}")
            return False

        # Test 2: Try tool completion with simple prompt
        print("\n--- Test 2: Tool completion with simple prompt ---")
        try:
            simple_tool_messages = [{"role": "user", "content": "Create a file called hello.txt"}]

            # Debug: Let's see what the prompt looks like
            print("Creating enhanced prompt...")

            # Format messages manually to see the prompt
            formatted_messages = [msg for msg in simple_tool_messages]
            prompt = llm._format_prompt_for_llama(formatted_messages)
            print(f"Base prompt length: {len(prompt)}")
            print(f"Base prompt preview: {prompt[:300]}...")

            # Add tool definitions
            from app.llm_tool_patch_optimized import (format_tool_definitions,
                                                      get_tool_instructions)
            from app.schema import ToolChoice

            tool_definitions = format_tool_definitions(tools)
            tool_instructions = get_tool_instructions(ToolChoice.AUTO)
            enhanced_prompt = f"{prompt}\n\n{tool_definitions}{tool_instructions}"

            print(f"Enhanced prompt length: {len(enhanced_prompt)}")
            print(f"Tool definitions length: {len(tool_definitions)}")
            print(f"Tool instructions: {tool_instructions}")

            print("\nTool definitions preview:")
            print(tool_definitions[:500] + "..." if len(tool_definitions) > 500 else tool_definitions)

            # Test the completion directly with the text model
            print("\n--- Test 3: Direct text model call ---")
            try:
                direct_completion = await asyncio.wait_for(
                    asyncio.get_event_loop().run_in_executor(
                        llm._executor,
                        lambda: llm.text_model.create_completion(
                            prompt=enhanced_prompt,
                            max_tokens=min(llm.max_tokens, 512),  # Smaller for testing
                            temperature=0.1,
                            stop=["<|user|>", "<|system|>"],
                        ),
                    ),
                    timeout=30,
                )

                completion_text = direct_completion.get("choices", [{}])[0].get("text", "").strip()
                print(f"Direct completion length: {len(completion_text)}")
                print(f"Direct completion: {completion_text[:300]}...")

                if not completion_text:
                    print("❌ ISSUE: Direct completion is empty!")
                    print("This suggests the model is immediately hitting a stop token or has another issue.")

                    # Try with different stop tokens
                    print("\n--- Test 4: Try without stop tokens ---")
                    no_stop_completion = await asyncio.wait_for(
                        asyncio.get_event_loop().run_in_executor(
                            llm._executor,
                            lambda: llm.text_model.create_completion(
                                prompt=enhanced_prompt,
                                max_tokens=100,
                                temperature=0.1,
                                stop=[],  # No stop tokens
                            ),
                        ),
                        timeout=30,
                    )

                    no_stop_text = no_stop_completion.get("choices", [{}])[0].get("text", "").strip()
                    print(f"No-stop completion length: {len(no_stop_text)}")
                    print(f"No-stop completion: {no_stop_text[:500]}...")

                else:
                    print("✅ Direct completion works")

            except Exception as e:
                print(f"❌ Direct completion failed: {e}")
                import traceback
                traceback.print_exc()

        except Exception as e:
            print(f"❌ Tool prompt setup failed: {e}")
            import traceback
            traceback.print_exc()

    except Exception as e:
        print(f"❌ Setup failed: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = asyncio.run(debug_empty_completion())
    print(f"\nDebugging completed: {'Success' if success else 'Failed'}")
