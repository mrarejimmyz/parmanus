#!/usr/bin/env python3
"""
🚀 ParManus AI - Phase 1 Unified Entry Point

This represents Phase 1 of the roadmap: Foundation & Stability
- Single entry point with multiple modes
- Backward compatibility with existing functionality
- Improved error handling and logging
- Configuration validation

Usage Examples:
  python parmanus.py --mode simple --prompt "hello world"
  python parmanus.py --mode full --prompt "create a webpage"
  python parmanus.py --interactive
"""

import argparse
import asyncio
import os
import sys
import traceback
from pathlib import Path
from typing import Optional

# Ensure we can import from app
sys.path.insert(0, str(Path(__file__).parent))

__version__ = "2.0.0-phase1"


class ParManusRunner:
    """Unified runner for all ParManus modes"""

    def __init__(self):
        self.mode = None
        self.config = None
        self.initialized = False

    async def run_simple_mode(self, args):
        """Run in simple mode using main_simple.py functionality"""
        try:
            print(f"🚀 ParManus AI v{__version__} - Simple Mode")

            # Import and use the simple mode
            from main_simple import main as simple_main

            # Temporarily override sys.argv for main_simple
            original_argv = sys.argv.copy()
            sys.argv = ["main_simple.py"]

            if args.prompt:
                sys.argv.extend(["--prompt", args.prompt])
            if args.no_wait:
                sys.argv.append("--no-wait")
            if args.api_type:
                sys.argv.extend(["--api-type", args.api_type])
            if args.config:
                sys.argv.extend(["--config", args.config])
            if args.agent:
                sys.argv.extend(["--agent", args.agent])

            # Run simple main
            await simple_main()

            # Restore original argv
            sys.argv = original_argv

        except Exception as e:
            print(f"❌ Simple mode error: {e}")
            print(
                f"💡 Try: python main_simple.py --prompt '{args.prompt if args.prompt else 'hello'}'"
            )

    async def run_full_mode(self, args):
        """Run in full mode using main.py functionality"""
        try:
            print(f"🚀 ParManus AI v{__version__} - Full Mode")

            # Import main components
            from main import main as full_main

            # Temporarily override sys.argv for main.py
            original_argv = sys.argv.copy()
            sys.argv = ["main.py"]

            if args.prompt:
                sys.argv.extend(["--prompt", args.prompt])
            if args.no_wait:
                sys.argv.append("--no-wait")
            if args.config:
                sys.argv.extend(["--config", args.config])
            if args.workspace:
                sys.argv.extend(["--workspace", args.workspace])
            if args.agent:
                sys.argv.extend(["--agent", args.agent])
            if args.max_steps:
                sys.argv.extend(["--max-steps", str(args.max_steps)])

            # Run full main
            await full_main()

            # Restore original argv
            sys.argv = original_argv

        except Exception as e:
            print(f"❌ Full mode error: {e}")
            print(
                f"💡 Try: python main.py --prompt '{args.prompt if args.prompt else 'hello'}'"
            )

    async def run_mcp_mode(self, args):
        """Run in MCP mode using run_mcp.py functionality"""
        try:
            print(f"🚀 ParManus AI v{__version__} - MCP Mode")

            # Import MCP runner
            from run_mcp import run_mcp

            # Temporarily override sys.argv for run_mcp.py
            original_argv = sys.argv.copy()
            sys.argv = ["run_mcp.py"]

            if args.prompt:
                sys.argv.extend(["--prompt", args.prompt])
            if args.interactive:
                sys.argv.append("--interactive")
            if args.server_url:
                sys.argv.extend(["--server-url", args.server_url])

            # Run MCP
            await run_mcp()

            # Restore original argv
            sys.argv = original_argv

        except Exception as e:
            print(f"❌ MCP mode error: {e}")
            print(
                f"💡 Try: python run_mcp.py --prompt '{args.prompt if args.prompt else 'hello'}'"
            )

    async def run_hybrid_mode(self, args):
        """Auto-detect and run the best available mode"""
        print(f"🚀 ParManus AI v{__version__} - Hybrid Mode (Auto-detect)")

        # Try full mode first
        try:
            # Test if full mode dependencies are available
            from app.agent.router import AgentRouter
            from app.config import get_config
            from app.llm_factory import create_llm_async

            print("✅ Full mode available - using advanced features")
            await self.run_full_mode(args)
            return

        except Exception as e:
            print(f"⚠️ Full mode not available: {e}")

        # Fall back to simple mode
        try:
            print("🔄 Falling back to simple mode")
            await self.run_simple_mode(args)
            return

        except Exception as e:
            print(f"❌ Simple mode also failed: {e}")

        # Last resort: direct LLM test
        print("🆘 Testing basic LLM connectivity...")
        try:
            from app.config import load_config
            from app.llm_factory import create_llm

            config = load_config(args.config)
            llm = create_llm(config.llm)

            if args.prompt:
                print(f"🧠 Direct LLM response:")
                response = await llm.ask([{"role": "user", "content": args.prompt}])
                print(f"🤖 {response}")
            else:
                print("✅ Basic LLM connection working")
                print("💡 Try with --prompt 'hello' to test response")

        except Exception as e:
            print(f"❌ All modes failed: {e}")
            print("🔧 Please check your configuration and dependencies")


def create_parser():
    """Create the command line parser"""
    parser = argparse.ArgumentParser(
        description=f"ParManus AI v{__version__} - Unified Entry Point",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
🎯 Mode Selection:
  simple   - Lightweight mode with basic functionality
  full     - Complete feature set with advanced tools
  mcp      - Model Context Protocol server mode
  hybrid   - Auto-detect best available mode (default)

📝 Examples:
  %(prog)s --mode simple --prompt "hello world"
  %(prog)s --mode full --prompt "create a webpage with bikes"
  %(prog)s --mode mcp --server-url http://localhost:3000
  %(prog)s --interactive  # Use hybrid mode interactively
  %(prog)s --help-modes   # Show detailed mode information
        """,
    )

    # Core options
    parser.add_argument(
        "--mode",
        choices=["simple", "full", "mcp", "hybrid"],
        default="hybrid",
        help="Execution mode (default: hybrid)",
    )

    parser.add_argument("--prompt", type=str, help="Input prompt to process")

    parser.add_argument(
        "--interactive", action="store_true", help="Run in interactive mode"
    )

    # Configuration
    parser.add_argument("--config", type=str, help="Path to configuration file")

    parser.add_argument("--workspace", type=str, help="Workspace directory")

    # LLM options
    parser.add_argument(
        "--api-type",
        choices=["local", "ollama", "openai"],
        help="Override LLM API type",
    )

    # Agent options
    parser.add_argument("--agent", type=str, help="Specific agent to use")

    parser.add_argument("--max-steps", type=int, help="Maximum execution steps")

    # MCP options
    parser.add_argument("--server-url", type=str, help="MCP server URL")

    parser.add_argument(
        "--no-wait", action="store_true", help="Exit after processing (non-interactive)"
    )

    # Other options
    parser.add_argument(
        "--version", action="version", version=f"ParManus AI v{__version__}"
    )

    parser.add_argument(
        "--help-modes",
        action="store_true",
        help="Show detailed information about modes",
    )

    parser.add_argument(
        "--test-system", action="store_true", help="Test system components"
    )

    return parser


def show_mode_help():
    """Show detailed mode information"""
    print(
        f"""
🚀 ParManus AI v{__version__} - Mode Information

🔧 SIMPLE MODE (--mode simple)
   • Lightweight operation with basic agents
   • Uses main_simple.py backend
   • Best for: Quick tasks, testing, limited resources
   • Dependencies: Minimal (ollama recommended)

🛠️ FULL MODE (--mode full)
   • Complete feature set with advanced tools
   • Uses main.py backend with full agent system
   • Best for: Complex tasks, computer control, browser automation
   • Dependencies: All ParManus components

🔗 MCP MODE (--mode mcp)
   • Model Context Protocol server mode
   • Uses run_mcp.py backend
   • Best for: External integrations, plugin development
   • Dependencies: MCP libraries

🎯 HYBRID MODE (--mode hybrid) [DEFAULT]
   • Auto-detects best available mode
   • Falls back gracefully if components missing
   • Best for: General use, uncertain environments
   • Dependencies: Adaptive

📚 Configuration Files:
   • config/config.toml - Main configuration
   • config/config_ollama.toml - Ollama-specific settings

🚀 Quick Start:
   python parmanus.py --prompt "hello world"
   python parmanus.py --mode simple --prompt "test simple mode"
   python parmanus.py --mode full --prompt "create a webpage"
    """
    )


async def test_system():
    """Test system components"""
    print(f"🧪 ParManus AI v{__version__} - System Test")
    print("=" * 50)

    # Test imports
    print("📦 Testing imports...")
    tests = {
        "Core Config": "from app.config import load_config",
        "LLM Factory": "from app.llm_factory import create_llm",
        "Memory System": "from app.memory import Memory",
        "Simple Backend": "import main_simple",
        "Full Backend": "import main",
        "MCP Backend": "import run_mcp",
        "Router System": "from app.agent.router import AgentRouter",
    }

    results = {}
    for name, import_test in tests.items():
        try:
            exec(import_test)
            results[name] = "✅ OK"
        except Exception as e:
            results[name] = f"❌ FAIL: {e}"

    for name, result in results.items():
        print(f"   {name}: {result}")

    # Test configuration
    print("\n⚙️ Testing configuration...")
    try:
        from app.config import load_config

        config = load_config()
        print(
            f"   ✅ Config loaded: {config.llm.backend if hasattr(config.llm, 'backend') else 'Unknown backend'}"
        )
    except Exception as e:
        print(f"   ❌ Config failed: {e}")

    # Test LLM connectivity
    print("\n🧠 Testing LLM connectivity...")
    try:
        from app.config import load_config
        from app.llm_factory import create_llm

        config = load_config()
        llm = create_llm(config.llm)
        print("   ✅ LLM created successfully")

        # Test simple query
        response = await llm.ask(
            [{"role": "user", "content": "Say 'System test successful'"}]
        )
        if "successful" in response.lower():
            print("   ✅ LLM responding correctly")
        else:
            print(f"   ⚠️ Unexpected response: {response[:100]}...")

    except Exception as e:
        print(f"   ❌ LLM test failed: {e}")

    print("\n🏁 System test complete!")


def parse_arguments():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(
        description=f"ParManus AI v{__version__} - Unified Entry Point",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
🎯 Mode Selection:
  simple   - Lightweight mode with basic functionality
  full     - Complete feature set with advanced tools
  mcp      - Model Context Protocol server mode
  hybrid   - Auto-detect best available mode (default)

📝 Examples:
  parmanus.py --mode simple --prompt "hello world"
  parmanus.py --mode full --prompt "create a webpage with bikes"
  parmanus.py --mode mcp --server-url http://localhost:3000
  parmanus.py --interactive  # Use hybrid mode interactively
  parmanus.py --help-modes   # Show detailed mode information
""",
    )

    # Mode selection
    parser.add_argument(
        "--mode",
        choices=["simple", "full", "mcp", "hybrid"],
        default="hybrid",
        help="Execution mode (default: hybrid)",
    )

    # Input options
    parser.add_argument("--prompt", help="Input prompt to process")
    parser.add_argument(
        "--interactive", action="store_true", help="Run in interactive mode"
    )

    # Configuration
    parser.add_argument("--config", help="Path to configuration file")
    parser.add_argument("--workspace", help="Workspace directory")

    # LLM options
    parser.add_argument(
        "--api-type",
        choices=["local", "ollama", "openai"],
        help="Override LLM API type",
    )

    # Agent options
    parser.add_argument("--agent", help="Specific agent to use")
    parser.add_argument("--max-steps", type=int, help="Maximum execution steps")

    # MCP options
    parser.add_argument("--server-url", help="MCP server URL")
    parser.add_argument(
        "--no-wait", action="store_true", help="Exit after processing (non-interactive)"
    )

    # Other options
    parser.add_argument(
        "--version", action="version", version=f"ParManus AI v{__version__}"
    )
    parser.add_argument(
        "--help-modes",
        action="store_true",
        help="Show detailed information about modes",
    )
    parser.add_argument(
        "--test-system", action="store_true", help="Test system components"
    )

    return parser.parse_args()


async def main():
    """Main entry point"""
    parser = create_parser()
    args = parser.parse_args()

    # Handle special commands
    if args.help_modes:
        show_mode_help()
        return

    if args.test_system:
        await test_system()
        return

    # Show startup banner
    print("=" * 60)
    print(f"🤖 ParManus AI v{__version__}")
    print("🎯 Phase 1: Foundation & Stability")
    print("=" * 60)

    # Create runner and execute
    runner = ParManusRunner()

    try:
        if args.mode == "simple":
            await runner.run_simple_mode(args)
        elif args.mode == "full":
            await runner.run_full_mode(args)
        elif args.mode == "mcp":
            await runner.run_mcp_mode(args)
        elif args.mode == "hybrid":
            await runner.run_hybrid_mode(args)
        else:
            print(f"❌ Unknown mode: {args.mode}")
            return

    except KeyboardInterrupt:
        print("\n🛑 Interrupted by user")
    except Exception as e:
        print(f"💥 Unexpected error: {e}")
        print(f"🔍 Traceback: {traceback.format_exc()}")
        print(f"💡 Try: python parmanus.py --test-system")


if __name__ == "__main__":
    # Handle Windows async issues
    if sys.platform.startswith("win"):
        asyncio.set_event_loop_policy(asyncio.WindowsProactorEventLoopPolicy())

    asyncio.run(main())
