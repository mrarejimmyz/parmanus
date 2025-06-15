#!/usr/bin/env python3
"""
ParManus AI Demonstration Script
Showcases the unified entry point capabilities
"""

import subprocess
import sys
import time
from pathlib import Path


def run_command(cmd, description):
    """Run a command and show the result"""
    print(f"\n{'='*60}")
    print(f"🔧 {description}")
    print(f"Command: {cmd}")
    print("=" * 60)

    try:
        result = subprocess.run(
            cmd, shell=True, capture_output=True, text=True, timeout=30
        )
        if result.returncode == 0:
            print("✅ Success!")
            if result.stdout:
                print("Output:")
                print(result.stdout)
        else:
            print("❌ Failed!")
            if result.stderr:
                print("Error:")
                print(result.stderr)
    except subprocess.TimeoutExpired:
        print("⏰ Command timed out (30s)")
    except Exception as e:
        print(f"❌ Error running command: {e}")


def main():
    """Main demonstration"""
    print("🚀 ParManus AI v2.0.0-phase1 Demonstration")
    print("=" * 60)
    print("This script demonstrates the unified entry point capabilities")
    print("of the newly implemented Phase 1 foundation system.")

    # Ensure we're in the right directory
    if not Path("parmanus.py").exists():
        print("❌ Error: parmanus.py not found. Please run from project root.")
        sys.exit(1)

    # 1. Show help
    run_command("python parmanus.py --help", "Displaying unified entry point help")

    # 2. Show version
    run_command("python parmanus.py --version", "Showing version information")

    # 3. Show mode help
    run_command(
        "python parmanus.py --help-modes", "Displaying detailed mode information"
    )

    # 4. Test system components
    run_command("python parmanus.py --test-system", "Testing system components")

    # 5. Run quick validation
    run_command("python quick_test.py", "Running Phase 1 validation tests")

    print(f"\n{'='*60}")
    print("🎉 Demonstration Complete!")
    print("✅ ParManus AI Phase 1 foundation is working correctly")
    print(f"{'='*60}")
    print("\n📝 Available modes:")
    print("  • simple  - Lightweight mode with basic functionality")
    print("  • full    - Complete feature set with advanced tools")
    print("  • mcp     - Model Context Protocol server mode")
    print("  • hybrid  - Auto-detect best available mode (default)")
    print("\n🔗 Usage examples:")
    print('  python parmanus.py --mode simple --prompt "hello world"')
    print("  python parmanus.py --mode full --interactive")
    print("  python parmanus.py --help-modes  # For detailed information")


if __name__ == "__main__":
    main()
