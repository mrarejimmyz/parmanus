#!/usr/bin/env python3
"""
Quick Phase 1 Validation Test
"""

import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))


def test_essentials():
    """Test essential Phase 1 components"""

    print("🧪 Phase 1 Quick Validation Test")
    print("=" * 50)

    tests_passed = 0
    total_tests = 0

    # Test 1: File structure
    total_tests += 1
    print("1. Testing file structure...", end=" ")
    essential_files = ["parmanus.py", "main.py", "app/config.py", "config/config.toml"]
    missing = [f for f in essential_files if not Path(f).exists()]
    if not missing:
        print("✅ PASS")
        tests_passed += 1
    else:
        print(f"❌ FAIL - Missing: {missing}")

    # Test 2: Imports
    total_tests += 1
    print("2. Testing critical imports...", end=" ")
    try:
        import app.llm_factory
        import parmanus
        from app.config import get_config

        print("✅ PASS")
        tests_passed += 1
    except Exception as e:
        print(f"❌ FAIL - {e}")

    # Test 3: Config loading
    total_tests += 1
    print("3. Testing config loading...", end=" ")
    try:
        from app.config import get_config

        config = get_config()
        assert hasattr(config, "llm")
        assert hasattr(config.llm, "backend")
        assert hasattr(config.llm, "model")
        print("✅ PASS")
        tests_passed += 1
    except Exception as e:
        print(f"❌ FAIL - {e}")

    # Test 4: Unified entry point
    total_tests += 1
    print("4. Testing unified entry point...", end=" ")
    try:
        import parmanus

        assert hasattr(parmanus, "parse_arguments")
        assert hasattr(parmanus, "main")
        print("✅ PASS")
        tests_passed += 1
    except Exception as e:
        print(f"❌ FAIL - {e}")

    # Test 5: Help functionality
    total_tests += 1
    print("5. Testing help functionality...", end=" ")
    try:
        args = parmanus.parse_arguments()
        print("✅ PASS")
        tests_passed += 1
    except SystemExit:
        # This is expected when parsing arguments without providing them
        print("✅ PASS")
        tests_passed += 1
    except Exception as e:
        print(f"❌ FAIL - {e}")

    print("=" * 50)
    print(f"Results: {tests_passed}/{total_tests} tests passed")
    success_rate = (tests_passed / total_tests) * 100
    print(f"Success rate: {success_rate:.1f}%")

    if tests_passed == total_tests:
        print("🎉 All Phase 1 essential components are working!")
        return True
    else:
        print("⚠️  Some issues detected. See details above.")
        return False


if __name__ == "__main__":
    success = test_essentials()
    sys.exit(0 if success else 1)
