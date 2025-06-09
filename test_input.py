#!/usr/bin/env python3
"""
Simple test script to send input to the running ParManusAI application
"""
import subprocess
import sys
import time


def test_parmanus():
    print("Testing ParManusAI with 'build me a web page' request...")

    # Start the application
    process = subprocess.Popen(
        [sys.executable, "main.py"],
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        bufsize=0,
    )

    try:
        # Wait a moment for the application to start
        time.sleep(2)

        # Send the test input
        test_input = "build me a web page\n"
        print(f"Sending input: {test_input.strip()}")

        process.stdin.write(test_input)
        process.stdin.flush()

        # Wait for response (timeout after 30 seconds)
        try:
            output, error = process.communicate(timeout=30)
            print("=== OUTPUT ===")
            print(output)
            if error:
                print("=== ERROR ===")
                print(error)
        except subprocess.TimeoutExpired:
            print("Process timed out after 30 seconds")
            process.kill()
            output, error = process.communicate()
            print("=== OUTPUT ===")
            print(output)
            if error:
                print("=== ERROR ===")
                print(error)

    except Exception as e:
        print(f"Error during test: {e}")
    finally:
        if process.poll() is None:
            process.terminate()


if __name__ == "__main__":
    test_parmanus()
