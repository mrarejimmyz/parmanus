#!/bin/bash

# Test script for validating optimizations
echo "Starting validation tests for ParManus optimizations..."

# Create test directory
mkdir -p /tmp/parmanus_test
cd /tmp/parmanus_test

# Test 1: Model Loading and Caching
echo "Test 1: Model Loading and Caching"
echo "================================="
echo "Running first model load (should take time)..."
time python3 -c "
from app.llm import LLM
import time
llm = LLM()
model = llm.text_model
print('Model loaded successfully')
"

echo -e "\nRunning second model load (should be much faster)..."
time python3 -c "
from app.llm import LLM
import time
llm = LLM()
model = llm.text_model
print('Model loaded successfully')
"

# Test 2: Tool Calling Implementation
echo -e "\nTest 2: Tool Calling Implementation"
echo "================================="
echo "Testing ask_tool method..."
python3 -c "
from app.llm import LLM
import asyncio

async def test_tool_calling():
    llm = LLM()
    tools = [
        {
            'name': 'test_tool',
            'description': 'A test tool',
            'parameters': {
                'type': 'object',
                'properties': {
                    'param1': {'type': 'string'}
                }
            }
        }
    ]
    
    result = await llm.ask_tool(
        messages=[{'role': 'user', 'content': 'Use the test tool with param1=test'}],
        tools=tools,
        tool_choice='auto'
    )
    
    print('Tool call result:', result)
    if 'tool_calls' in result:
        print('Tool calls found:', len(result['tool_calls']))
    else:
        print('No tool calls found')

asyncio.run(test_tool_calling())
"

# Test 3: Timeout Handling
echo -e "\nTest 3: Timeout Handling"
echo "================================="
echo "Testing timeout handling (should return partial results)..."
python3 -c "
from app.llm import LLM
import asyncio

async def test_timeout():
    llm = LLM()
    try:
        # Use very short timeout to trigger timeout handling
        result = await llm.ask(
            messages=[{'role': 'user', 'content': 'Write a very long essay about artificial intelligence'}],
            timeout=1
        )
        print('Result received with timeout handling:', result[:50], '...')
    except Exception as e:
        print('Error:', str(e))

asyncio.run(test_timeout())
"

# Test 4: Interrupt Handling
echo -e "\nTest 4: Interrupt Handling"
echo "================================="
echo "Testing interrupt handling (will send SIGINT after 2 seconds)..."
python3 -c "
import signal
import sys
import time
from app.cleanup import signal_handler

# Register our signal handler
signal.signal(signal.SIGINT, signal_handler)

print('Starting long operation...')
try:
    # Simulate long operation
    for i in range(10):
        print(f'Processing step {i+1}/10...')
        time.sleep(1)
        if i == 1:  # After 2 seconds
            print('Sending interrupt signal to self...')
            import os
            os.kill(os.getpid(), signal.SIGINT)
except KeyboardInterrupt:
    print('KeyboardInterrupt caught in main code')
except Exception as e:
    print(f'Other exception: {e}')

print('This line should not be reached if interrupt handling works correctly')
" || echo "Interrupt handler worked correctly"

echo -e "\nAll validation tests completed."
