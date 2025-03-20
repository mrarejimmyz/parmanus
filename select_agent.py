#!/usr/bin/env python3
"""
Agent Selector - Allows users to choose different agent modes
"""

import asyncio
import sys

from app.agent import CoTAgent
from app.agent.manus import Manus
from app.logger import logger


async def run_cot_agent():
    """Run Chain of Thought agent"""
    agent = CoTAgent()
    prompt = input("Enter your question: ")
    if not prompt.strip():
        logger.warning("Empty question provided")
        return

    logger.warning("Processing your question...")
    result = await agent.run(prompt)
    logger.info("Question processing completed")


async def run_react_agent():
    """Run ReAct agent (Manus)"""
    agent = Manus()
    prompt = input("Enter your request: ")
    if not prompt.strip():
        logger.warning("Empty request provided")
        return

    logger.warning("Processing your request...")
    await agent.run(prompt)
    logger.info("Request processing completed")


async def main():
    print("\nSelect the agent mode to use:")
    print("1. ReAct mode - Can use tools to execute tasks")
    print("2. Chain of Thought (CoT) mode - Shows detailed thinking process")

    choice = input("\nEnter option (1 or 2): ").strip()

    if choice == "1":
        await run_react_agent()
    elif choice == "2":
        await run_cot_agent()
    else:
        print("Invalid choice, please enter 1 or 2")


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        logger.warning("Operation interrupted")
        sys.exit(0)
