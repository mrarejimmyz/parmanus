#!/usr/bin/env python3
"""
CoT Agent Example - Demonstrates how to use Chain of Thought mode for reasoning
"""

import asyncio
import sys
from pathlib import Path

# Add project root directory to path
sys.path.append(str(Path(__file__).resolve().parent.parent))

from app.agent import CoTAgent
from app.logger import logger


async def main():
    # Create CoT agent
    agent = CoTAgent()

    # Provide a problem that requires thinking
    question = "Zhang, Li, and Wang have a total of 85 apples. Zhang has twice as many apples as Li, and Wang has 5 more apples than Li. How many apples does each person have?"

    logger.info(f"Question: {question}")

    # Run the agent and get results
    result = await agent.run(question)

    logger.info(f"Execution complete!")


if __name__ == "__main__":
    asyncio.run(main())
