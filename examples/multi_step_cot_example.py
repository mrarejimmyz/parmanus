#!/usr/bin/env python3
"""
Multi-step CoT Agent Example - Demonstrates how to use Chain of Thought mode in multi-turn conversations
"""

import asyncio
import sys
from pathlib import Path

# Add project root directory to path
sys.path.append(str(Path(__file__).resolve().parent.parent))

from app.agent import CoTAgent
from app.logger import logger
from app.schema import Message


async def main():
    # Create CoT agent
    agent = CoTAgent(max_steps=3)  # Set maximum steps to 3

    # Initial question
    initial_question = "As artificial intelligence technology develops, what ethical challenges might we face?"

    logger.info(f"Initial question: {initial_question}")

    # Add initial question to agent's memory
    agent.memory.add_message(Message.user_message(initial_question))

    # Step 1: Get initial thoughts
    logger.info("Step 1: Initial thoughts")
    response1 = await agent.step()
    print(f"\nResponse:\n{response1}\n")

    # Step 2: Ask follow-up question
    follow_up_question = "Among the ethical challenges you mentioned, which one do you think is most urgent to address? Why?"
    logger.info(f"Follow-up question: {follow_up_question}")

    agent.memory.add_message(Message.user_message(follow_up_question))
    response2 = await agent.step()
    print(f"\nResponse:\n{response2}\n")

    # Step 3: Ask final follow-up
    final_question = "Can you suggest some specific solutions for addressing this most urgent ethical challenge?"
    logger.info(f"Final question: {final_question}")

    agent.memory.add_message(Message.user_message(final_question))
    response3 = await agent.step()
    print(f"\nResponse:\n{response3}\n")

    logger.info("Multi-step CoT conversation complete!")


if __name__ == "__main__":
    asyncio.run(main())
