import asyncio
import argparse
import os
import sys
import time

# Import the main patch module first to ensure all patches are applied
from app.main_patch import logger

# Then import the agent
from app.agent.manus import Manus

async def main():
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Manus AI Agent')
    parser.add_argument('--prompt', type=str, help='Input prompt for the agent')
    parser.add_argument('--no-wait', action='store_true', help='Exit immediately if no prompt is provided')
    args = parser.parse_args()
    
    # Create and initialize Manus agent
    agent = await Manus.create()
    try:
        # Continuous loop to wait for input
        while True:
            # Get prompt from command line args, environment variable, or interactive input
            prompt = None
            # First check command line argument
            if args.prompt:
                prompt = args.prompt
                # Clear the argument after using it once
                args.prompt = None
            # Then check environment variable
            elif os.environ.get('MANUS_PROMPT'):
                prompt = os.environ.get('MANUS_PROMPT')
                # Clear the environment variable after using it once
                os.environ.pop('MANUS_PROMPT', None)
            # Finally try interactive input
            else:
                if sys.stdin.isatty():
                    try:
                        prompt = input("Enter your prompt: ")
                    except EOFError:
                        logger.warning("EOF received when reading input.")
                        if args.no_wait:
                            break
                        time.sleep(1)  # Wait a bit before trying again
                        continue
                else:
                    # In non-interactive mode, wait for input to become available
                    logger.warning("Waiting for input in non-interactive mode. Use --prompt argument or MANUS_PROMPT environment variable.")
                    if args.no_wait:
                        break
                    # Wait for a while before checking again
                    time.sleep(5)
                    continue
            
            if not prompt or not prompt.strip():
                logger.warning("Empty prompt provided.")
                continue
            
            logger.warning("Processing your request...")
            await agent.run(prompt)
            logger.info("Request processing completed.")
            
            # If we're not in interactive mode and no more inputs are expected, break
            if not sys.stdin.isatty() and args.no_wait:
                break
    except KeyboardInterrupt:
        logger.warning("Operation interrupted.")
    finally:
        # Ensure agent resources are cleaned up before exiting
        await agent.cleanup()

if __name__ == "__main__":
    asyncio.run(main())
