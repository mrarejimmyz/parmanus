import argparse
import asyncio
import os
import sys
import time

# Import agent router and voice modules
from app.agent.router import AgentRouter

# Import configuration
from app.config import get_config

# Import the main patch module first to ensure all patches are applied
from app.main_patch import logger

# Import memory system
from app.memory import Memory
from app.voice.stt import SpeechToText
from app.voice.tts import TextToSpeech
from app.gpu_manager import get_gpu_manager


async def main():
    # Parse command line arguments
    # Initialize GPU manager for optimization
    gpu_manager = get_gpu_manager()
    logger.info(f"GPU Manager initialized: CUDA={gpu_manager.cuda_available}")

    parser = argparse.ArgumentParser(
        description="ParManus AI Agent with Parmanus Integration"
    )
    parser.add_argument("--config", type=str, help="Path to configuration file")
    parser.add_argument("--prompt", type=str, help="Input prompt for the agent")
    parser.add_argument(
        "--no-wait",
        action="store_true",
        help="Exit immediately if no prompt is provided",
    )
    parser.add_argument(
        "--voice", action="store_true", help="Enable voice interaction mode"
    )
    parser.add_argument(
        "--agent",
        type=str,
        help="Specify agent to use (manus, code, browser, file, planner)",
    )
    args = parser.parse_args()

    # Load configuration
    config_path = args.config if args.config else None
    config = get_config(config_path)

    # Initialize voice modules if enabled
    tts = None
    stt = None

    if config.voice and (config.voice.speak or config.voice.listen or args.voice):
        logger.info("Initializing voice modules...")

        # Initialize TTS
        if config.voice.speak or args.voice:
            tts = TextToSpeech(enabled=True)
            logger.info("Text-to-speech enabled")

        # Initialize STT
        if config.voice.listen or args.voice:
            stt = SpeechToText(
                enabled=True,
                trigger_word=config.voice.agent_name if config.voice else "friday",
            )
            logger.info("Speech-to-text enabled")

    # Initialize memory system
    memory_config = config.memory if config.memory else None
    memory = Memory(
        recover_last_session=(
            memory_config.recover_last_session if memory_config else False
        ),
        memory_compression=memory_config.memory_compression if memory_config else False,
    )

    # Initialize agent router
    router = AgentRouter()

    # Welcome message
    welcome_msg = "ParManus AI Agent with Parmanus Integration ready!"
    logger.info(welcome_msg)
    if tts:
        await tts.speak_async(welcome_msg)

    try:
        # Continuous loop to wait for input
        while True:
            prompt = None

            # Get prompt from various sources
            if args.prompt:
                prompt = args.prompt
                args.prompt = None  # Clear after use
            elif os.environ.get("MANUS_PROMPT"):
                prompt = os.environ.get("MANUS_PROMPT")
                os.environ.pop("MANUS_PROMPT", None)  # Clear after use
            elif stt and stt.enabled:
                # Voice input mode
                logger.info("Listening for voice input...")
                if tts:
                    await tts.speak_async("I'm listening")
                prompt = await stt.listen_async(timeout=10.0)
                if prompt:
                    logger.info(f"Voice input received: {prompt}")
            else:
                # Text input mode
                if sys.stdin.isatty():
                    try:
                        prompt = input("Enter your prompt (or 'quit' to exit): ")
                        if prompt.lower() in ["quit", "exit", "bye"]:
                            break
                    except EOFError:
                        logger.warning("EOF received when reading input.")
                        if args.no_wait:
                            break
                        time.sleep(1)
                        continue
                else:
                    logger.warning("Waiting for input in non-interactive mode.")
                    if args.no_wait:
                        break
                    time.sleep(5)
                    continue

            if not prompt or not prompt.strip():
                logger.warning("Empty prompt provided.")
                continue

            # Add prompt to memory
            memory.push("user", prompt)

            # Route to appropriate agent
            if args.agent:
                # Use specified agent
                agent = await router._create_agent(args.agent)
            else:
                # Auto-route based on query
                agent = await router.route(prompt)

            logger.info(f"Processing request with {agent.name}...")
            if tts:
                await tts.speak_async("Processing your request")

            # Process the request
            try:
                result = await agent.run(prompt)

                # Add result to memory
                memory.push("assistant", result)

                # Output result
                logger.info("Request processing completed.")
                print(f"\n{agent.name} Response:\n{result}\n")

                # Speak result if TTS is enabled
                if tts:
                    # Extract a summary for TTS (first few sentences)
                    tts_text = _extract_summary_for_tts(result)
                    await tts.speak_async(tts_text)

            except Exception as e:
                error_msg = f"Error processing request: {e}"
                logger.error(error_msg)
                print(f"Error: {error_msg}")
                if tts:
                    await tts.speak_async(
                        "Sorry, I encountered an error processing your request"
                    )

            # Save session if configured
            if memory_config and memory_config.save_session:
                memory.save_session()

            # Clean up agent resources
            if hasattr(agent, "cleanup"):
                await agent.cleanup()

            # Exit if in non-interactive mode
            if not sys.stdin.isatty() and args.no_wait:
                break

    except KeyboardInterrupt:
        logger.warning("Operation interrupted.")
        if tts:
            await tts.speak_async("Goodbye!")
    finally:
        # Cleanup
        if stt and stt.is_listening:
            stt.stop_continuous_listening()

        # Save final session
        if memory_config and memory_config.save_session:
            memory.save_session()

        logger.info("ParManus AI Agent shutdown complete.")


def _extract_summary_for_tts(text: str, max_sentences: int = 2) -> str:
    """Extract a summary from text for TTS output.

    Args:
        text: Full text to summarize.
        max_sentences: Maximum number of sentences to include.

    Returns:
        Summary text suitable for TTS.
    """
    if not text:
        return "Task completed"

    # Split into sentences
    sentences = text.split(".")

    # Take first few sentences
    summary_sentences = []
    for sentence in sentences[:max_sentences]:
        sentence = sentence.strip()
        if sentence and len(sentence) > 10:  # Skip very short fragments
            summary_sentences.append(sentence)

    if summary_sentences:
        return ". ".join(summary_sentences) + "."
    else:
        return "Task completed"


if __name__ == "__main__":
    asyncio.run(main())
