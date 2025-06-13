import os
import sys
import time
import traceback
from typing import Any

from app.config import Config, load_config
from app.logger import logger
from app.memory import Memory
from app.agent_wrappers import create_agent
from app.agent_router import route_agent

# Conditional import for ParManus components
try:
    from app.llm_hybrid import create_llm_with_tools
    PARMANUS_AVAILABLE = True
except ImportError as e:
    logger.warning(f"ParManus components not fully available: {e}")
    PARMANUS_AVAILABLE = False


async def initialize_system(args) -> tuple[Config, Any, Any]:
    """Initialize configuration, LLM, and memory."""
    logger.info("Initializing ParManus AI Agent System...")
    config = load_config(args.config)

    if args.api_type:
        if args.api_type != "ollama":
            logger.warning(f"Only Ollama is supported. Ignoring --api-type {args.api_type}")
        config.api_type = "ollama"
    if args.workspace:
        config.workspace_root = args.workspace
    if args.max_steps:
        config.max_steps = args.max_steps

    os.makedirs(config.workspace_root, exist_ok=True)

    try:
        llm = create_llm_with_tools(config)
    except Exception as e:
        logger.error(f"Failed to initialize Ollama LLM: {e}")
        logger.error("Make sure Ollama is running: ollama serve")
        logger.error("And the model is available: ollama pull llama3.2-vision")
        sys.exit(1)

    memory = Memory() # Removed config as initial_prompt
    return config, llm, memory


def display_startup_info(config: Config, args, parmanus_available: bool):
    """Display startup information."""
    logger.info("🚀 ParManus AI Agent System Ready!")
    logger.info(f"🧠 Backend: Ollama (Hybrid)")
    logger.info(f"🛠️ Tools Model: llama3.2")
    logger.info(f"👁️ Vision Model: llama3.2-vision")
    logger.info(f"📁 Workspace: {config.workspace_root}")
    if parmanus_available and not args.simple:
        logger.info("🛠️ Full tool system + vision available")
    else:
        logger.info("⚡ Simple mode active")


async def process_prompt(prompt: str, args, llm, config: Config, memory: Memory, parmanus_available: bool):
    """Process a single user prompt."""
    if not prompt or not prompt.strip():
        return

    memory.push("user", prompt)

    try:
        agent_name = args.agent if args.agent else route_agent(prompt, parmanus_available and not args.simple)
        agent = create_agent(agent_name, llm, config)

        logger.info(f"🎯 Using {agent_name} agent...")

        start_time = time.time()
        result = await agent.run(prompt)
        end_time = time.time()

        memory.push("assistant", result)

        logger.info(f"✅ Task completed in {end_time - start_time:.2f} seconds.")
        print(f"\n🤖 Agent Response:\n{result}")

    except Exception as e:
        logger.error(f"Error processing prompt: {e}")
        logger.error(traceback.format_exc())
        memory.push("error", f"Error processing prompt: {e}")


