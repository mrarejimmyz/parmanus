"""
Complete ParManus AI Agent System with Full Tool Integration
Optimized for local GGUF models while maintaining all functionality.
"""

import argparse
import asyncio
import json
import os
import sys
import time
import traceback
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

# Core imports
from pydantic import BaseModel, Field
import tomllib

# Conditional imports
try:
    from loguru import logger
except ImportError:
    import logging
    logger = logging.getLogger(__name__)
    logging.basicConfig(level=logging.INFO)

# Import ParManus components
try:
    from app.llm_hybrid import create_llm_with_tools
    from app.agent.manus import Manus
    from app.agent.code import CodeAgent
    from app.agent.browser import BrowserAgent
    from app.config import Config as ParManusConfig
    from app.memory import Memory
    from app.schema import Message, AgentState
    PARMANUS_AVAILABLE = True
except ImportError as e:
    logger.warning(f"ParManus components not fully available: {e}")
    PARMANUS_AVAILABLE = False


class Config(BaseModel):
    """Unified configuration model - Ollama only."""
    
    # LLM Configuration (Ollama only)
    api_type: str = Field(default="ollama", description="API type: ollama (only option)")
    model: str = Field(default="llama3.2-vision", description="Ollama model name")
    max_tokens: int = Field(default=2048, description="Maximum tokens")
    temperature: float = Field(default=0.0, description="Temperature")
    
    # Ollama Configuration
    base_url: str = Field(default="http://localhost:11434/v1", description="Ollama API endpoint")
    api_key: str = Field(default="ollama", description="Ollama API key")
    
    # Vision Configuration (same model)
    vision_enabled: bool = Field(default=True, description="Enable vision")
    
    # Workspace and paths
    workspace_root: str = Field(default="./workspace", description="Workspace directory")
    
    # Agent settings
    max_steps: int = Field(default=20, description="Maximum agent steps")
    max_observe: int = Field(default=10000, description="Maximum observation length")
    
    # Memory settings
    save_session: bool = Field(default=False, description="Save sessions")
    recover_last_session: bool = Field(default=False, description="Recover last session")
    
    # Browser settings
    headless: bool = Field(default=False, description="Headless browser")
    disable_security: bool = Field(default=True, description="Disable browser security")
    
    # Voice settings
    voice_enabled: bool = Field(default=False, description="Enable voice")
    speak: bool = Field(default=False, description="Enable TTS")
    listen: bool = Field(default=False, description="Enable STT")


class SimpleAgent:
    """Simplified agent for basic functionality when ParManus is not available."""
    
    def __init__(self, name: str, llm, config: Config):
        self.name = name
        self.llm = llm
        self.config = config
        self.messages = []
    
    async def run(self, prompt: str) -> str:
        """Run the agent with a simple prompt."""
        try:
            system_prompt = self._get_system_prompt()
            messages = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": prompt}
            ]
            
            response = await self.llm.ask(messages)
            return response
            
        except Exception as e:
            logger.error(f"Error in simple agent {self.name}: {e}")
            return f"Error: {e}"
    
    def _get_system_prompt(self) -> str:
        """Get system prompt based on agent type."""
        prompts = {
            "manus": "You are Manus, a helpful AI assistant. Provide clear, accurate, and helpful responses.",
            "code": "You are a coding assistant. Help with programming tasks, debugging, and code explanations.",
            "browser": "You are a browser automation assistant. Help with web scraping and browser tasks.",
            "file": "You are a file management assistant. Help with file operations and data processing.",
            "planner": "You are a planning assistant. Help break down tasks and create actionable plans."
        }
        return prompts.get(self.name, prompts["manus"])


class ParManusAgentWrapper:
    """Wrapper for full ParManus agents."""
    
    def __init__(self, agent_class, llm, config: Config):
        self.agent_class = agent_class
        self.llm = llm
        self.config = config
        self.agent = None
    
    async def run(self, prompt: str) -> str:
        """Run the ParManus agent."""
        try:
            if not self.agent:
                # Create ParManus config
                parmanus_config = self._create_parmanus_config()
                
                # Initialize agent
                if self.agent_class == Manus:
                    self.agent = await Manus.create()
                else:
                    self.agent = self.agent_class()
                
                # Set LLM
                self.agent.llm = self.llm
            
            # Run agent
            result = await self.agent.run(prompt)
            return result
            
        except Exception as e:
            logger.error(f"Error in ParManus agent: {e}")
            return f"Error: {e}"
        finally:
            # Cleanup
            if self.agent and hasattr(self.agent, 'cleanup'):
                try:
                    await self.agent.cleanup()
                except Exception as e:
                    logger.warning(f"Error during agent cleanup: {e}")
    
    def _create_parmanus_config(self):
        """Create ParManus config from our config."""
        # This would map our config to ParManus config format
        # For now, return a basic config
        return type('Config', (), {
            'workspace_root': self.config.workspace_root,
            'max_steps': self.config.max_steps,
            'max_observe': self.config.max_observe,
        })()


class Memory:
    """Simple memory system."""
    
    def __init__(self, config: Config):
        self.config = config
        self.messages = []
        self.session_file = Path("session.json")
        
        if config.recover_last_session and self.session_file.exists():
            self.load_session()
    
    def push(self, role: str, content: str):
        """Add message to memory."""
        self.messages.append({
            "role": role,
            "content": content,
            "timestamp": time.time()
        })
    
    def save_session(self):
        """Save session to file."""
        if self.config.save_session:
            try:
                with open(self.session_file, 'w') as f:
                    json.dump(self.messages, f, indent=2)
                logger.info("Session saved")
            except Exception as e:
                logger.error(f"Error saving session: {e}")
    
    def load_session(self):
        """Load session from file."""
        try:
            with open(self.session_file, 'r') as f:
                self.messages = json.load(f)
            logger.info("Session loaded")
        except Exception as e:
            logger.error(f"Error loading session: {e}")


def load_config(config_path: Optional[str] = None) -> Config:
    """Load configuration from file or use defaults."""
    if config_path and Path(config_path).exists():
        try:
            with open(config_path, "rb") as f:
                config_dict = tomllib.load(f)
            
            # Flatten nested config
            flat_config = {}
            
            # LLM config
            if "llm" in config_dict:
                llm_config = config_dict["llm"]
                flat_config.update({
                    "api_type": llm_config.get("api_type", "local"),
                    "model": llm_config.get("model", "Llama-3.2-11B-Vision-Instruct"),
                    "model_path": llm_config.get("model_path", "models/Llama-3.2-11B-Vision-Instruct.Q4_K_M.gguf"),
                    "base_url": llm_config.get("base_url", "http://localhost:11434/v1"),
                    "api_key": llm_config.get("api_key", "ollama"),
                    "max_tokens": llm_config.get("max_tokens", 2048),
                    "temperature": llm_config.get("temperature", 0.0),
                    "n_gpu_layers": llm_config.get("n_gpu_layers", -1),
                    "gpu_memory_limit": llm_config.get("gpu_memory_limit", 7000),
                })
                
                # Vision config
                if "vision" in llm_config:
                    vision_config = llm_config["vision"]
                    flat_config.update({
                        "vision_enabled": vision_config.get("enabled", True),
                        "vision_model": vision_config.get("model", "llava-v1.6-mistral-7b"),
                        "vision_model_path": vision_config.get("model_path", "models/llava-1.6-mistral-7b-gguf/ggml-model-q4_k.gguf"),
                        "vision_clip_path": vision_config.get("clip_model_path", "models/llava-1.6-mistral-7b-gguf/mmproj-model-f16.gguf"),
                    })
            
            # Memory config
            if "memory" in config_dict:
                memory_config = config_dict["memory"]
                flat_config.update({
                    "save_session": memory_config.get("save_session", False),
                    "recover_last_session": memory_config.get("recover_last_session", False),
                })
            
            # Browser config
            if "browser" in config_dict:
                browser_config = config_dict["browser"]
                flat_config.update({
                    "headless": browser_config.get("headless", False),
                    "disable_security": browser_config.get("disable_security", True),
                })
            
            return Config(**flat_config)
            
        except Exception as e:
            logger.error(f"Error loading config: {e}")
            return Config()
    
    # Try default config locations
    default_paths = ["config/config.toml", "config.toml"]
    for path in default_paths:
        if Path(path).exists():
            return load_config(path)
    
    logger.info("Using default configuration")
    return Config()


def route_agent(prompt: str, use_full_agents: bool = True) -> str:
    """Route to appropriate agent based on prompt."""
    prompt_lower = prompt.lower()
    
    if any(word in prompt_lower for word in ["code", "program", "script", "debug", "function", "python", "javascript"]):
        return "code"
    elif any(word in prompt_lower for word in ["browse", "web", "scrape", "website", "url", "browser"]):
        return "browser"
    elif any(word in prompt_lower for word in ["file", "save", "read", "write", "data", "edit"]):
        return "file"
    elif any(word in prompt_lower for word in ["plan", "schedule", "task", "organize", "steps"]):
        return "planner"
    else:
        return "manus"


def create_agent(agent_name: str, llm, config: Config):
    """Create appropriate agent based on name and availability."""
    if PARMANUS_AVAILABLE:
        # Use full ParManus agents
        agent_map = {
            "manus": Manus,
            "code": CodeAgent if 'CodeAgent' in globals() else None,
            "browser": BrowserAgent if 'BrowserAgent' in globals() else None,
        }
        
        agent_class = agent_map.get(agent_name)
        if agent_class:
            return ParManusAgentWrapper(agent_class, llm, config)
    
    # Fallback to simple agent
    return SimpleAgent(agent_name, llm, config)


async def main():
    """Main entry point with full functionality."""
    parser = argparse.ArgumentParser(description="ParManus AI Agent - Complete System")
    parser.add_argument("--config", type=str, help="Path to configuration file")
    parser.add_argument("--prompt", type=str, help="Input prompt")
    parser.add_argument("--no-wait", action="store_true", help="Exit immediately if no prompt")
    parser.add_argument("--agent", type=str, help="Specify agent (manus, code, browser, file, planner)")
    parser.add_argument("--api-type", type=str, choices=["local", "ollama"], help="Override API type")
    parser.add_argument("--simple", action="store_true", help="Use simple agents only")
    parser.add_argument("--workspace", type=str, help="Workspace directory")
    parser.add_argument("--max-steps", type=int, help="Maximum agent steps")
    args = parser.parse_args()
    
    try:
        # Load configuration
        config = load_config(args.config)
        
        # Override settings from command line
        if args.api_type:
            if args.api_type != "ollama":
                logger.warning(f"Only Ollama is supported. Ignoring --api-type {args.api_type}")
            config.api_type = "ollama"
        if args.workspace:
            config.workspace_root = args.workspace
        if args.max_steps:
            config.max_steps = args.max_steps
        
        # Create workspace directory
        os.makedirs(config.workspace_root, exist_ok=True)
        
        # Initialize LLM (Ollama only)
        try:
            llm = create_llm_with_tools(config)
        except Exception as e:
            logger.error(f"Failed to initialize Ollama LLM: {e}")
            logger.error("Make sure Ollama is running: ollama serve")
            logger.error("And the model is available: ollama pull llama3.2-vision")
            sys.exit(1)
        
        # Initialize memory
        memory = Memory(config)
        
        # Display startup info
        logger.info("🚀 ParManus AI Agent System Ready!")
        logger.info(f"🧠 Backend: Ollama (Hybrid)")
        logger.info(f"🛠️ Tools Model: llama3.2")
        logger.info(f"👁️ Vision Model: llama3.2-vision")
        logger.info(f"📁 Workspace: {config.workspace_root}")
        if PARMANUS_AVAILABLE and not args.simple:
            logger.info("🛠️ Full tool system + vision available")
        else:
            logger.info("⚡ Simple mode active")
        
        # Track command line prompt processing
        processed_cmd_prompt = False
        
        # Main loop
        while True:
            prompt = None
            
            # Get prompt
            if not processed_cmd_prompt and args.prompt:
                prompt = args.prompt
                processed_cmd_prompt = True
                logger.info(f"📝 Processing: {prompt[:100]}...")
            else:
                if sys.stdin.isatty():
                    try:
                        prompt = input("\n💬 Enter your prompt (or 'quit' to exit): ")
                        if prompt.lower() in ["quit", "exit", "bye", "q"]:
                            break
                    except (EOFError, KeyboardInterrupt):
                        if args.no_wait:
                            break
                        continue
                else:
                    if args.no_wait and processed_cmd_prompt:
                        break
                    time.sleep(5)
                    continue
            
            if not prompt or not prompt.strip():
                continue
            
            # Add to memory
            memory.push("user", prompt)
            
            try:
                # Route to agent
                agent_name = args.agent if args.agent else route_agent(prompt, not args.simple)
                
                # Create agent
                agent = create_agent(agent_name, llm, config)
                
                logger.info(f"🎯 Using {agent_name} agent...")
                
                # Process request
                start_time = time.time()
                result = await agent.run(prompt)
                end_time = time.time()
                
                # Add result to memory
                memory.push("assistant", result)
                
                # Display result
                print(f"\n🤖 {agent_name.title()} Response:")
                print("=" * 50)
                print(result)
                print("=" * 50)
                print(f"⏱️ Completed in {end_time - start_time:.2f} seconds")
                
                # Show token usage if available
                if hasattr(llm, 'get_token_count'):
                    tokens = llm.get_token_count()
                    print(f"🔢 Tokens: {tokens['total_tokens']} (prompt: {tokens['prompt_tokens']}, completion: {tokens['completion_tokens']})")
                
                # Save session
                memory.save_session()
                
                # Exit if non-interactive
                if not sys.stdin.isatty() and args.no_wait and processed_cmd_prompt:
                    break
                    
            except Exception as e:
                error_msg = f"Error processing request: {e}"
                logger.error(error_msg, exc_info=True)
                print(f"\n❌ Error: {error_msg}")
                
                # Add error to memory
                memory.push("assistant", f"Error: {error_msg}")
    
    except KeyboardInterrupt:
        logger.info("\n👋 Operation interrupted by user")
    except Exception as e:
        logger.error(f"💥 Critical error: {e}", exc_info=True)
        print(f"\n💥 Critical error: {e}")
    finally:
        logger.info("🛑 ParManus AI Agent shutdown complete")


if __name__ == "__main__":
    asyncio.run(main())

