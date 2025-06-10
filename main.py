"""
Hybrid ParManus AI Agent - Supports both Local GGUF models and Ollama
Optimized main entry point handling all functionality.
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

# Conditional imports based on availability
try:
    import httpx
    HTTPX_AVAILABLE = True
except ImportError:
    HTTPX_AVAILABLE = False

try:
    from openai import AsyncOpenAI
    OPENAI_AVAILABLE = True
except ImportError:
    OPENAI_AVAILABLE = False

try:
    from llama_cpp import Llama
    LLAMA_CPP_AVAILABLE = True
except ImportError:
    LLAMA_CPP_AVAILABLE = False

try:
    from loguru import logger
except ImportError:
    import logging
    logger = logging.getLogger(__name__)
    logging.basicConfig(level=logging.INFO)


class Config(BaseModel):
    """Unified configuration model supporting both local and Ollama."""
    
    # LLM Configuration
    api_type: str = Field(default="local", description="API type: local, ollama")
    model: str = Field(default="Llama-3.2-11B-Vision-Instruct", description="Model name")
    model_path: str = Field(default="models/Llama-3.2-11B-Vision-Instruct.Q4_K_M.gguf", description="Path to GGUF model")
    max_tokens: int = Field(default=2048, description="Maximum tokens")
    temperature: float = Field(default=0.0, description="Temperature")
    n_gpu_layers: int = Field(default=-1, description="GPU layers (-1 = all)")
    gpu_memory_limit: int = Field(default=7000, description="GPU memory limit MB")
    
    # Ollama Configuration (fallback)
    base_url: str = Field(default="http://localhost:11434/v1", description="Ollama API endpoint")
    api_key: str = Field(default="ollama", description="Ollama API key")
    
    # Vision Configuration
    vision_enabled: bool = Field(default=True, description="Enable vision")
    vision_model: str = Field(default="llava-v1.6-mistral-7b", description="Vision model")
    vision_model_path: str = Field(default="models/llava-1.6-mistral-7b-gguf/ggml-model-q4_k.gguf", description="Vision model path")
    vision_clip_path: str = Field(default="models/llava-1.6-mistral-7b-gguf/mmproj-model-f16.gguf", description="CLIP model path")
    
    # Other configurations
    voice_enabled: bool = Field(default=False, description="Enable voice")
    save_session: bool = Field(default=False, description="Save sessions")
    recover_last_session: bool = Field(default=False, description="Recover last session")
    headless: bool = Field(default=False, description="Headless browser")


class TokenCounter:
    """Simple token counter."""
    
    def __init__(self):
        self.prompt_tokens = 0
        self.completion_tokens = 0
        self.total_tokens = 0
    
    def update(self, prompt_tokens: int, completion_tokens: int):
        self.prompt_tokens += prompt_tokens
        self.completion_tokens += completion_tokens
        self.total_tokens = self.prompt_tokens + self.completion_tokens


class LocalLLM:
    """Local GGUF model implementation using llama-cpp-python."""
    
    def __init__(self, config: Config):
        if not LLAMA_CPP_AVAILABLE:
            raise ImportError("llama-cpp-python not available. Install with: pip install llama-cpp-python")
        
        self.config = config
        self.token_counter = TokenCounter()
        
        # Load main model
        if Path(config.model_path).exists():
            logger.info(f"Loading local model: {config.model_path}")
            self.model = Llama(
                model_path=config.model_path,
                n_gpu_layers=config.n_gpu_layers,
                n_ctx=4096,
                verbose=False
            )
        else:
            raise FileNotFoundError(f"Model not found: {config.model_path}")
        
        # Load vision model if enabled
        self.vision_model = None
        if config.vision_enabled and Path(config.vision_model_path).exists():
            logger.info(f"Loading vision model: {config.vision_model_path}")
            try:
                self.vision_model = Llama(
                    model_path=config.vision_model_path,
                    clip_model_path=config.vision_clip_path,
                    n_gpu_layers=config.n_gpu_layers,
                    n_ctx=2048,
                    verbose=False
                )
            except Exception as e:
                logger.warning(f"Failed to load vision model: {e}")
    
    def _format_prompt(self, messages: List[Dict[str, Any]]) -> str:
        """Format messages for llama prompt."""
        prompt = ""
        for msg in messages:
            role = msg.get('role', 'user')
            content = msg.get('content', '')
            if role == 'system':
                prompt += f"<|system|>\n{content}\n"
            elif role == 'user':
                prompt += f"<|user|>\n{content}\n"
            elif role == 'assistant':
                prompt += f"<|assistant|>\n{content}\n"
        prompt += "<|assistant|>\n"
        return prompt
    
    async def ask(self, messages: Union[str, List[Dict[str, Any]]], **kwargs) -> str:
        """Ask the local model."""
        try:
            if isinstance(messages, str):
                messages = [{"role": "user", "content": messages}]
            
            prompt = self._format_prompt(messages)
            
            # Run in thread to avoid blocking
            loop = asyncio.get_event_loop()
            response = await loop.run_in_executor(
                None,
                lambda: self.model(
                    prompt,
                    max_tokens=self.config.max_tokens,
                    temperature=self.config.temperature,
                    stop=["<|user|>", "<|system|>"],
                    **kwargs
                )
            )
            
            content = response['choices'][0]['text'].strip()
            
            # Update token counter
            usage = response.get('usage', {})
            self.token_counter.update(
                usage.get('prompt_tokens', 0),
                usage.get('completion_tokens', 0)
            )
            
            return content
            
        except Exception as e:
            logger.error(f"Error in local LLM: {e}")
            raise
    
    async def ask_vision(self, messages: Union[str, List[Dict[str, Any]]], images: List[str] = None, **kwargs) -> str:
        """Ask the vision model."""
        if not self.vision_model:
            raise ValueError("Vision model not available")
        
        try:
            if isinstance(messages, str):
                prompt = messages
            else:
                prompt = self._format_prompt(messages)
            
            # Run in thread to avoid blocking
            loop = asyncio.get_event_loop()
            response = await loop.run_in_executor(
                None,
                lambda: self.vision_model(
                    prompt,
                    max_tokens=self.config.max_tokens,
                    temperature=self.config.temperature,
                    **kwargs
                )
            )
            
            content = response['choices'][0]['text'].strip()
            
            # Update token counter
            usage = response.get('usage', {})
            self.token_counter.update(
                usage.get('prompt_tokens', 0),
                usage.get('completion_tokens', 0)
            )
            
            return content
            
        except Exception as e:
            logger.error(f"Error in vision model: {e}")
            raise


class OllamaLLM:
    """Ollama API implementation."""
    
    def __init__(self, config: Config):
        if not OPENAI_AVAILABLE:
            raise ImportError("openai package not available. Install with: pip install openai")
        
        self.config = config
        self.token_counter = TokenCounter()
        
        self.client = AsyncOpenAI(
            base_url=config.base_url,
            api_key=config.api_key,
        )
        
        logger.info(f"Initialized Ollama LLM: {config.base_url}")
    
    def _format_messages(self, messages: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Format messages for OpenAI API."""
        formatted = []
        for msg in messages:
            if isinstance(msg, str):
                formatted.append({"role": "user", "content": msg})
            elif isinstance(msg, dict):
                formatted.append(msg)
            else:
                formatted.append({"role": "user", "content": str(msg)})
        return formatted
    
    async def ask(self, messages: Union[str, List[Dict[str, Any]]], **kwargs) -> str:
        """Ask Ollama."""
        try:
            if isinstance(messages, str):
                messages = [{"role": "user", "content": messages}]
            
            formatted_messages = self._format_messages(messages)
            
            response = await self.client.chat.completions.create(
                model=self.config.model,
                messages=formatted_messages,
                max_tokens=self.config.max_tokens,
                temperature=self.config.temperature,
                **kwargs
            )
            
            content = response.choices[0].message.content
            
            if hasattr(response, 'usage') and response.usage:
                self.token_counter.update(
                    response.usage.prompt_tokens,
                    response.usage.completion_tokens
                )
            
            return content
            
        except Exception as e:
            logger.error(f"Error in Ollama: {e}")
            raise
    
    async def ask_vision(self, messages: Union[str, List[Dict[str, Any]]], images: List[str] = None, **kwargs) -> str:
        """Ask Ollama vision model."""
        return await self.ask(messages, **kwargs)  # Ollama handles vision in unified API


class SimpleAgent:
    """Simplified agent implementation."""
    
    def __init__(self, name: str, llm: Union[LocalLLM, OllamaLLM], config: Config):
        self.name = name
        self.llm = llm
        self.config = config
    
    async def run(self, prompt: str) -> str:
        """Run the agent."""
        try:
            system_prompt = self._get_system_prompt()
            messages = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": prompt}
            ]
            
            response = await self.llm.ask(messages)
            return response
            
        except Exception as e:
            logger.error(f"Error in agent {self.name}: {e}")
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


def route_agent(prompt: str) -> str:
    """Simple agent routing."""
    prompt_lower = prompt.lower()
    
    if any(word in prompt_lower for word in ["code", "program", "script", "debug", "function"]):
        return "code"
    elif any(word in prompt_lower for word in ["browse", "web", "scrape", "website", "url"]):
        return "browser"
    elif any(word in prompt_lower for word in ["file", "save", "read", "write", "data"]):
        return "file"
    elif any(word in prompt_lower for word in ["plan", "schedule", "task", "organize", "steps"]):
        return "planner"
    else:
        return "manus"


async def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description="ParManus AI Agent - Hybrid Local/Ollama")
    parser.add_argument("--config", type=str, help="Path to configuration file")
    parser.add_argument("--prompt", type=str, help="Input prompt")
    parser.add_argument("--no-wait", action="store_true", help="Exit immediately if no prompt")
    parser.add_argument("--agent", type=str, help="Specify agent (manus, code, browser, file, planner)")
    parser.add_argument("--api-type", type=str, choices=["local", "ollama"], help="Override API type")
    args = parser.parse_args()
    
    try:
        # Load configuration
        config = load_config(args.config)
        
        # Override API type if specified
        if args.api_type:
            config.api_type = args.api_type
        
        # Initialize LLM based on API type
        if config.api_type == "local":
            if not LLAMA_CPP_AVAILABLE:
                logger.error("llama-cpp-python not available. Install with: pip install llama-cpp-python")
                logger.info("Falling back to Ollama if available...")
                config.api_type = "ollama"
            else:
                try:
                    llm = LocalLLM(config)
                except Exception as e:
                    logger.error(f"Failed to initialize local LLM: {e}")
                    logger.info("Falling back to Ollama...")
                    config.api_type = "ollama"
        
        if config.api_type == "ollama":
            if not OPENAI_AVAILABLE:
                logger.error("openai package not available. Install with: pip install openai")
                sys.exit(1)
            llm = OllamaLLM(config)
        
        # Initialize memory
        memory = Memory(config)
        
        logger.info(f"ParManus AI Agent ready! Using {config.api_type} backend")
        logger.info(f"Model: {config.model}")
        
        # Track command line prompt processing
        processed_cmd_prompt = False
        
        # Main loop
        while True:
            prompt = None
            
            # Get prompt
            if not processed_cmd_prompt and args.prompt:
                prompt = args.prompt
                processed_cmd_prompt = True
                logger.info(f"Using command line prompt: {prompt[:100]}...")
            else:
                if sys.stdin.isatty():
                    try:
                        prompt = input("Enter your prompt (or 'quit' to exit): ")
                        if prompt.lower() in ["quit", "exit", "bye"]:
                            break
                    except EOFError:
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
                agent_name = args.agent if args.agent else route_agent(prompt)
                agent = SimpleAgent(agent_name, llm, config)
                
                logger.info(f"Processing with {agent.name} agent...")
                
                # Process request
                result = await agent.run(prompt)
                
                # Add result to memory
                memory.push("assistant", result)
                
                # Output result
                print(f"\n{agent.name.title()} Response:\n{result}\n")
                
                # Save session
                memory.save_session()
                
                # Exit if non-interactive
                if not sys.stdin.isatty() and args.no_wait and processed_cmd_prompt:
                    break
                    
            except Exception as e:
                error_msg = f"Error processing request: {e}"
                logger.error(error_msg, exc_info=True)
                print(f"Error: {error_msg}")
    
    except KeyboardInterrupt:
        logger.info("Operation interrupted")
    except Exception as e:
        logger.error(f"Critical error: {e}", exc_info=True)
        print(f"Critical error: {e}")
    finally:
        logger.info("ParManus AI Agent shutdown complete")


if __name__ == "__main__":
    asyncio.run(main())

