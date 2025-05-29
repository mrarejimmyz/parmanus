import json
import threading
import tomllib
from pathlib import Path
from typing import Dict, List, Optional
from pydantic import BaseModel, Field

def get_project_root() -> Path:
    """Get the project root directory"""
    return Path(__file__).resolve().parent.parent

PROJECT_ROOT = get_project_root()
WORKSPACE_ROOT = PROJECT_ROOT / "workspace"

class LLMSettings(BaseModel):
    model: str = Field(..., description="Model name")
    model_path: str = Field("/models/llama-jb.gguf", description="Path to the model file")
    max_tokens: int = Field(4096, description="Maximum number of tokens per request")
    max_input_tokens: Optional[int] = Field(
        None,
        description="Maximum input tokens to use across all requests (None for unlimited)",
    )
    temperature: float = Field(0.0, description="Sampling temperature")

    # Optional vision model settings
    vision: Optional["LLMSettings"] = Field(None, description="Vision model settings")

class ProxySettings(BaseModel):
    server: str = Field(None, description="Proxy server address")
    username: Optional[str] = Field(None, description="Proxy username")
    password: Optional[str] = Field(None, description="Proxy password")

class SearchSettings(BaseModel):
    engine: str = Field(default="Google", description="Search engine the llm to use")
    fallback_engines: List[str] = Field(
        default_factory=lambda: ["DuckDuckGo", "Baidu", "Bing"],
        description="Fallback search engines to try if the primary engine fails",
    )
    retry_delay: int = Field(
        default=60,
        description="Seconds to wait before retrying all engines again after they all fail",
    )
    max_retries: int = Field(
        default=3,
        description="Maximum number of times to retry all engines when all fail",
    )
    lang: str = Field(default="en", description="Language code for search results")
    country: str = Field(default="us", description="Country code for search results")

class BrowserSettings(BaseModel):
    headless: bool = Field(default=False, description="Whether to run browser in headless mode")
    disable_security: bool = Field(
        default=True, description="Disable browser security features"
    )
    extra_chromium_args: List[str] = Field(
        default_factory=list, description="Extra arguments to pass to the browser"
    )
    chrome_instance_path: Optional[str] = Field(
        None, description="Path to a Chrome instance to use"
    )
    wss_url: Optional[str] = Field(
        None, description="Connect to a browser instance via WebSocket"
    )
    cdp_url: Optional[str] = Field(
        None, description="Connect to a browser instance via CDP"
    )
    proxy: Optional[ProxySettings] = Field(None, description="Proxy settings")

class SandboxSettings(BaseModel):
    use_sandbox: bool = Field(default=False, description="Whether to use sandbox")
    image: str = Field(default="python:3.12-slim", description="Docker image to use")
    work_dir: str = Field(default="/workspace", description="Working directory in container")
    memory_limit: str = Field(default="1g", description="Memory limit for container")
    cpu_limit: float = Field(default=2.0, description="CPU limit for container")
    timeout: int = Field(default=300, description="Timeout in seconds")
    network_enabled: bool = Field(default=True, description="Whether to enable network")

class MCPServerConfig(BaseModel):
    type: str = Field(default="sse", description="Server type (sse or stdio)")
    url: Optional[str] = Field(None, description="Server URL for SSE connections")
    command: Optional[str] = Field(None, description="Command for stdio connections")
    args: List[str] = Field(default_factory=list, description="Arguments for stdio command")

class MCPConfig(BaseModel):
    server_reference: str = Field(
        default="app.mcp.server", description="MCP server module reference"
    )
    servers: Dict[str, MCPServerConfig] = Field(
        default_factory=dict, description="MCP server configurations"
    )

class Config(BaseModel):
    llm: LLMSettings
    browser: Optional[BrowserSettings] = None
    search: Optional[SearchSettings] = None
    sandbox: Optional[SandboxSettings] = None
    mcp_config: MCPConfig = Field(default_factory=MCPConfig)
    workspace_root: str = Field(default=str(WORKSPACE_ROOT), description="Workspace root directory")

# Thread-local storage for config
_thread_local = threading.local()

def load_config(config_path: Optional[str] = None) -> Config:
    """
    Load configuration from a TOML file.
    Args:
        config_path: Path to the configuration file
    Returns:
        Config object
    """
    if config_path is None:
        config_path = PROJECT_ROOT / "config" / "config.toml"

    try:
        with open(config_path, "rb") as f:
            config_dict = tomllib.load(f)

        # Convert llm section to LLMSettings if it's a dict
        if "llm" in config_dict and isinstance(config_dict["llm"], dict):
            llm_dict = config_dict["llm"]

            # Handle vision settings if present
            if "vision" in llm_dict and isinstance(llm_dict["vision"], dict):
                vision_settings = LLMSettings(**llm_dict["vision"])
                llm_dict["vision"] = vision_settings

            # Create LLMSettings instance
            config_dict["llm"] = LLMSettings(**llm_dict)

        # Handle MCP configuration
        mcp_config = MCPConfig(server_reference="app.mcp.server")
        if "mcp" in config_dict:
            if isinstance(config_dict["mcp"], dict):
                if "server_reference" in config_dict["mcp"]:
                    mcp_config.server_reference = config_dict["mcp"]["server_reference"]

                # Process servers if present
                if "servers" in config_dict["mcp"] and isinstance(config_dict["mcp"]["servers"], dict):
                    for server_id, server_data in config_dict["mcp"]["servers"].items():
                        mcp_config.servers[server_id] = MCPServerConfig(**server_data)

        config_dict["mcp_config"] = mcp_config

        # Ensure workspace_root is set
        if "workspace_root" not in config_dict:
            config_dict["workspace_root"] = str(WORKSPACE_ROOT)

        return Config(**config_dict)
    except Exception as e:
        # If config file doesn't exist or is invalid, create a default config
        print(f"Error loading config: {e}")
        print("Using default configuration")

        # Default configuration for local GGUF models
        vision_settings = LLMSettings(
            model="qwen-vl-7b",
            model_path="/models/qwen-vl-7b-awq.gguf",
            max_tokens=4096,
            temperature=0.0
        )

        llm_settings = LLMSettings(
            model="llama-jb",
            model_path="/models/llama-jb.gguf",
            max_tokens=4096,
            temperature=0.0,
            vision=vision_settings
        )

        default_config = Config(
            llm=llm_settings,
            workspace_root=str(WORKSPACE_ROOT),
            mcp_config=mcp_config
        )

        return default_config

def get_config() -> Config:
    """
    Get the configuration for the current thread.
    Returns:
        Config object
    """
    if not hasattr(_thread_local, "config"):
        _thread_local.config = load_config()
    return _thread_local.config

# Global config instance
config = get_config()
