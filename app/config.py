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


class VisionSettings(BaseModel):
    """Vision model configuration."""

    enabled: bool = Field(default=False, description="Enable vision capabilities")
    model: str = Field(default="llava-v1.6-mistral-7b", description="Vision model name")
    model_path: str = Field(
        default="models/llava-model.gguf", description="Path to vision model"
    )
    clip_model_path: str = Field(
        default="models/mmproj-model.gguf", description="Path to CLIP model"
    )
    max_tokens: int = Field(default=2048, description="Maximum tokens for vision model")
    temperature: float = Field(default=0.1, description="Temperature for vision model")
    n_gpu_layers: int = Field(
        default=0, description="Number of GPU layers for vision model"
    )


class LLMSettings(BaseModel):
    """LLM configuration."""

    model: str = Field(default="llama-jb", description="Main model name")
    model_path: str = Field(
        default="models/llama-jb.gguf", description="Path to main model"
    )
    max_tokens: int = Field(default=2048, description="Maximum tokens for generation")
    temperature: float = Field(default=0.0, description="Temperature for generation")
    n_gpu_layers: int = Field(
        default=0, description="Number of GPU layers for main model"
    )
    vision: VisionSettings = Field(
        default_factory=VisionSettings, description="Vision model settings"
    )


class ProxySettings(BaseModel):
    server: str = Field(None, description="Proxy server address")
    username: Optional[str] = Field(None, description="Proxy username")
    password: Optional[str] = Field(None, description="Proxy password")


class SearchSettings(BaseModel):
    engine: str = Field(default="Google", description="Search engine the llm to use")
    fallback_engines: List[str] = Field(default_factory=lambda: [])
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
    headless: bool = Field(
        default=False, description="Whether to run browser in headless mode"
    )
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
    work_dir: str = Field(
        default="/workspace", description="Working directory in container"
    )
    memory_limit: str = Field(default="1g", description="Memory limit for container")
    cpu_limit: float = Field(default=2.0, description="CPU limit for container")
    timeout: int = Field(default=300, description="Timeout in seconds")
    network_enabled: bool = Field(default=True, description="Whether to enable network")


class MCPServerConfig(BaseModel):
    type: str = Field(default="sse", description="Server type (sse or stdio)")
    url: Optional[str] = Field(None, description="Server URL for SSE connections")
    command: Optional[str] = Field(None, description="Command for stdio connections")
    args: List[str] = Field(
        default_factory=list, description="Arguments for stdio command"
    )


class MCPConfig(BaseModel):
    server_reference: str = Field(
        default="app.mcp.server", description="MCP server module reference"
    )
    servers: Dict[str, MCPServerConfig] = Field(
        default_factory=dict, description="MCP server configurations"
    )


class AgentRouterSettings(BaseModel):
    """Configuration for agent routing system."""

    enabled: bool = Field(default=True, description="Whether agent routing is enabled")
    default_agent: str = Field(default="manus", description="Default agent to use")


class MemorySettings(BaseModel):
    """Configuration for memory system."""

    save_session: bool = Field(default=False, description="Whether to save sessions")
    recover_last_session: bool = Field(
        default=False, description="Whether to recover last session"
    )
    memory_compression: bool = Field(
        default=False, description="Whether to compress memory"
    )


class VoiceSettings(BaseModel):
    """Configuration for voice interaction."""

    speak: bool = Field(default=False, description="Whether to enable text-to-speech")
    listen: bool = Field(default=False, description="Whether to enable speech-to-text")
    agent_name: str = Field(
        default="Friday", description="Agent name for voice interaction"
    )


class Config(BaseModel):
    llm: LLMSettings
    browser: Optional[BrowserSettings] = None
    search: Optional[SearchSettings] = None
    sandbox: Optional[SandboxSettings] = None
    mcp_config: MCPConfig = Field(default_factory=MCPConfig)
    workspace_root: str = Field(
        default=str(WORKSPACE_ROOT), description="Workspace root directory"
    )
    agent_router: Optional[AgentRouterSettings] = Field(
        default_factory=AgentRouterSettings
    )
    memory: Optional[MemorySettings] = Field(default_factory=MemorySettings)
    voice: Optional[VoiceSettings] = Field(default_factory=VoiceSettings)

    @property
    def browser_config(self) -> Optional[BrowserSettings]:
        """
        Compatibility property for browser_config access.
        Returns the browser settings object.
        """
        return self.browser

    def get_browser_config(self) -> Dict:
        """
        Get browser configuration as a dictionary for backwards compatibility.
        Returns a dictionary with browser settings.
        """
        if self.browser is None:
            return {
                "headless": False,
                "disable_security": True,
                "chrome_instance_path": None,
                "extra_chromium_args": [],
            }

        return {
            "headless": self.browser.headless,
            "disable_security": self.browser.disable_security,
            "chrome_instance_path": self.browser.chrome_instance_path,
            "extra_chromium_args": self.browser.extra_chromium_args,
            "wss_url": self.browser.wss_url,
            "cdp_url": self.browser.cdp_url,
            "proxy": self.browser.proxy,
        }


# Thread-local storage for config
_thread_local = threading.local()


def load_config(config_path: Optional[str] = None) -> Config:
    """Load configuration from a TOML file."""
    if config_path is None:
        config_path = PROJECT_ROOT / "config" / "config.toml"

    # Initialize MCP config
    mcp_config = MCPConfig(server_reference="app.mcp.server")

    try:
        with open(config_path, "rb") as f:
            config_dict = tomllib.load(f)

        # Create vision settings first if present
        vision_settings = None
        if "llm" in config_dict and "vision" in config_dict["llm"]:
            vision_settings = VisionSettings(**config_dict["llm"]["vision"])

        # Create LLM settings with proper vision settings
        if "llm" in config_dict:
            llm_dict = config_dict["llm"].copy()
            if vision_settings:
                llm_dict["vision"] = vision_settings
            config_dict["llm"] = LLMSettings(**llm_dict)

        # Handle browser configuration
        if "browser" in config_dict:
            config_dict["browser"] = BrowserSettings(**config_dict["browser"])

        # Set workspace root if not present
        if "workspace_root" not in config_dict:
            config_dict["workspace_root"] = str(WORKSPACE_ROOT)

        return Config(**config_dict)

    except Exception as e:
        print(f"Error loading config: {e}")
        print("Using default configuration")

        # Create default vision settings
        vision_settings = VisionSettings(
            enabled=False,
            model="llava-v1.6-mistral-7b",
            model_path="models/llava-model.gguf",
            clip_model_path="models/mmproj-model.gguf",
            max_tokens=2048,
            temperature=0.1,
            n_gpu_layers=0,
        )

        # Create default LLM settings
        llm_settings = LLMSettings(
            model="llama-jb",
            model_path="models/llama-jb.gguf",
            max_tokens=2048,
            temperature=0.0,
            vision=vision_settings,  # Pass VisionSettings instance
        )

        # Return default config
        return Config(
            llm=llm_settings,
            browser=BrowserSettings(headless=False, disable_security=True),
            workspace_root=str(WORKSPACE_ROOT),
            mcp_config=mcp_config,
        )


def get_config(config_path: Optional[str] = None) -> Config:
    """
    Get the configuration for the current thread.
    Args:
        config_path: Optional path to configuration file
    Returns:
        Config object
    """
    # If a specific config path is provided, always load it fresh
    if config_path:
        return load_config(config_path)

    # Otherwise use thread-local caching
    if not hasattr(_thread_local, "config"):
        _thread_local.config = load_config()
    return _thread_local.config


# Global config instance
config = get_config()
