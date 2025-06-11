from app.tool.base import BaseTool
from app.tool.bash import Bash
from app.tool.browser_use_tool import BrowserUseTool
from app.tool.create_chat_completion import CreateChatCompletion
from app.tool.planning import PlanningTool
from app.tool.str_replace_editor import StrReplaceEditor
from app.tool.terminate import Terminate
from app.tool.tool_collection import ToolCollection
from app.tool.web_search import WebSearch

# Import computer control tools after base tools to avoid circular imports
try:
    from app.tool.automation import AutomationTool
    from app.tool.computer_control import ComputerControlTool

    _computer_tools_available = True
except ImportError as e:
    print(f"Warning: Computer control tools not available: {e}")
    ComputerControlTool = None
    AutomationTool = None
    _computer_tools_available = False


__all__ = [
    "BaseTool",
    "Bash",
    "BrowserUseTool",
    "Terminate",
    "StrReplaceEditor",
    "WebSearch",
    "ToolCollection",
    "CreateChatCompletion",
    "PlanningTool",
]

if _computer_tools_available:
    __all__.extend(["ComputerControlTool", "AutomationTool"])
