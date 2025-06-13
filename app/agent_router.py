from typing import Any

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


