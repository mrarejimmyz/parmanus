"""Script to fix the tool calls issue by properly updating both patch files."""

import os
import shutil
from pathlib import Path

# Get root directory
root_dir = os.path.dirname(os.path.abspath(__file__))

# Files to patch
files_to_patch = [
    os.path.join(root_dir, "app/llm_tool_patch.py"),
    os.path.join(root_dir, "app/llm_tool_patch_optimized.py"),
]

# Create backups
for file_path in files_to_patch:
    timestamp = Path(file_path).stat().st_mtime
    backup_path = f"{file_path}.backup_{int(timestamp)}"
    shutil.copy2(file_path, backup_path)
    print(f"Created backup: {backup_path}")

# Fix: Add missing _format_tool_call function to both files
for file_path in files_to_patch:
    with open(file_path, "r") as f:
        content = f.read()

    # Find the correct location to insert _format_tool_call function
    # It should be after the import statements and before _parse_tool_calls
    insertion_point = content.find("def _parse_tool_calls")

    # If insertion point found, insert the _format_tool_call function
    if insertion_point > 0:
        # Go back to the nearest function definition
        previous_def = content.rfind("def ", 0, insertion_point)

        # Insert the _format_tool_call function between the two
        format_tool_call_function = '''
def _format_tool_call(data: Dict[str, Any]) -> Dict[str, Any]:
    """Format a tool call into the expected structure."""
    try:
        if not isinstance(data, dict):
            raise ValueError(f"Tool call data must be a dictionary, got {type(data)}")

        if not data:
            raise ValueError("Tool call data cannot be empty")

        # Extract tool name from either direct name field or function name
        name = None
        if "function" in data and isinstance(data["function"], dict):
            name = data["function"].get("name")
        elif "name" in data:
            name = data["name"]

        if not isinstance(name, str) or not name.strip():
            raise ValueError(f"Tool call missing valid name, got {name}")

        # Get arguments from either direct arguments field or function arguments
        args = {}
        if "function" in data and isinstance(data["function"], dict):
            args = data["function"].get("arguments", {})
        elif "arguments" in data:
            args = data["arguments"]

        # Normalize args to a string
        if isinstance(args, (dict, list)):
            try:
                args = json.dumps(args)
            except Exception as e:
                logger.warning(
                    f"Failed to JSON encode arguments: {e}, using str() instead"
                )
                args = str(args)
        elif not isinstance(args, str):
            args = str(args)

        return {
            "id": data.get("id", f"call_{time.time_ns()}"),
            "type": "function",
            "function": {"name": name.strip(), "arguments": args},
        }
    except Exception as e:
        logger.error(f"Failed to format tool call: {e}")
        raise ValueError(f"Tool call formatting failed: {str(e)}")

'''

        # Insert the function at the appropriate location
        new_content = (
            content[:previous_def] + format_tool_call_function + content[previous_def:]
        )

        # Fix line breaks in _parse_tool_calls - make sure comment and code are on separate lines
        tool_calls_pattern = (
            "# Parse tool calls\ntool_calls = self._parse_tool_calls(completion_text)"
        )
        bad_format_pattern = (
            "# Parse tool calls tool_calls = self._parse_tool_calls(completion_text)"
        )

        if bad_format_pattern in new_content:
            new_content = new_content.replace(bad_format_pattern, tool_calls_pattern)

        # Save updated file
        with open(file_path, "w") as f:
            f.write(new_content)

        print(f"Added missing _format_tool_call function to {file_path}")
    else:
        print(f"Could not find insertion point in {file_path}")

print("Fixes applied successfully!")
