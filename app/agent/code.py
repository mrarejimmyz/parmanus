"""Code agent based on Parmanus's CoderAgent."""

from typing import Any, Dict, List, Optional

from app.agent.toolcall import ToolCallAgent
from app.logger import logger
from app.memory import Memory
from app.tool.base import BaseTool
from app.tool.python_execute import PythonExecute


class CodeAgent(ToolCallAgent):
    """Agent specialized in code execution and development."""

    name: str = "CodeAgent"
    description: str = "An agent that can write and execute code in multiple languages"

    def __init__(self, **kwargs):
        """Initialize the CodeAgent with code execution tools."""
        super().__init__(**kwargs)

        # Initialize code execution tools
        self.tools = self._initialize_code_tools()

        # Set code-specific system prompt
        self.system_prompt = self._get_code_system_prompt()

    def _initialize_code_tools(self) -> List[BaseTool]:
        """Initialize code execution tools.

        Returns:
            List of code execution tools.
        """
        tools = []

        # Python execution tool
        try:
            python_tool = PythonExecute()
            tools.append(python_tool)
        except Exception as e:
            logger.warning(f"Failed to initialize Python tool: {e}")

        # Additional interpreters will be added here
        # C, Go, Java, Bash interpreters

        return tools

    def _get_code_system_prompt(self) -> str:
        """Get the system prompt for code agent.

        Returns:
            System prompt for code execution tasks.
        """
        return """You are a specialized code execution agent. You can:

1. Write code in multiple programming languages (Python, C, Go, Java, Bash)
2. Execute code and provide results
3. Debug and fix code issues
4. Explain code functionality
5. Optimize code performance

When writing code:
- Always test your code before providing the final version
- Handle errors gracefully
- Provide clear explanations of what the code does
- Use best practices for the specific language

Available tools:
- Python execution: For Python scripts and data analysis
- C compilation and execution: For C programs
- Go compilation and execution: For Go programs
- Java compilation and execution: For Java programs
- Bash execution: For shell scripts and system commands

Always execute code to verify it works before presenting the final solution."""

    async def process(self, prompt: str, speech_module: Optional[Any] = None) -> str:
        """Process a code-related query.

        Args:
            prompt: The code-related query or request.
            speech_module: Optional speech module for voice interaction.

        Returns:
            The result of processing the code request.
        """
        logger.info(f"CodeAgent processing: {prompt}")

        # Add user message to memory
        self.update_memory("user", prompt)

        # Execute the code task with feedback loop
        result = await self.execute_with_feedback(prompt)

        # Add result to memory
        self.update_memory("assistant", result)

        return result

    async def execute_with_feedback(self, prompt: str, max_attempts: int = 5) -> str:
        """Execute code with feedback loop for corrections.

        Args:
            prompt: The code request.
            max_attempts: Maximum number of correction attempts.

        Returns:
            The final result after corrections.
        """
        attempt = 0
        last_error = None

        while attempt < max_attempts:
            try:
                # Generate code solution
                code_response = await self.step()

                # If the response contains executable code, try to run it
                if self._contains_code(code_response):
                    execution_result = await self._execute_code_from_response(
                        code_response
                    )

                    if execution_result["success"]:
                        return f"{code_response}\n\nExecution Result:\n{execution_result['output']}"
                    else:
                        # Code failed, prepare correction prompt
                        last_error = execution_result["error"]
                        correction_prompt = f"""The code execution failed with error:
{last_error}

Please fix the code and try again. Original request: {prompt}"""

                        self.update_memory("user", correction_prompt)
                        attempt += 1
                        continue
                else:
                    # Response doesn't contain executable code, return as is
                    return code_response

            except Exception as e:
                last_error = str(e)
                logger.error(f"Code execution attempt {attempt + 1} failed: {e}")
                attempt += 1

                if attempt < max_attempts:
                    correction_prompt = f"""An error occurred: {e}
Please provide a corrected solution for: {prompt}"""
                    self.update_memory("user", correction_prompt)

        return f"Failed to execute code after {max_attempts} attempts. Last error: {last_error}"

    def _contains_code(self, response: str) -> bool:
        """Check if response contains executable code.

        Args:
            response: The response to check.

        Returns:
            True if response contains code blocks.
        """
        code_indicators = ["```python", "```c", "```go", "```java", "```bash", "```sh"]
        return any(indicator in response.lower() for indicator in code_indicators)

    async def _execute_code_from_response(self, response: str) -> Dict[str, Any]:
        """Extract and execute code from response.

        Args:
            response: Response containing code blocks.

        Returns:
            Dictionary with execution results.
        """
        # Extract code blocks from response
        code_blocks = self._extract_code_blocks(response)

        results = []
        for code_block in code_blocks:
            language = code_block["language"]
            code = code_block["code"]

            # Execute based on language
            if language == "python":
                result = await self._execute_python(code)
            elif language in ["c", "cpp"]:
                result = await self._execute_c(code)
            elif language == "go":
                result = await self._execute_go(code)
            elif language == "java":
                result = await self._execute_java(code)
            elif language in ["bash", "sh"]:
                result = await self._execute_bash(code)
            else:
                result = {
                    "success": False,
                    "error": f"Unsupported language: {language}",
                }

            results.append(result)

        # Return combined results
        if not results:
            return {"success": False, "error": "No executable code found"}

        # If any execution failed, return the first failure
        for result in results:
            if not result["success"]:
                return result

        # All executions succeeded
        combined_output = "\n".join([r["output"] for r in results if r.get("output")])
        return {"success": True, "output": combined_output}

    def _extract_code_blocks(self, text: str) -> List[Dict[str, str]]:
        """Extract code blocks from markdown text.

        Args:
            text: Text containing code blocks.

        Returns:
            List of code blocks with language and code.
        """
        import re

        # Pattern to match code blocks
        pattern = r"```(\w+)?\n(.*?)```"
        matches = re.findall(pattern, text, re.DOTALL)

        code_blocks = []
        for match in matches:
            language = match[0].lower() if match[0] else "text"
            code = match[1].strip()
            code_blocks.append({"language": language, "code": code})

        return code_blocks

    async def _execute_python(self, code: str) -> Dict[str, Any]:
        """Execute Python code.

        Args:
            code: Python code to execute.

        Returns:
            Execution result.
        """
        try:
            # Use the Python execution tool
            python_tool = next(
                (tool for tool in self.tools if isinstance(tool, PythonExecute)), None
            )
            if python_tool:
                result = await python_tool.execute(code)
                return {"success": True, "output": result}
            else:
                return {
                    "success": False,
                    "error": "Python execution tool not available",
                }
        except Exception as e:
            return {"success": False, "error": str(e)}

    async def _execute_c(self, code: str) -> Dict[str, Any]:
        """Execute C code.

        Args:
            code: C code to execute.

        Returns:
            Execution result.
        """
        # Will be implemented when C interpreter is added
        return {"success": False, "error": "C execution not yet implemented"}

    async def _execute_go(self, code: str) -> Dict[str, Any]:
        """Execute Go code.

        Args:
            code: Go code to execute.

        Returns:
            Execution result.
        """
        # Will be implemented when Go interpreter is added
        return {"success": False, "error": "Go execution not yet implemented"}

    async def _execute_java(self, code: str) -> Dict[str, Any]:
        """Execute Java code.

        Args:
            code: Java code to execute.

        Returns:
            Execution result.
        """
        # Will be implemented when Java interpreter is added
        return {"success": False, "error": "Java execution not yet implemented"}

    async def _execute_bash(self, code: str) -> Dict[str, Any]:
        """Execute Bash code.

        Args:
            code: Bash code to execute.

        Returns:
            Execution result.
        """
        # Will be implemented when Bash interpreter is added
        return {"success": False, "error": "Bash execution not yet implemented"}

    @classmethod
    async def create(cls, **kwargs) -> "CodeAgent":
        """Create a new CodeAgent instance.

        Args:
            **kwargs: Additional arguments for agent creation.

        Returns:
            Initialized CodeAgent instance.
        """
        agent = cls(**kwargs)
        await agent.initialize()
        return agent
