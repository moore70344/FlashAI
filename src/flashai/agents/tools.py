"""
Tool System for FlashAI Agents

Provides a framework for defining and using tools that agents can invoke.
Tools can perform actions like:
- File operations
- Web searches
- Code execution
- API calls
- Database queries
"""

from __future__ import annotations

import asyncio
import json
import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Callable, Optional, Type
from pathlib import Path


logger = logging.getLogger(__name__)


@dataclass
class ToolParameter:
    """Definition of a tool parameter."""

    name: str
    description: str
    type: str = "string"
    required: bool = True
    default: Any = None
    enum: Optional[list[Any]] = None

    def to_schema(self) -> dict[str, Any]:
        schema = {
            "type": self.type,
            "description": self.description,
        }
        if self.enum:
            schema["enum"] = self.enum
        return schema


@dataclass
class ToolDefinition:
    """Definition of a tool that agents can use."""

    name: str
    description: str
    parameters: list[ToolParameter] = field(default_factory=list)
    returns: str = "string"
    category: str = "general"
    requires_approval: bool = False
    timeout: float = 30.0

    def to_schema(self) -> dict[str, Any]:
        """Convert to JSON Schema for LLM tool calling."""
        properties = {}
        required = []

        for param in self.parameters:
            properties[param.name] = param.to_schema()
            if param.required:
                required.append(param.name)

        return {
            "name": self.name,
            "description": self.description,
            "parameters": {
                "type": "object",
                "properties": properties,
                "required": required,
            },
        }


class Tool(ABC):
    """
    Base class for tools that agents can use.

    Subclass this to create custom tools.
    """

    def __init__(self, definition: ToolDefinition):
        self.definition = definition
        self.name = definition.name

    @abstractmethod
    async def execute(self, **kwargs) -> Any:
        """
        Execute the tool with given parameters.

        Args:
            **kwargs: Tool-specific parameters

        Returns:
            Tool execution result
        """
        pass

    def validate_params(self, params: dict[str, Any]) -> tuple[bool, Optional[str]]:
        """Validate parameters against definition."""
        for param_def in self.definition.parameters:
            if param_def.required and param_def.name not in params:
                return False, f"Missing required parameter: {param_def.name}"

            if param_def.name in params and param_def.enum:
                if params[param_def.name] not in param_def.enum:
                    return False, f"Invalid value for {param_def.name}. Must be one of: {param_def.enum}"

        return True, None


class FunctionTool(Tool):
    """A tool that wraps a Python function."""

    def __init__(
        self,
        name: str,
        description: str,
        func: Callable,
        parameters: Optional[list[ToolParameter]] = None,
    ):
        definition = ToolDefinition(
            name=name,
            description=description,
            parameters=parameters or [],
        )
        super().__init__(definition)
        self._func = func

    async def execute(self, **kwargs) -> Any:
        valid, error = self.validate_params(kwargs)
        if not valid:
            raise ValueError(error)

        if asyncio.iscoroutinefunction(self._func):
            return await self._func(**kwargs)
        else:
            return self._func(**kwargs)


class ToolRegistry:
    """
    Registry for managing available tools.

    Tools can be registered and retrieved by name.
    """

    def __init__(self):
        self._tools: dict[str, Tool] = {}
        self._categories: dict[str, list[str]] = {}

    def register(self, tool: Tool) -> None:
        """Register a tool."""
        self._tools[tool.name] = tool

        category = tool.definition.category
        if category not in self._categories:
            self._categories[category] = []
        self._categories[category].append(tool.name)

        logger.info(f"Registered tool: {tool.name}")

    def register_function(
        self,
        name: str,
        description: str,
        func: Callable,
        parameters: Optional[list[ToolParameter]] = None,
    ) -> None:
        """Register a function as a tool."""
        tool = FunctionTool(name, description, func, parameters)
        self.register(tool)

    def get(self, name: str) -> Optional[Tool]:
        """Get a tool by name."""
        return self._tools.get(name)

    def list_tools(self) -> dict[str, ToolDefinition]:
        """List all registered tools."""
        return {name: tool.definition for name, tool in self._tools.items()}

    def list_by_category(self, category: str) -> list[Tool]:
        """List tools in a category."""
        tool_names = self._categories.get(category, [])
        return [self._tools[name] for name in tool_names if name in self._tools]

    def get_schemas(self) -> list[dict[str, Any]]:
        """Get JSON schemas for all tools (for LLM tool calling)."""
        return [tool.definition.to_schema() for tool in self._tools.values()]


# Built-in tools

class FileReadTool(Tool):
    """Tool for reading file contents."""

    def __init__(self, allowed_paths: Optional[list[Path]] = None):
        definition = ToolDefinition(
            name="file_read",
            description="Read the contents of a file",
            parameters=[
                ToolParameter(
                    name="path",
                    description="Path to the file to read",
                    type="string",
                ),
                ToolParameter(
                    name="encoding",
                    description="File encoding",
                    type="string",
                    required=False,
                    default="utf-8",
                ),
            ],
            category="file",
        )
        super().__init__(definition)
        self.allowed_paths = allowed_paths

    async def execute(self, path: str, encoding: str = "utf-8") -> str:
        file_path = Path(path)

        # Security check
        if self.allowed_paths:
            allowed = any(
                file_path.resolve().is_relative_to(p.resolve())
                for p in self.allowed_paths
            )
            if not allowed:
                raise PermissionError(f"Access denied to path: {path}")

        if not file_path.exists():
            raise FileNotFoundError(f"File not found: {path}")

        return file_path.read_text(encoding=encoding)


class FileWriteTool(Tool):
    """Tool for writing file contents."""

    def __init__(self, allowed_paths: Optional[list[Path]] = None):
        definition = ToolDefinition(
            name="file_write",
            description="Write content to a file",
            parameters=[
                ToolParameter(
                    name="path",
                    description="Path to the file to write",
                    type="string",
                ),
                ToolParameter(
                    name="content",
                    description="Content to write to the file",
                    type="string",
                ),
                ToolParameter(
                    name="encoding",
                    description="File encoding",
                    type="string",
                    required=False,
                    default="utf-8",
                ),
            ],
            category="file",
            requires_approval=True,
        )
        super().__init__(definition)
        self.allowed_paths = allowed_paths

    async def execute(self, path: str, content: str, encoding: str = "utf-8") -> str:
        file_path = Path(path)

        # Security check
        if self.allowed_paths:
            allowed = any(
                file_path.resolve().is_relative_to(p.resolve())
                for p in self.allowed_paths
            )
            if not allowed:
                raise PermissionError(f"Access denied to path: {path}")

        file_path.parent.mkdir(parents=True, exist_ok=True)
        file_path.write_text(content, encoding=encoding)

        return f"Successfully wrote {len(content)} bytes to {path}"


class WebSearchTool(Tool):
    """Tool for searching the web."""

    def __init__(self):
        definition = ToolDefinition(
            name="web_search",
            description="Search the web for information",
            parameters=[
                ToolParameter(
                    name="query",
                    description="Search query",
                    type="string",
                ),
                ToolParameter(
                    name="num_results",
                    description="Number of results to return",
                    type="integer",
                    required=False,
                    default=5,
                ),
            ],
            category="web",
        )
        super().__init__(definition)

    async def execute(self, query: str, num_results: int = 5) -> list[dict[str, str]]:
        # This is a placeholder - in production, would use a real search API
        logger.info(f"Web search: {query}")
        return [
            {
                "title": f"Result {i+1} for: {query}",
                "url": f"https://example.com/search?q={query}&page={i}",
                "snippet": f"This is a placeholder result for the search query: {query}",
            }
            for i in range(num_results)
        ]


class CodeExecutionTool(Tool):
    """Tool for executing Python code in a sandbox."""

    def __init__(self, sandbox: bool = True):
        definition = ToolDefinition(
            name="execute_code",
            description="Execute Python code and return the result",
            parameters=[
                ToolParameter(
                    name="code",
                    description="Python code to execute",
                    type="string",
                ),
                ToolParameter(
                    name="timeout",
                    description="Execution timeout in seconds",
                    type="number",
                    required=False,
                    default=10.0,
                ),
            ],
            category="code",
            requires_approval=True,
        )
        super().__init__(definition)
        self.sandbox = sandbox

    async def execute(self, code: str, timeout: float = 10.0) -> dict[str, Any]:
        import io
        import sys
        from contextlib import redirect_stdout, redirect_stderr

        # Capture output
        stdout_capture = io.StringIO()
        stderr_capture = io.StringIO()

        result = None
        error = None

        try:
            # Create restricted globals for sandbox
            if self.sandbox:
                restricted_globals = {
                    "__builtins__": {
                        "print": print,
                        "len": len,
                        "range": range,
                        "str": str,
                        "int": int,
                        "float": float,
                        "list": list,
                        "dict": dict,
                        "set": set,
                        "tuple": tuple,
                        "bool": bool,
                        "sum": sum,
                        "min": min,
                        "max": max,
                        "abs": abs,
                        "round": round,
                        "sorted": sorted,
                        "enumerate": enumerate,
                        "zip": zip,
                        "map": map,
                        "filter": filter,
                    }
                }
            else:
                restricted_globals = {}

            with redirect_stdout(stdout_capture), redirect_stderr(stderr_capture):
                exec(code, restricted_globals)

                # Try to get a result variable
                if "result" in restricted_globals:
                    result = restricted_globals["result"]

        except Exception as e:
            error = str(e)

        return {
            "stdout": stdout_capture.getvalue(),
            "stderr": stderr_capture.getvalue(),
            "result": result,
            "error": error,
        }


class CalculatorTool(Tool):
    """Simple calculator tool."""

    def __init__(self):
        definition = ToolDefinition(
            name="calculator",
            description="Perform mathematical calculations",
            parameters=[
                ToolParameter(
                    name="expression",
                    description="Mathematical expression to evaluate",
                    type="string",
                ),
            ],
            category="utility",
        )
        super().__init__(definition)

    async def execute(self, expression: str) -> float:
        import ast
        import operator

        # Safe evaluation of mathematical expressions
        allowed_operators = {
            ast.Add: operator.add,
            ast.Sub: operator.sub,
            ast.Mult: operator.mul,
            ast.Div: operator.truediv,
            ast.Pow: operator.pow,
            ast.USub: operator.neg,
            ast.UAdd: operator.pos,
        }

        def _eval(node):
            if isinstance(node, ast.Constant):
                return node.value
            elif isinstance(node, ast.BinOp):
                left = _eval(node.left)
                right = _eval(node.right)
                return allowed_operators[type(node.op)](left, right)
            elif isinstance(node, ast.UnaryOp):
                operand = _eval(node.operand)
                return allowed_operators[type(node.op)](operand)
            else:
                raise ValueError(f"Unsupported operation: {type(node)}")

        try:
            tree = ast.parse(expression, mode="eval")
            return _eval(tree.body)
        except Exception as e:
            raise ValueError(f"Invalid expression: {e}")


def create_default_registry(base_path: Optional[Path] = None) -> ToolRegistry:
    """Create a tool registry with default tools."""
    registry = ToolRegistry()

    # File tools
    allowed_paths = [base_path] if base_path else None
    registry.register(FileReadTool(allowed_paths))
    registry.register(FileWriteTool(allowed_paths))

    # Web tools
    registry.register(WebSearchTool())

    # Code tools
    registry.register(CodeExecutionTool(sandbox=True))

    # Utility tools
    registry.register(CalculatorTool())

    return registry
