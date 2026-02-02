"""
Base Agent Classes for FlashAI Agentic Framework

Provides the foundation for building autonomous AI agents with:
- Configurable behavior and capabilities
- Tool usage and external integrations
- Memory and context management
- Energy-based reasoning integration
"""

from __future__ import annotations

import asyncio
import logging
import uuid
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Callable, Optional, TYPE_CHECKING

if TYPE_CHECKING:
    from flashai.core.engine import FlashAIEngine
    from flashai.agents.tools import Tool, ToolRegistry
    from flashai.agents.memory import AgentMemory


logger = logging.getLogger(__name__)


class AgentStatus(Enum):
    """Status of an agent."""
    IDLE = "idle"
    RUNNING = "running"
    WAITING = "waiting"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


class AgentCapability(Enum):
    """Capabilities an agent can have."""
    REASONING = "reasoning"           # Can perform energy-based reasoning
    LEARNING = "learning"             # Can learn from data
    TOOL_USE = "tool_use"             # Can use external tools
    WEB_SEARCH = "web_search"         # Can search the web
    FILE_ACCESS = "file_access"       # Can read/write files
    CODE_EXECUTION = "code_execution" # Can execute code
    MEMORY = "memory"                 # Has persistent memory
    COMMUNICATION = "communication"   # Can communicate with other agents
    SCREEN_ACCESS = "screen_access"   # Can capture/analyze screen


@dataclass
class AgentConfig:
    """Configuration for an agent."""

    name: str
    description: str = ""
    capabilities: list[AgentCapability] = field(default_factory=list)
    max_iterations: int = 50
    timeout_seconds: float = 300.0
    retry_on_error: bool = True
    max_retries: int = 3
    verbose: bool = False

    # Tool configuration
    allowed_tools: list[str] = field(default_factory=list)
    tool_timeout: float = 30.0

    # Memory configuration
    use_memory: bool = True
    memory_window: int = 20
    persist_memory: bool = True

    # Reasoning configuration
    reasoning_depth: int = 10
    confidence_threshold: float = 0.7
    energy_threshold: float = 0.5

    # Safety configuration
    require_approval_for: list[str] = field(default_factory=list)
    sandbox_code_execution: bool = True

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "description": self.description,
            "capabilities": [c.value for c in self.capabilities],
            "max_iterations": self.max_iterations,
            "timeout_seconds": self.timeout_seconds,
            "allowed_tools": self.allowed_tools,
            "use_memory": self.use_memory,
            "reasoning_depth": self.reasoning_depth,
            "confidence_threshold": self.confidence_threshold,
        }


@dataclass
class AgentResult:
    """Result of an agent execution."""

    agent_id: str
    status: AgentStatus
    result: Any = None
    error: Optional[str] = None
    iterations: int = 0
    duration_seconds: float = 0.0
    tool_calls: list[dict[str, Any]] = field(default_factory=list)
    reasoning_steps: list[dict[str, Any]] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def success(self) -> bool:
        return self.status == AgentStatus.COMPLETED

    def to_dict(self) -> dict[str, Any]:
        return {
            "agent_id": self.agent_id,
            "status": self.status.value,
            "result": self.result,
            "error": self.error,
            "iterations": self.iterations,
            "duration_seconds": self.duration_seconds,
            "tool_calls_count": len(self.tool_calls),
            "reasoning_steps_count": len(self.reasoning_steps),
            "metadata": self.metadata,
        }


@dataclass
class AgentMessage:
    """Message passed between agents or components."""

    sender: str
    content: Any
    message_type: str = "text"
    timestamp: str = field(default_factory=lambda: datetime.utcnow().isoformat())
    metadata: dict[str, Any] = field(default_factory=dict)


class Agent(ABC):
    """
    Base class for FlashAI agents.

    An agent is an autonomous entity that can:
    - Receive tasks and goals
    - Plan and execute actions
    - Use tools and external services
    - Learn and adapt over time
    - Communicate with other agents

    Subclass this to create custom agents with specific behaviors.
    """

    def __init__(
        self,
        config: AgentConfig,
        engine: Optional["FlashAIEngine"] = None,
        tool_registry: Optional["ToolRegistry"] = None,
        memory: Optional["AgentMemory"] = None,
    ):
        self.agent_id = f"agent_{uuid.uuid4().hex[:8]}"
        self.config = config
        self.engine = engine
        self.tool_registry = tool_registry
        self.memory = memory

        self.status = AgentStatus.IDLE
        self._current_task: Optional[str] = None
        self._iteration_count = 0
        self._start_time: Optional[datetime] = None
        self._hooks: dict[str, list[Callable]] = {}

        logger.info(f"Agent created: {self.agent_id} ({config.name})")

    @property
    def name(self) -> str:
        return self.config.name

    @property
    def capabilities(self) -> list[AgentCapability]:
        return self.config.capabilities

    def has_capability(self, capability: AgentCapability) -> bool:
        return capability in self.capabilities

    @abstractmethod
    async def plan(self, task: str, context: Optional[dict[str, Any]] = None) -> list[dict[str, Any]]:
        """
        Create a plan for accomplishing a task.

        Args:
            task: The task description
            context: Optional context information

        Returns:
            List of planned steps
        """
        pass

    @abstractmethod
    async def execute_step(self, step: dict[str, Any]) -> dict[str, Any]:
        """
        Execute a single step of a plan.

        Args:
            step: The step to execute

        Returns:
            Step result
        """
        pass

    async def run(
        self,
        task: str,
        context: Optional[dict[str, Any]] = None,
    ) -> AgentResult:
        """
        Run the agent on a task.

        Args:
            task: The task to accomplish
            context: Optional context information

        Returns:
            Agent execution result
        """
        self.status = AgentStatus.RUNNING
        self._current_task = task
        self._iteration_count = 0
        self._start_time = datetime.utcnow()

        tool_calls = []
        reasoning_steps = []
        result = None
        error = None

        try:
            # Add to memory
            if self.memory:
                await self.memory.add_message(AgentMessage(
                    sender="user",
                    content=task,
                    message_type="task",
                ))

            # Trigger pre-run hooks
            await self._trigger_hook("pre_run", {"task": task, "context": context})

            # Create plan
            plan = await self.plan(task, context)
            logger.info(f"Agent {self.agent_id} created plan with {len(plan)} steps")

            # Execute plan
            for step in plan:
                if self._iteration_count >= self.config.max_iterations:
                    raise RuntimeError(f"Max iterations ({self.config.max_iterations}) exceeded")

                # Check timeout
                elapsed = (datetime.utcnow() - self._start_time).total_seconds()
                if elapsed > self.config.timeout_seconds:
                    raise TimeoutError(f"Agent timeout ({self.config.timeout_seconds}s)")

                self._iteration_count += 1

                # Execute step
                step_result = await self.execute_step(step)

                # Track tool calls
                if step.get("type") == "tool_call":
                    tool_calls.append({
                        "tool": step.get("tool"),
                        "input": step.get("input"),
                        "output": step_result,
                    })

                # Track reasoning
                if step.get("type") == "reasoning":
                    reasoning_steps.append(step_result)

                # Check if done
                if step_result.get("done"):
                    result = step_result.get("result")
                    break

            self.status = AgentStatus.COMPLETED

        except asyncio.CancelledError:
            self.status = AgentStatus.CANCELLED
            error = "Agent execution cancelled"

        except Exception as e:
            self.status = AgentStatus.FAILED
            error = str(e)
            logger.error(f"Agent {self.agent_id} failed: {e}")

        finally:
            duration = (datetime.utcnow() - self._start_time).total_seconds() if self._start_time else 0

            # Trigger post-run hooks
            await self._trigger_hook("post_run", {
                "task": task,
                "result": result,
                "error": error,
            })

        return AgentResult(
            agent_id=self.agent_id,
            status=self.status,
            result=result,
            error=error,
            iterations=self._iteration_count,
            duration_seconds=duration,
            tool_calls=tool_calls,
            reasoning_steps=reasoning_steps,
        )

    async def reason(self, query: str, context: Optional[dict[str, Any]] = None) -> dict[str, Any]:
        """Perform reasoning using the FlashAI engine."""
        if not self.engine:
            raise RuntimeError("No engine configured for reasoning")

        if not self.has_capability(AgentCapability.REASONING):
            raise RuntimeError("Agent does not have reasoning capability")

        return await self.engine.reason(query=query, context=context)

    async def use_tool(self, tool_name: str, input_data: Any) -> Any:
        """Use a tool from the registry."""
        if not self.tool_registry:
            raise RuntimeError("No tool registry configured")

        if not self.has_capability(AgentCapability.TOOL_USE):
            raise RuntimeError("Agent does not have tool_use capability")

        if self.config.allowed_tools and tool_name not in self.config.allowed_tools:
            raise PermissionError(f"Tool {tool_name} not allowed for this agent")

        tool = self.tool_registry.get(tool_name)
        if not tool:
            raise ValueError(f"Tool not found: {tool_name}")

        return await tool.execute(input_data)

    async def communicate(self, recipient: str, message: Any) -> None:
        """Send a message to another agent."""
        if not self.has_capability(AgentCapability.COMMUNICATION):
            raise RuntimeError("Agent does not have communication capability")

        # This would be implemented by the orchestrator
        raise NotImplementedError("Communication requires orchestrator")

    def register_hook(self, event: str, callback: Callable) -> None:
        """Register a callback for an event."""
        if event not in self._hooks:
            self._hooks[event] = []
        self._hooks[event].append(callback)

    async def _trigger_hook(self, event: str, data: dict[str, Any]) -> None:
        """Trigger all hooks for an event."""
        for callback in self._hooks.get(event, []):
            try:
                if asyncio.iscoroutinefunction(callback):
                    await callback(data)
                else:
                    callback(data)
            except Exception as e:
                logger.error(f"Hook error: {e}")

    async def stop(self) -> None:
        """Stop the agent gracefully."""
        self.status = AgentStatus.CANCELLED
        logger.info(f"Agent {self.agent_id} stopped")


class SimpleReasoningAgent(Agent):
    """
    A simple agent that uses energy-based reasoning to accomplish tasks.

    This agent:
    1. Analyzes the task using reasoning
    2. Creates a simple plan
    3. Executes each step with reasoning verification
    """

    async def plan(self, task: str, context: Optional[dict[str, Any]] = None) -> list[dict[str, Any]]:
        """Create a simple reasoning-based plan."""
        steps = []

        # Initial analysis
        steps.append({
            "type": "reasoning",
            "action": "analyze",
            "query": f"Analyze this task and identify the key components: {task}",
        })

        # Main execution
        steps.append({
            "type": "reasoning",
            "action": "solve",
            "query": f"Solve this task step by step: {task}",
        })

        # Verification
        steps.append({
            "type": "reasoning",
            "action": "verify",
            "query": "Verify the solution is correct and complete.",
            "done": True,
        })

        return steps

    async def execute_step(self, step: dict[str, Any]) -> dict[str, Any]:
        """Execute a reasoning step."""
        if step.get("type") != "reasoning":
            return {"error": "Unknown step type"}

        result = await self.reason(step["query"])

        return {
            "action": step.get("action"),
            "result": result,
            "done": step.get("done", False),
            "energy": result.get("final_energy"),
            "confidence": result.get("confidence"),
        }


class ToolUsingAgent(Agent):
    """
    An agent that can use tools to accomplish tasks.

    This agent:
    1. Analyzes the task to determine needed tools
    2. Creates a plan with tool calls
    3. Executes tools and reasons about results
    """

    async def plan(self, task: str, context: Optional[dict[str, Any]] = None) -> list[dict[str, Any]]:
        """Create a plan that may include tool usage."""
        steps = []

        # Get available tools
        available_tools = []
        if self.tool_registry:
            available_tools = list(self.tool_registry.list_tools().keys())

        # Plan the task
        plan_query = f"""
        Task: {task}

        Available tools: {', '.join(available_tools) if available_tools else 'None'}

        Create a step-by-step plan to accomplish this task.
        For each step, indicate if a tool should be used.
        """

        # Use reasoning to create plan
        if self.has_capability(AgentCapability.REASONING):
            plan_result = await self.reason(plan_query, context)
            # Parse plan from result (simplified)
            steps.append({
                "type": "reasoning",
                "action": "execute_plan",
                "query": task,
                "plan_context": plan_result,
                "done": True,
            })
        else:
            steps.append({
                "type": "direct",
                "action": "execute",
                "query": task,
                "done": True,
            })

        return steps

    async def execute_step(self, step: dict[str, Any]) -> dict[str, Any]:
        """Execute a step, potentially using tools."""
        step_type = step.get("type")

        if step_type == "tool_call":
            tool_name = step.get("tool")
            tool_input = step.get("input")
            try:
                result = await self.use_tool(tool_name, tool_input)
                return {
                    "tool": tool_name,
                    "result": result,
                    "done": step.get("done", False),
                }
            except Exception as e:
                return {
                    "tool": tool_name,
                    "error": str(e),
                    "done": step.get("done", False),
                }

        elif step_type == "reasoning":
            result = await self.reason(step["query"])
            return {
                "result": result,
                "done": step.get("done", False),
            }

        else:
            return {
                "result": f"Executed: {step.get('action', 'unknown')}",
                "done": step.get("done", False),
            }
