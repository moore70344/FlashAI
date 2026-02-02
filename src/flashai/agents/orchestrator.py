"""
Agent Orchestrator

Coordinates multiple agents to work together on complex tasks.
"""

from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Optional, TYPE_CHECKING
from pathlib import Path

from flashai.agents.base import Agent, AgentConfig, AgentResult, AgentStatus
from flashai.agents.tools import ToolRegistry, create_default_registry
from flashai.agents.memory import AgentMemory

if TYPE_CHECKING:
    from flashai.core.engine import FlashAIEngine


logger = logging.getLogger(__name__)


@dataclass
class TaskAssignment:
    """Assignment of a task to an agent."""

    task_id: str
    agent_id: str
    task: str
    status: str = "pending"
    result: Optional[AgentResult] = None
    created_at: str = field(default_factory=lambda: datetime.utcnow().isoformat())


class AgentOrchestrator:
    """
    Orchestrates multiple agents to accomplish complex tasks.

    Features:
    - Agent registration and management
    - Task delegation and routing
    - Inter-agent communication
    - Result aggregation
    """

    def __init__(
        self,
        engine: Optional["FlashAIEngine"] = None,
        base_path: Optional[Path] = None,
    ):
        self.engine = engine
        self.base_path = Path(base_path) if base_path else Path.cwd()

        self._agents: dict[str, Agent] = {}
        self._tool_registry = create_default_registry(self.base_path)
        self._tasks: dict[str, TaskAssignment] = {}
        self._message_queue: asyncio.Queue = asyncio.Queue()

        logger.info("AgentOrchestrator initialized")

    @property
    def tool_registry(self) -> ToolRegistry:
        return self._tool_registry

    def register_agent(self, agent: Agent) -> None:
        """Register an agent with the orchestrator."""
        self._agents[agent.agent_id] = agent
        logger.info(f"Registered agent: {agent.agent_id} ({agent.name})")

    def create_agent(
        self,
        config: AgentConfig,
        agent_class: type = None,
    ) -> Agent:
        """Create and register a new agent."""
        from flashai.agents.base import SimpleReasoningAgent

        agent_class = agent_class or SimpleReasoningAgent

        # Create memory for the agent
        memory = AgentMemory(
            agent_id=f"agent_{len(self._agents)}",
            storage_path=self.base_path / "data" / "agent_memory",
            persist=config.persist_memory,
        )

        agent = agent_class(
            config=config,
            engine=self.engine,
            tool_registry=self._tool_registry,
            memory=memory,
        )

        self.register_agent(agent)
        return agent

    def get_agent(self, agent_id: str) -> Optional[Agent]:
        """Get an agent by ID."""
        return self._agents.get(agent_id)

    def list_agents(self) -> list[dict[str, Any]]:
        """List all registered agents."""
        return [
            {
                "agent_id": agent.agent_id,
                "name": agent.name,
                "status": agent.status.value,
                "capabilities": [c.value for c in agent.capabilities],
            }
            for agent in self._agents.values()
        ]

    async def assign_task(
        self,
        task: str,
        agent_id: Optional[str] = None,
        context: Optional[dict[str, Any]] = None,
    ) -> TaskAssignment:
        """
        Assign a task to an agent.

        If no agent_id is provided, selects the best available agent.
        """
        import uuid

        # Select agent
        if agent_id:
            agent = self.get_agent(agent_id)
            if not agent:
                raise ValueError(f"Agent not found: {agent_id}")
        else:
            agent = self._select_best_agent(task)
            if not agent:
                raise RuntimeError("No suitable agent available")

        # Create assignment
        task_id = f"task_{uuid.uuid4().hex[:8]}"
        assignment = TaskAssignment(
            task_id=task_id,
            agent_id=agent.agent_id,
            task=task,
        )
        self._tasks[task_id] = assignment

        # Run the task
        assignment.status = "running"
        try:
            result = await agent.run(task, context)
            assignment.result = result
            assignment.status = "completed" if result.success else "failed"
        except Exception as e:
            assignment.status = "failed"
            logger.error(f"Task {task_id} failed: {e}")

        return assignment

    def _select_best_agent(self, task: str) -> Optional[Agent]:
        """Select the best agent for a task based on capabilities."""
        available = [a for a in self._agents.values() if a.status == AgentStatus.IDLE]

        if not available:
            # Return first agent that's not failed
            for agent in self._agents.values():
                if agent.status != AgentStatus.FAILED:
                    return agent
            return None

        # Simple selection - prefer agents with more capabilities
        return max(available, key=lambda a: len(a.capabilities))

    async def run_parallel(
        self,
        tasks: list[tuple[str, Optional[str]]],
        context: Optional[dict[str, Any]] = None,
    ) -> list[TaskAssignment]:
        """Run multiple tasks in parallel."""
        async def run_task(task: str, agent_id: Optional[str]) -> TaskAssignment:
            return await self.assign_task(task, agent_id, context)

        results = await asyncio.gather(
            *[run_task(task, agent_id) for task, agent_id in tasks],
            return_exceptions=True,
        )

        assignments = []
        for result in results:
            if isinstance(result, Exception):
                logger.error(f"Parallel task failed: {result}")
            else:
                assignments.append(result)

        return assignments

    async def run_sequential(
        self,
        tasks: list[str],
        agent_id: Optional[str] = None,
        context: Optional[dict[str, Any]] = None,
    ) -> list[TaskAssignment]:
        """Run tasks sequentially, passing results between them."""
        assignments = []
        running_context = context.copy() if context else {}

        for task in tasks:
            assignment = await self.assign_task(task, agent_id, running_context)
            assignments.append(assignment)

            # Add result to context for next task
            if assignment.result and assignment.result.result:
                running_context["previous_result"] = assignment.result.result

        return assignments

    def get_task(self, task_id: str) -> Optional[TaskAssignment]:
        """Get a task assignment by ID."""
        return self._tasks.get(task_id)

    def get_task_history(self, limit: int = 50) -> list[TaskAssignment]:
        """Get recent task history."""
        tasks = list(self._tasks.values())
        tasks.sort(key=lambda t: t.created_at, reverse=True)
        return tasks[:limit]

    async def broadcast_message(self, message: Any, sender: str = "orchestrator") -> None:
        """Broadcast a message to all agents."""
        for agent in self._agents.values():
            if agent.memory:
                await agent.memory.add_message(
                    {"sender": sender, "content": message},
                    importance=0.3,
                )

    async def shutdown(self) -> None:
        """Shutdown all agents."""
        for agent in self._agents.values():
            await agent.stop()
        logger.info("All agents stopped")


class AgentExecutor:
    """
    High-level executor for running agent workflows.

    Provides a simple interface for common agent patterns.
    """

    def __init__(self, orchestrator: AgentOrchestrator):
        self.orchestrator = orchestrator

    async def run_single(
        self,
        task: str,
        agent_config: Optional[AgentConfig] = None,
        context: Optional[dict[str, Any]] = None,
    ) -> AgentResult:
        """Run a single task with a new or existing agent."""
        from flashai.agents.base import AgentCapability

        if agent_config:
            agent = self.orchestrator.create_agent(agent_config)
        else:
            # Create default agent
            config = AgentConfig(
                name="default_agent",
                description="Default task execution agent",
                capabilities=[AgentCapability.REASONING, AgentCapability.TOOL_USE],
            )
            agent = self.orchestrator.create_agent(config)

        result = await agent.run(task, context)
        return result

    async def run_workflow(
        self,
        workflow: list[dict[str, Any]],
        context: Optional[dict[str, Any]] = None,
    ) -> list[AgentResult]:
        """
        Run a workflow defined as a list of steps.

        Each step can specify:
        - task: The task to perform
        - agent: Optional agent ID or config
        - parallel: List of tasks to run in parallel
        - depends_on: Previous step results to include
        """
        results = []
        step_results: dict[str, Any] = {}

        for i, step in enumerate(workflow):
            step_context = context.copy() if context else {}

            # Add dependencies
            if "depends_on" in step:
                for dep in step["depends_on"]:
                    if dep in step_results:
                        step_context[f"step_{dep}_result"] = step_results[dep]

            if "parallel" in step:
                # Run parallel tasks
                tasks = [(t, None) for t in step["parallel"]]
                assignments = await self.orchestrator.run_parallel(tasks, step_context)
                for j, assignment in enumerate(assignments):
                    if assignment.result:
                        results.append(assignment.result)
                        step_results[f"{i}_{j}"] = assignment.result.result
            else:
                # Run single task
                assignment = await self.orchestrator.assign_task(
                    step["task"],
                    step.get("agent"),
                    step_context,
                )
                if assignment.result:
                    results.append(assignment.result)
                    step_results[str(i)] = assignment.result.result

        return results
