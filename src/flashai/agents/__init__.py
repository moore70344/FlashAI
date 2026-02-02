"""
FlashAI Agentic Framework

Provides infrastructure for building autonomous AI agents that can:
- Execute multi-step tasks
- Use tools and external services
- Maintain context and memory
- Coordinate with other agents
"""

from flashai.agents.base import Agent, AgentConfig, AgentResult
from flashai.agents.executor import AgentExecutor
from flashai.agents.tools import Tool, ToolRegistry
from flashai.agents.memory import AgentMemory
from flashai.agents.orchestrator import AgentOrchestrator

__all__ = [
    "Agent",
    "AgentConfig",
    "AgentResult",
    "AgentExecutor",
    "Tool",
    "ToolRegistry",
    "AgentMemory",
    "AgentOrchestrator",
]
