"""
FlashAI - Portable AI System with Energy-Based Reasoning Model

A self-contained AI system designed to run from portable storage,
featuring energy-based reasoning, user adaptation, and GitHub integration.
"""

__version__ = "1.0.0"
__author__ = "FlashAI Team"

from flashai.core.engine import FlashAIEngine
from flashai.core.config import FlashAIConfig
from flashai.state.manager import StateManager

__all__ = [
    "FlashAIEngine",
    "FlashAIConfig",
    "StateManager",
    "__version__",
]
