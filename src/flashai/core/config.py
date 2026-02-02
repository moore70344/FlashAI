"""
FlashAI Configuration System

Manages all configuration for the portable AI system including
paths, model parameters, and GitHub integration settings.
"""

from __future__ import annotations

import os
import yaml
from pathlib import Path
from typing import Any, Optional
from dataclasses import dataclass, field
from pydantic import BaseModel, Field


class EBMConfig(BaseModel):
    """Energy-Based Model configuration."""

    # Model architecture
    input_dim: int = Field(default=512, description="Input embedding dimension")
    hidden_dims: list[int] = Field(default=[256, 128, 64], description="Hidden layer dimensions")
    energy_dim: int = Field(default=32, description="Energy function output dimension")

    # Training parameters
    learning_rate: float = Field(default=0.001, description="Learning rate for optimization")
    energy_margin: float = Field(default=1.0, description="Margin for contrastive energy loss")
    noise_scale: float = Field(default=0.1, description="Scale for noise contrastive estimation")

    # Inference parameters
    inference_steps: int = Field(default=10, description="Number of energy minimization steps")
    step_size: float = Field(default=0.01, description="Step size for energy minimization")
    temperature: float = Field(default=1.0, description="Temperature for sampling")

    # Reasoning parameters
    max_reasoning_depth: int = Field(default=5, description="Maximum reasoning chain depth")
    constraint_weight: float = Field(default=0.5, description="Weight for constraint satisfaction")
    coherence_threshold: float = Field(default=0.7, description="Threshold for reasoning coherence")


class UserAdaptationConfig(BaseModel):
    """User adaptation and learning configuration."""

    # Profile settings
    max_interaction_history: int = Field(default=1000, description="Max interactions to store")
    preference_decay: float = Field(default=0.95, description="Decay rate for old preferences")

    # Learning parameters
    adaptation_rate: float = Field(default=0.1, description="Rate of adaptation to user patterns")
    min_samples_for_adaptation: int = Field(default=5, description="Min samples before adapting")

    # Privacy
    encrypt_profile: bool = Field(default=True, description="Encrypt user profile data")
    anonymize_logs: bool = Field(default=True, description="Anonymize interaction logs")


class GitHubConfig(BaseModel):
    """GitHub integration configuration."""

    # Repository settings
    repo_owner: str = Field(default="", description="GitHub repository owner")
    repo_name: str = Field(default="", description="GitHub repository name")
    default_branch: str = Field(default="main", description="Default branch name")

    # Webhook settings
    webhook_secret: str = Field(default="", description="Webhook secret for verification")
    webhook_events: list[str] = Field(
        default=["push", "pull_request", "issues", "issue_comment"],
        description="Events to listen for"
    )

    # API settings
    api_base_url: str = Field(default="https://api.github.com", description="GitHub API base URL")
    api_timeout: int = Field(default=30, description="API request timeout in seconds")

    # Actions
    enable_auto_learn: bool = Field(default=True, description="Auto-learn from repo changes")
    enable_issue_responses: bool = Field(default=False, description="Auto-respond to issues")


class ServerConfig(BaseModel):
    """Server configuration for the FlashAI API."""

    host: str = Field(default="127.0.0.1", description="Server host")
    port: int = Field(default=8420, description="Server port")
    workers: int = Field(default=1, description="Number of worker processes")
    reload: bool = Field(default=False, description="Enable auto-reload for development")
    cors_origins: list[str] = Field(default=["*"], description="Allowed CORS origins")


class FlashAIConfig(BaseModel):
    """Main FlashAI configuration."""

    # System identification
    system_id: str = Field(default="", description="Unique system identifier")
    system_name: str = Field(default="FlashAI", description="System display name")

    # Paths (relative to flash drive root)
    data_dir: str = Field(default="data", description="Data directory")
    models_dir: str = Field(default="models", description="Models directory")
    config_dir: str = Field(default="config", description="Config directory")

    # Sub-configurations
    ebm: EBMConfig = Field(default_factory=EBMConfig)
    user_adaptation: UserAdaptationConfig = Field(default_factory=UserAdaptationConfig)
    github: GitHubConfig = Field(default_factory=GitHubConfig)
    server: ServerConfig = Field(default_factory=ServerConfig)

    # Runtime settings
    device: str = Field(default="auto", description="Compute device: auto, cpu, cuda, mps")
    log_level: str = Field(default="INFO", description="Logging level")
    enable_telemetry: bool = Field(default=False, description="Enable usage telemetry")

    @classmethod
    def load(cls, config_path: Path | str) -> "FlashAIConfig":
        """Load configuration from YAML file."""
        config_path = Path(config_path)
        if not config_path.exists():
            return cls()

        with open(config_path, "r") as f:
            data = yaml.safe_load(f) or {}

        return cls(**data)

    def save(self, config_path: Path | str) -> None:
        """Save configuration to YAML file."""
        config_path = Path(config_path)
        config_path.parent.mkdir(parents=True, exist_ok=True)

        with open(config_path, "w") as f:
            yaml.dump(self.model_dump(), f, default_flow_style=False, sort_keys=False)

    def get_absolute_path(self, base_path: Path, relative_path: str) -> Path:
        """Convert relative path to absolute based on base path."""
        return base_path / relative_path

    def resolve_device(self) -> str:
        """Resolve 'auto' device to actual device."""
        if self.device != "auto":
            return self.device

        try:
            import torch
            if torch.cuda.is_available():
                return "cuda"
            if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
                return "mps"
        except ImportError:
            pass

        return "cpu"


def get_default_config() -> FlashAIConfig:
    """Get default FlashAI configuration."""
    return FlashAIConfig()


def load_config_from_env() -> FlashAIConfig:
    """Load configuration with environment variable overrides."""
    config = FlashAIConfig()

    # Override with environment variables
    if os.environ.get("FLASHAI_GITHUB_TOKEN"):
        pass  # Token handled separately for security

    if os.environ.get("FLASHAI_REPO_OWNER"):
        config.github.repo_owner = os.environ["FLASHAI_REPO_OWNER"]

    if os.environ.get("FLASHAI_REPO_NAME"):
        config.github.repo_name = os.environ["FLASHAI_REPO_NAME"]

    if os.environ.get("FLASHAI_WEBHOOK_SECRET"):
        config.github.webhook_secret = os.environ["FLASHAI_WEBHOOK_SECRET"]

    if os.environ.get("FLASHAI_DEVICE"):
        config.device = os.environ["FLASHAI_DEVICE"]

    return config
