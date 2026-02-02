"""
Reasoning Chain - Multi-step reasoning with energy verification.

Implements chain-of-thought reasoning where each step is verified
through energy minimization to ensure coherence and correctness.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Optional
from enum import Enum

import torch
import torch.nn as nn


class ReasoningStepType(Enum):
    """Types of reasoning steps."""
    OBSERVATION = "observation"
    HYPOTHESIS = "hypothesis"
    INFERENCE = "inference"
    VERIFICATION = "verification"
    CONCLUSION = "conclusion"


@dataclass
class ReasoningStep:
    """A single step in the reasoning chain."""

    step_id: int
    step_type: ReasoningStepType
    content: str
    embedding: Optional[torch.Tensor] = None
    energy: float = 0.0
    confidence: float = 0.0
    dependencies: list[int] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "step_id": self.step_id,
            "step_type": self.step_type.value,
            "content": self.content,
            "energy": self.energy,
            "confidence": self.confidence,
            "dependencies": self.dependencies,
            "metadata": self.metadata,
        }


@dataclass
class ReasoningChain:
    """
    A complete reasoning chain from query to conclusion.

    The chain tracks:
    - Individual reasoning steps
    - Energy scores for verification
    - Dependencies between steps
    - Overall coherence metrics
    """

    query: str
    steps: list[ReasoningStep] = field(default_factory=list)
    total_energy: float = 0.0
    coherence_score: float = 0.0
    is_complete: bool = False
    final_answer: Optional[str] = None

    def add_step(
        self,
        step_type: ReasoningStepType,
        content: str,
        embedding: Optional[torch.Tensor] = None,
        energy: float = 0.0,
        confidence: float = 0.0,
        dependencies: Optional[list[int]] = None,
        metadata: Optional[dict[str, Any]] = None,
    ) -> ReasoningStep:
        """Add a new reasoning step to the chain."""
        step = ReasoningStep(
            step_id=len(self.steps),
            step_type=step_type,
            content=content,
            embedding=embedding,
            energy=energy,
            confidence=confidence,
            dependencies=dependencies or [],
            metadata=metadata or {},
        )
        self.steps.append(step)
        return step

    def get_step(self, step_id: int) -> Optional[ReasoningStep]:
        """Get a step by ID."""
        if 0 <= step_id < len(self.steps):
            return self.steps[step_id]
        return None

    def get_step_embeddings(self) -> Optional[torch.Tensor]:
        """Get stacked embeddings of all steps."""
        embeddings = [s.embedding for s in self.steps if s.embedding is not None]
        if not embeddings:
            return None
        return torch.stack(embeddings)

    def compute_average_energy(self) -> float:
        """Compute average energy across all steps."""
        if not self.steps:
            return 0.0
        return sum(s.energy for s in self.steps) / len(self.steps)

    def compute_average_confidence(self) -> float:
        """Compute average confidence across all steps."""
        if not self.steps:
            return 0.0
        return sum(s.confidence for s in self.steps) / len(self.steps)

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "query": self.query,
            "steps": [s.to_dict() for s in self.steps],
            "total_energy": self.total_energy,
            "coherence_score": self.coherence_score,
            "is_complete": self.is_complete,
            "final_answer": self.final_answer,
            "metrics": {
                "num_steps": len(self.steps),
                "avg_energy": self.compute_average_energy(),
                "avg_confidence": self.compute_average_confidence(),
            },
        }


class ReasoningChainBuilder(nn.Module):
    """
    Builds reasoning chains with energy-guided generation.

    Uses energy minimization to:
    1. Generate candidate reasoning steps
    2. Verify step coherence
    3. Select optimal paths
    """

    def __init__(
        self,
        hidden_dim: int,
        vocab_size: int = 50000,
        max_steps: int = 10,
        num_heads: int = 8,
    ):
        super().__init__()

        self.hidden_dim = hidden_dim
        self.max_steps = max_steps

        # Token embeddings
        self.token_embedding = nn.Embedding(vocab_size, hidden_dim)
        self.position_embedding = nn.Embedding(512, hidden_dim)

        # Step type embeddings
        self.step_type_embedding = nn.Embedding(
            len(ReasoningStepType),
            hidden_dim,
        )

        # Reasoning generator (decoder-style)
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=hidden_dim,
            nhead=num_heads,
            dim_feedforward=hidden_dim * 4,
            dropout=0.1,
            activation="gelu",
            batch_first=True,
        )
        self.reasoning_decoder = nn.TransformerDecoder(
            decoder_layer,
            num_layers=4,
        )

        # Output projection
        self.output_projection = nn.Linear(hidden_dim, vocab_size)

        # Step type classifier
        self.step_type_classifier = nn.Linear(hidden_dim, len(ReasoningStepType))

        # Energy estimation head
        self.energy_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.GELU(),
            nn.Linear(hidden_dim // 2, 1),
        )

    def encode_query(self, query_tokens: torch.Tensor) -> torch.Tensor:
        """Encode query tokens to embeddings."""
        batch_size, seq_len = query_tokens.shape
        positions = torch.arange(seq_len, device=query_tokens.device)

        token_embeds = self.token_embedding(query_tokens)
        pos_embeds = self.position_embedding(positions)

        return token_embeds + pos_embeds

    def generate_step(
        self,
        query_embedding: torch.Tensor,
        previous_steps: Optional[torch.Tensor] = None,
        step_type: Optional[ReasoningStepType] = None,
    ) -> tuple[torch.Tensor, torch.Tensor, float]:
        """
        Generate a reasoning step.

        Args:
            query_embedding: Encoded query [batch, seq, hidden]
            previous_steps: Embeddings of previous steps [batch, num_steps, hidden]
            step_type: Optional forced step type

        Returns:
            Tuple of (step_embedding, logits, energy)
        """
        # Prepare memory from query and previous steps
        if previous_steps is not None:
            memory = torch.cat([query_embedding, previous_steps], dim=1)
        else:
            memory = query_embedding

        # Generate step embedding
        # Start with step type embedding if provided
        if step_type is not None:
            type_idx = torch.tensor(
                [list(ReasoningStepType).index(step_type)],
                device=query_embedding.device,
            )
            step_start = self.step_type_embedding(type_idx).unsqueeze(0)
        else:
            step_start = torch.zeros(
                1, 1, self.hidden_dim,
                device=query_embedding.device,
            )

        # Decode step
        step_embedding = self.reasoning_decoder(
            step_start.expand(query_embedding.shape[0], -1, -1),
            memory,
        )

        # Get output logits
        logits = self.output_projection(step_embedding)

        # Estimate energy
        energy = self.energy_head(step_embedding.mean(dim=1)).squeeze(-1)

        return step_embedding.squeeze(1), logits.squeeze(1), energy.mean().item()

    def predict_step_type(self, step_embedding: torch.Tensor) -> ReasoningStepType:
        """Predict the type of a reasoning step."""
        logits = self.step_type_classifier(step_embedding)
        type_idx = logits.argmax(dim=-1).item()
        return list(ReasoningStepType)[type_idx]

    def build_chain(
        self,
        query_embedding: torch.Tensor,
        max_steps: Optional[int] = None,
        energy_threshold: float = 0.5,
    ) -> list[tuple[torch.Tensor, ReasoningStepType, float]]:
        """
        Build a complete reasoning chain.

        Args:
            query_embedding: Encoded query
            max_steps: Maximum number of steps (default: self.max_steps)
            energy_threshold: Stop if energy exceeds this threshold

        Returns:
            List of (step_embedding, step_type, energy) tuples
        """
        max_steps = max_steps or self.max_steps
        chain = []
        previous_steps = None

        for _ in range(max_steps):
            step_embed, _, energy = self.generate_step(
                query_embedding,
                previous_steps,
            )

            step_type = self.predict_step_type(step_embed)

            chain.append((step_embed, step_type, energy))

            # Update previous steps
            if previous_steps is None:
                previous_steps = step_embed.unsqueeze(1)
            else:
                previous_steps = torch.cat([
                    previous_steps,
                    step_embed.unsqueeze(1),
                ], dim=1)

            # Check termination conditions
            if step_type == ReasoningStepType.CONCLUSION:
                break
            if energy > energy_threshold:
                break

        return chain
