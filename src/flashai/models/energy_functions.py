"""
Energy Functions for the Energy-Based Reasoning Model.

These functions compute energy scores that guide reasoning. Lower energy
indicates better solutions/answers. The model learns by minimizing energy
for correct solutions and maximizing for incorrect ones.
"""

from __future__ import annotations

import math
from abc import ABC, abstractmethod
from typing import Any, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


class EnergyFunction(nn.Module, ABC):
    """
    Abstract base class for energy functions.

    Energy functions map (input, output) pairs to scalar energy values.
    Lower energy indicates higher compatibility/correctness.
    """

    @abstractmethod
    def forward(
        self,
        x: torch.Tensor,
        y: torch.Tensor,
        constraints: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Compute energy for input-output pairs.

        Args:
            x: Input embeddings [batch, input_dim]
            y: Output/answer embeddings [batch, output_dim]
            constraints: Optional constraint embeddings [batch, constraint_dim]

        Returns:
            Energy scores [batch]
        """
        pass

    @abstractmethod
    def compute_gradient(
        self,
        x: torch.Tensor,
        y: torch.Tensor,
        constraints: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Compute gradient of energy with respect to y.

        Used for energy minimization during inference.

        Args:
            x: Input embeddings [batch, input_dim]
            y: Output/answer embeddings [batch, output_dim]
            constraints: Optional constraint embeddings

        Returns:
            Gradient with respect to y [batch, output_dim]
        """
        pass


class ContrastiveEnergy(EnergyFunction):
    """
    Contrastive energy function that learns to assign low energy
    to correct input-output pairs and high energy to incorrect ones.

    Based on the NCE (Noise Contrastive Estimation) approach.
    """

    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        hidden_dims: list[int],
        constraint_dim: int = 0,
    ):
        super().__init__()

        self.input_dim = input_dim
        self.output_dim = output_dim
        self.constraint_dim = constraint_dim

        # Build the energy network
        total_input = input_dim + output_dim + constraint_dim
        layers = []
        prev_dim = total_input

        for hidden_dim in hidden_dims:
            layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.LayerNorm(hidden_dim),
                nn.GELU(),
                nn.Dropout(0.1),
            ])
            prev_dim = hidden_dim

        # Final energy output (scalar)
        layers.append(nn.Linear(prev_dim, 1))

        self.energy_net = nn.Sequential(*layers)

        # Learnable temperature for energy scaling
        self.log_temperature = nn.Parameter(torch.zeros(1))

    def forward(
        self,
        x: torch.Tensor,
        y: torch.Tensor,
        constraints: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Compute energy scores."""
        # Concatenate inputs
        if constraints is not None:
            combined = torch.cat([x, y, constraints], dim=-1)
        else:
            combined = torch.cat([x, y], dim=-1)

        # Compute raw energy
        raw_energy = self.energy_net(combined).squeeze(-1)

        # Scale by learned temperature
        temperature = torch.exp(self.log_temperature)
        energy = raw_energy / temperature

        return energy

    def compute_gradient(
        self,
        x: torch.Tensor,
        y: torch.Tensor,
        constraints: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Compute gradient for energy minimization."""
        y_grad = y.clone().requires_grad_(True)
        energy = self.forward(x, y_grad, constraints)
        energy_sum = energy.sum()
        energy_sum.backward()
        return y_grad.grad


class CoherenceEnergy(EnergyFunction):
    """
    Energy function that measures coherence/consistency of reasoning.

    Uses attention-based mechanism to detect contradictions and
    ensure logical consistency across reasoning steps.
    """

    def __init__(
        self,
        hidden_dim: int,
        num_heads: int = 4,
        num_layers: int = 2,
    ):
        super().__init__()

        self.hidden_dim = hidden_dim

        # Self-attention for coherence checking
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=num_heads,
            dim_feedforward=hidden_dim * 4,
            dropout=0.1,
            activation="gelu",
            batch_first=True,
        )
        self.coherence_encoder = nn.TransformerEncoder(
            encoder_layer,
            num_layers=num_layers,
        )

        # Coherence scoring head
        self.coherence_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.GELU(),
            nn.Linear(hidden_dim // 2, 1),
        )

    def forward(
        self,
        x: torch.Tensor,
        y: torch.Tensor,
        constraints: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Compute coherence energy.

        Args:
            x: Input sequence [batch, seq_len, hidden_dim]
            y: Output sequence [batch, seq_len, hidden_dim]
            constraints: Optional constraints

        Returns:
            Coherence energy (lower = more coherent)
        """
        # Concatenate input and output sequences
        combined = torch.cat([x, y], dim=1)

        # Apply coherence encoder
        encoded = self.coherence_encoder(combined)

        # Pool and score
        pooled = encoded.mean(dim=1)
        coherence_score = self.coherence_head(pooled).squeeze(-1)

        # Convert to energy (invert so lower = more coherent)
        energy = -coherence_score

        return energy

    def compute_gradient(
        self,
        x: torch.Tensor,
        y: torch.Tensor,
        constraints: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Compute gradient for coherence optimization."""
        y_grad = y.clone().requires_grad_(True)
        energy = self.forward(x, y_grad, constraints)
        energy_sum = energy.sum()
        energy_sum.backward()
        return y_grad.grad


class ConstraintEnergy(EnergyFunction):
    """
    Energy function that enforces explicit constraints.

    Maps constraint violations to energy penalties.
    """

    def __init__(
        self,
        hidden_dim: int,
        num_constraint_types: int = 10,
    ):
        super().__init__()

        self.hidden_dim = hidden_dim

        # Constraint type embeddings
        self.constraint_embeddings = nn.Embedding(num_constraint_types, hidden_dim)

        # Constraint satisfaction network
        self.satisfaction_net = nn.Sequential(
            nn.Linear(hidden_dim * 3, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.GELU(),
            nn.Linear(hidden_dim // 2, 1),
            nn.Softplus(),  # Ensure non-negative violation penalty
        )

    def forward(
        self,
        x: torch.Tensor,
        y: torch.Tensor,
        constraints: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Compute constraint violation energy.

        Args:
            x: Input embeddings [batch, hidden_dim]
            y: Output embeddings [batch, hidden_dim]
            constraints: Constraint indices [batch, num_constraints]

        Returns:
            Constraint violation energy
        """
        if constraints is None:
            return torch.zeros(x.shape[0], device=x.device)

        batch_size = x.shape[0]
        num_constraints = constraints.shape[1]

        # Get constraint embeddings
        constraint_embeds = self.constraint_embeddings(constraints)  # [batch, num_constraints, hidden]

        # Expand x and y for broadcasting
        x_expanded = x.unsqueeze(1).expand(-1, num_constraints, -1)
        y_expanded = y.unsqueeze(1).expand(-1, num_constraints, -1)

        # Concatenate and compute violations
        combined = torch.cat([x_expanded, y_expanded, constraint_embeds], dim=-1)
        violations = self.satisfaction_net(combined).squeeze(-1)  # [batch, num_constraints]

        # Sum violations as total constraint energy
        total_energy = violations.sum(dim=-1)

        return total_energy

    def compute_gradient(
        self,
        x: torch.Tensor,
        y: torch.Tensor,
        constraints: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Compute gradient for constraint satisfaction."""
        y_grad = y.clone().requires_grad_(True)
        energy = self.forward(x, y_grad, constraints)
        energy_sum = energy.sum()
        energy_sum.backward()
        return y_grad.grad


class CompositeEnergy(EnergyFunction):
    """
    Combines multiple energy functions with learnable weights.
    """

    def __init__(
        self,
        energy_functions: list[EnergyFunction],
        initial_weights: Optional[list[float]] = None,
    ):
        super().__init__()

        self.energy_functions = nn.ModuleList(energy_functions)

        if initial_weights is None:
            initial_weights = [1.0] * len(energy_functions)

        # Learnable weights (log-space for positivity)
        self.log_weights = nn.Parameter(
            torch.tensor([math.log(w) for w in initial_weights])
        )

    def forward(
        self,
        x: torch.Tensor,
        y: torch.Tensor,
        constraints: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Compute weighted sum of energies."""
        weights = torch.exp(self.log_weights)

        total_energy = torch.zeros(x.shape[0], device=x.device)
        for i, energy_fn in enumerate(self.energy_functions):
            energy = energy_fn(x, y, constraints)
            total_energy = total_energy + weights[i] * energy

        return total_energy

    def compute_gradient(
        self,
        x: torch.Tensor,
        y: torch.Tensor,
        constraints: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Compute weighted gradient."""
        y_grad = y.clone().requires_grad_(True)
        energy = self.forward(x, y_grad, constraints)
        energy_sum = energy.sum()
        energy_sum.backward()
        return y_grad.grad

    def get_energy_breakdown(
        self,
        x: torch.Tensor,
        y: torch.Tensor,
        constraints: Optional[torch.Tensor] = None,
    ) -> dict[str, torch.Tensor]:
        """Get individual energy components for analysis."""
        weights = torch.exp(self.log_weights)
        breakdown = {}

        for i, energy_fn in enumerate(self.energy_functions):
            name = energy_fn.__class__.__name__
            raw_energy = energy_fn(x, y, constraints)
            weighted_energy = weights[i] * raw_energy
            breakdown[f"{name}_raw"] = raw_energy
            breakdown[f"{name}_weighted"] = weighted_energy

        return breakdown
