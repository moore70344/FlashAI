"""
Energy-Based Reasoning Model

Core implementation of the EBM that performs reasoning through
energy minimization. Inspired by approaches like Kona from Logical
Intelligence, this model learns by recognizing and correcting mistakes
rather than just predicting the most likely output.

Key concepts:
- Reasoning as optimization: Find answers that minimize energy
- Energy landscape: Maps input-output pairs to scalar energy values
- Contrastive learning: Learn from both correct and incorrect examples
- Constraint satisfaction: Enforce logical constraints through energy penalties
"""

from __future__ import annotations

import asyncio
import logging
from pathlib import Path
from typing import Any, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR

from flashai.core.config import EBMConfig
from flashai.models.energy_functions import (
    ContrastiveEnergy,
    CoherenceEnergy,
    ConstraintEnergy,
    CompositeEnergy,
)
from flashai.models.reasoning_chain import (
    ReasoningChain,
    ReasoningChainBuilder,
    ReasoningStep,
    ReasoningStepType,
)


logger = logging.getLogger(__name__)


class TextEncoder(nn.Module):
    """Encodes text into embeddings for the EBM."""

    def __init__(self, vocab_size: int, hidden_dim: int, num_layers: int = 4):
        super().__init__()

        self.embedding = nn.Embedding(vocab_size, hidden_dim)
        self.position_embedding = nn.Embedding(1024, hidden_dim)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=8,
            dim_feedforward=hidden_dim * 4,
            dropout=0.1,
            activation="gelu",
            batch_first=True,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        self.layer_norm = nn.LayerNorm(hidden_dim)

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        """Encode tokens to embeddings."""
        batch_size, seq_len = tokens.shape
        positions = torch.arange(seq_len, device=tokens.device)

        x = self.embedding(tokens) + self.position_embedding(positions)
        x = self.encoder(x)
        x = self.layer_norm(x)

        # Mean pooling
        return x.mean(dim=1)


class SimpleTokenizer:
    """Simple character-level tokenizer for demonstration."""

    def __init__(self, vocab_size: int = 50000):
        self.vocab_size = vocab_size
        self.char_to_idx = {}
        self.idx_to_char = {}
        self._build_vocab()

    def _build_vocab(self):
        """Build basic vocabulary."""
        # Special tokens
        self.pad_token = 0
        self.unk_token = 1
        self.bos_token = 2
        self.eos_token = 3

        # ASCII printable characters
        for i, char in enumerate(
            "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789 .,!?-_:;'\"()[]{}+=<>/@#$%^&*\n\t"
        ):
            self.char_to_idx[char] = i + 4
            self.idx_to_char[i + 4] = char

    def encode(self, text: str, max_length: int = 512) -> torch.Tensor:
        """Encode text to token IDs."""
        tokens = [self.bos_token]
        for char in text[:max_length - 2]:
            tokens.append(self.char_to_idx.get(char, self.unk_token))
        tokens.append(self.eos_token)

        # Pad to max_length
        while len(tokens) < max_length:
            tokens.append(self.pad_token)

        return torch.tensor(tokens[:max_length])

    def decode(self, tokens: torch.Tensor) -> str:
        """Decode token IDs to text."""
        chars = []
        for token in tokens.tolist():
            if token in [self.pad_token, self.bos_token, self.eos_token]:
                continue
            chars.append(self.idx_to_char.get(token, "?"))
        return "".join(chars)


class EnergyBasedReasoningModel(nn.Module):
    """
    Energy-Based Reasoning Model.

    This model performs reasoning through energy minimization:
    1. Encode the query and context
    2. Generate candidate answers/reasoning steps
    3. Compute energy for each candidate
    4. Refine candidates by gradient descent on energy
    5. Select the lowest energy solution

    The model learns by:
    - Minimizing energy for correct solutions
    - Maximizing energy for incorrect solutions (contrastive)
    - Satisfying explicit constraints through energy penalties
    """

    def __init__(
        self,
        config: EBMConfig,
        device: str = "cpu",
        vocab_size: int = 50000,
    ):
        super().__init__()

        self.config = config
        self.device = device
        self.vocab_size = vocab_size

        # Text encoder
        self.text_encoder = TextEncoder(
            vocab_size=vocab_size,
            hidden_dim=config.input_dim,
        )

        # Tokenizer
        self.tokenizer = SimpleTokenizer(vocab_size)

        # Energy functions
        self.contrastive_energy = ContrastiveEnergy(
            input_dim=config.input_dim,
            output_dim=config.input_dim,
            hidden_dims=config.hidden_dims,
        )

        self.coherence_energy = CoherenceEnergy(
            hidden_dim=config.input_dim,
        )

        self.constraint_energy = ConstraintEnergy(
            hidden_dim=config.input_dim,
        )

        # Composite energy function
        self.energy_function = CompositeEnergy(
            energy_functions=[
                self.contrastive_energy,
                self.coherence_energy,
                self.constraint_energy,
            ],
            initial_weights=[1.0, 0.5, config.constraint_weight],
        )

        # Reasoning chain builder
        self.chain_builder = ReasoningChainBuilder(
            hidden_dim=config.input_dim,
            vocab_size=vocab_size,
            max_steps=config.max_reasoning_depth,
        )

        # Answer generator
        self.answer_generator = nn.Sequential(
            nn.Linear(config.input_dim * 2, config.input_dim),
            nn.GELU(),
            nn.Linear(config.input_dim, config.input_dim),
        )

        # Move to device
        self.to(device)

        # Training components (initialized on first training call)
        self._optimizer = None
        self._scheduler = None

        # User preference adaptation
        self._preference_bias = nn.Parameter(torch.zeros(config.input_dim))

        logger.info(f"EBM initialized on device: {device}")

    def encode_text(self, text: str) -> torch.Tensor:
        """Encode text to embedding."""
        tokens = self.tokenizer.encode(text).unsqueeze(0).to(self.device)
        return self.text_encoder(tokens)

    def compute_energy(
        self,
        query_embedding: torch.Tensor,
        answer_embedding: torch.Tensor,
        constraints: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Compute total energy for query-answer pair."""
        return self.energy_function(query_embedding, answer_embedding, constraints)

    def minimize_energy(
        self,
        query_embedding: torch.Tensor,
        initial_answer: torch.Tensor,
        constraints: Optional[torch.Tensor] = None,
        num_steps: Optional[int] = None,
        step_size: Optional[float] = None,
    ) -> tuple[torch.Tensor, list[float]]:
        """
        Minimize energy to refine answer.

        Uses gradient descent on the energy function to find
        lower-energy (better) answers.

        Args:
            query_embedding: Encoded query
            initial_answer: Initial answer embedding to refine
            constraints: Optional constraint indices
            num_steps: Number of optimization steps
            step_size: Learning rate for optimization

        Returns:
            Tuple of (refined_answer, energy_history)
        """
        num_steps = num_steps or self.config.inference_steps
        step_size = step_size or self.config.step_size

        answer = initial_answer.clone().detach().requires_grad_(True)
        energy_history = []

        for _ in range(num_steps):
            # Compute energy
            energy = self.compute_energy(query_embedding, answer, constraints)
            energy_history.append(energy.item())

            # Compute gradient
            energy.backward()

            # Update answer (gradient descent)
            with torch.no_grad():
                answer = answer - step_size * answer.grad
                answer = answer.detach().requires_grad_(True)

        return answer.detach(), energy_history

    async def reason(
        self,
        query: str,
        context: Optional[dict[str, Any]] = None,
        constraints: Optional[list[str]] = None,
    ) -> dict[str, Any]:
        """
        Perform energy-based reasoning on a query.

        Args:
            query: The question or problem to reason about
            context: Optional context information
            constraints: Optional list of constraints to satisfy

        Returns:
            Reasoning result with chain, energy scores, and answer
        """
        # Encode query
        query_embedding = self.encode_text(query)

        # Apply user preference bias
        query_embedding = query_embedding + self._preference_bias

        # Initialize reasoning chain
        chain = ReasoningChain(query=query)

        # Build reasoning chain
        self.eval()
        with torch.no_grad():
            chain_steps = self.chain_builder.build_chain(
                query_embedding.unsqueeze(1),
                energy_threshold=self.config.coherence_threshold,
            )

        # Process each step
        for step_embed, step_type, energy in chain_steps:
            chain.add_step(
                step_type=step_type,
                content=f"[{step_type.value}] Reasoning step",
                embedding=step_embed,
                energy=energy,
                confidence=1.0 - min(energy, 1.0),
            )

        # Generate initial answer from reasoning chain
        if chain.steps:
            chain_embedding = chain.get_step_embeddings()
            if chain_embedding is not None:
                pooled_chain = chain_embedding.mean(dim=0, keepdim=True)
            else:
                pooled_chain = query_embedding
        else:
            pooled_chain = query_embedding

        combined = torch.cat([query_embedding, pooled_chain], dim=-1)
        initial_answer = self.answer_generator(combined)

        # Minimize energy to refine answer
        refined_answer, energy_history = self.minimize_energy(
            query_embedding,
            initial_answer,
        )

        # Compute final energy
        final_energy = self.compute_energy(
            query_embedding,
            refined_answer,
        ).item()

        # Update chain
        chain.total_energy = final_energy
        chain.coherence_score = 1.0 - min(final_energy, 1.0)
        chain.is_complete = True
        chain.final_answer = f"Answer derived through {len(chain.steps)} reasoning steps"

        return {
            "query": query,
            "chain": chain.to_dict(),
            "final_energy": final_energy,
            "energy_history": energy_history,
            "confidence": chain.coherence_score,
            "answer_embedding": refined_answer.cpu().numpy().tolist(),
        }

    async def learn(
        self,
        data: dict[str, Any],
        feedback: Optional[dict[str, Any]] = None,
    ) -> dict[str, Any]:
        """
        Learn from data and feedback.

        Args:
            data: Training data with examples
            feedback: Optional feedback on previous predictions

        Returns:
            Learning result with metrics
        """
        self.train()

        # Initialize optimizer if needed
        if self._optimizer is None:
            self._optimizer = AdamW(
                self.parameters(),
                lr=self.config.learning_rate,
            )
            self._scheduler = CosineAnnealingLR(
                self._optimizer,
                T_max=1000,
            )

        total_loss = 0.0
        patterns_learned = []

        # Process training examples
        examples = data.get("examples", [])
        for example in examples:
            query = example.get("query", "")
            correct_answer = example.get("answer", "")
            negative_samples = example.get("negative_samples", [])

            # Encode
            query_emb = self.encode_text(query)
            correct_emb = self.encode_text(correct_answer)

            # Compute energy for correct answer (should be low)
            positive_energy = self.compute_energy(query_emb, correct_emb)

            # Compute energy for negative samples (should be high)
            negative_energies = []
            for neg in negative_samples:
                neg_emb = self.encode_text(neg)
                neg_energy = self.compute_energy(query_emb, neg_emb)
                negative_energies.append(neg_energy)

            # Contrastive loss
            if negative_energies:
                neg_energy_tensor = torch.stack(negative_energies)
                # Margin-based loss
                margin = self.config.energy_margin
                loss = F.relu(positive_energy - neg_energy_tensor + margin).mean()
            else:
                # Just minimize positive energy
                loss = positive_energy

            # Backprop
            self._optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.parameters(), 1.0)
            self._optimizer.step()
            self._scheduler.step()

            total_loss += loss.item()

        # Process feedback
        if feedback:
            correction = feedback.get("correction", {})
            if correction:
                patterns_learned.append({
                    "type": "feedback_correction",
                    "details": correction,
                })

        avg_loss = total_loss / max(len(examples), 1)

        return {
            "status": "learned",
            "examples_processed": len(examples),
            "average_loss": avg_loss,
            "patterns": patterns_learned,
            "learning_rate": self._scheduler.get_last_lr()[0] if self._scheduler else self.config.learning_rate,
        }

    def apply_preferences(self, preferences: dict[str, Any]) -> None:
        """Apply user preferences to modify model behavior."""
        # Update preference bias based on user patterns
        if "embedding_bias" in preferences:
            bias_data = preferences["embedding_bias"]
            if isinstance(bias_data, list):
                bias_tensor = torch.tensor(bias_data, device=self.device)
                if bias_tensor.shape == self._preference_bias.shape:
                    self._preference_bias.data = bias_tensor

        logger.info("Applied user preferences to model")

    def reset(self) -> None:
        """Reset model to initial state."""
        # Reinitialize all parameters
        for module in self.modules():
            if hasattr(module, "reset_parameters"):
                module.reset_parameters()

        # Reset preference bias
        self._preference_bias.data.zero_()

        # Reset optimizer
        self._optimizer = None
        self._scheduler = None

        logger.info("Model reset to initial state")

    def save(self, path: Path) -> None:
        """Save model state to file."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        state = {
            "model_state_dict": self.state_dict(),
            "config": self.config.model_dump(),
            "preference_bias": self._preference_bias.data.cpu().numpy().tolist(),
        }

        if self._optimizer is not None:
            state["optimizer_state_dict"] = self._optimizer.state_dict()

        torch.save(state, path)
        logger.info(f"Model saved to {path}")

    def load(self, path: Path) -> None:
        """Load model state from file."""
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"Model file not found: {path}")

        state = torch.load(path, map_location=self.device)

        self.load_state_dict(state["model_state_dict"])

        if "preference_bias" in state:
            bias_data = state["preference_bias"]
            self._preference_bias.data = torch.tensor(
                bias_data,
                device=self.device,
            )

        logger.info(f"Model loaded from {path}")

    def get_energy_breakdown(
        self,
        query: str,
        answer: str,
    ) -> dict[str, float]:
        """Get detailed energy breakdown for analysis."""
        query_emb = self.encode_text(query)
        answer_emb = self.encode_text(answer)

        breakdown = self.energy_function.get_energy_breakdown(
            query_emb,
            answer_emb,
        )

        return {k: v.item() for k, v in breakdown.items()}
