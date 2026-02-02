"""Tests for the Energy-Based Model."""

import pytest
import torch

from flashai.core.config import EBMConfig
from flashai.models.ebm import EnergyBasedReasoningModel
from flashai.models.energy_functions import ContrastiveEnergy, CoherenceEnergy


class TestEnergyFunctions:
    """Tests for energy function implementations."""

    def test_contrastive_energy_initialization(self):
        """Test ContrastiveEnergy can be initialized."""
        energy_fn = ContrastiveEnergy(
            input_dim=64,
            output_dim=64,
            hidden_dims=[32, 16],
        )
        assert energy_fn is not None
        assert energy_fn.input_dim == 64

    def test_contrastive_energy_forward(self):
        """Test ContrastiveEnergy forward pass."""
        energy_fn = ContrastiveEnergy(
            input_dim=64,
            output_dim=64,
            hidden_dims=[32, 16],
        )

        x = torch.randn(2, 64)
        y = torch.randn(2, 64)

        energy = energy_fn(x, y)

        assert energy.shape == (2,)
        assert not torch.isnan(energy).any()

    def test_contrastive_energy_gradient(self):
        """Test ContrastiveEnergy gradient computation."""
        energy_fn = ContrastiveEnergy(
            input_dim=64,
            output_dim=64,
            hidden_dims=[32, 16],
        )

        x = torch.randn(2, 64)
        y = torch.randn(2, 64)

        grad = energy_fn.compute_gradient(x, y)

        assert grad.shape == y.shape
        assert not torch.isnan(grad).any()

    def test_coherence_energy_initialization(self):
        """Test CoherenceEnergy can be initialized."""
        energy_fn = CoherenceEnergy(hidden_dim=64, num_heads=4)
        assert energy_fn is not None

    def test_coherence_energy_forward(self):
        """Test CoherenceEnergy forward pass."""
        energy_fn = CoherenceEnergy(hidden_dim=64, num_heads=4)

        x = torch.randn(2, 5, 64)  # [batch, seq_len, hidden]
        y = torch.randn(2, 5, 64)

        energy = energy_fn(x, y)

        assert energy.shape == (2,)
        assert not torch.isnan(energy).any()


class TestEnergyBasedReasoningModel:
    """Tests for the main EBM model."""

    @pytest.fixture
    def config(self):
        """Create test configuration."""
        return EBMConfig(
            input_dim=64,
            hidden_dims=[32, 16],
            energy_dim=8,
            inference_steps=3,
            max_reasoning_depth=2,
        )

    @pytest.fixture
    def model(self, config):
        """Create test model."""
        return EnergyBasedReasoningModel(
            config=config,
            device="cpu",
            vocab_size=1000,
        )

    def test_model_initialization(self, model):
        """Test model can be initialized."""
        assert model is not None
        assert model.device == "cpu"

    def test_text_encoding(self, model):
        """Test text encoding."""
        embedding = model.encode_text("Hello world")

        assert embedding.shape == (1, 64)
        assert not torch.isnan(embedding).any()

    def test_energy_computation(self, model):
        """Test energy computation."""
        query_emb = model.encode_text("What is 2+2?")
        answer_emb = model.encode_text("4")

        energy = model.compute_energy(query_emb, answer_emb)

        assert energy.shape == (1,)
        assert not torch.isnan(energy).any()

    def test_energy_minimization(self, model):
        """Test energy minimization."""
        query_emb = model.encode_text("Test query")
        initial_answer = torch.randn(1, 64)

        refined, history = model.minimize_energy(
            query_emb,
            initial_answer,
            num_steps=3,
        )

        assert refined.shape == initial_answer.shape
        assert len(history) == 3
        # Energy should generally decrease
        assert history[-1] <= history[0] + 0.5  # Allow some tolerance

    @pytest.mark.asyncio
    async def test_reason(self, model):
        """Test reasoning method."""
        result = await model.reason(
            query="What is the answer?",
            context={"domain": "test"},
        )

        assert "query" in result
        assert "chain" in result
        assert "final_energy" in result
        assert "confidence" in result

    @pytest.mark.asyncio
    async def test_learn(self, model):
        """Test learning method."""
        result = await model.learn(
            data={
                "examples": [
                    {
                        "query": "What is 1+1?",
                        "answer": "2",
                        "negative_samples": ["3", "11"],
                    }
                ]
            }
        )

        assert "status" in result
        assert result["status"] == "learned"
        assert "examples_processed" in result

    def test_model_save_load(self, model, tmp_path):
        """Test model save and load."""
        save_path = tmp_path / "model.pt"

        # Save
        model.save(save_path)
        assert save_path.exists()

        # Load
        model.load(save_path)  # Should not raise

    def test_model_reset(self, model):
        """Test model reset."""
        # Modify model state
        model._preference_bias.data.fill_(1.0)

        # Reset
        model.reset()

        # Check bias is zeroed
        assert model._preference_bias.data.abs().sum() == 0

    def test_energy_breakdown(self, model):
        """Test energy breakdown analysis."""
        breakdown = model.get_energy_breakdown(
            query="Test query",
            answer="Test answer",
        )

        assert isinstance(breakdown, dict)
        assert len(breakdown) > 0
