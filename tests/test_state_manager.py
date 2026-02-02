"""Tests for the State Manager."""

import pytest
import json
from pathlib import Path

from flashai.state.manager import StateManager, Checkpoint


class TestStateManager:
    """Tests for state management."""

    @pytest.fixture
    def state_manager(self, tmp_path):
        """Create test state manager."""
        base_path = tmp_path / "flashai"
        base_path.mkdir()
        checkpoints_path = tmp_path / "checkpoints"
        checkpoints_path.mkdir()
        exports_path = tmp_path / "exports"
        exports_path.mkdir()

        # Create test directories
        (base_path / "config").mkdir()
        (base_path / "data" / "user_profiles").mkdir(parents=True)
        (base_path / "models").mkdir()

        # Create test files
        (base_path / "config" / "flashai.yaml").write_text("test: true")
        (base_path / "models" / "test_model.pt").write_bytes(b"model data")

        return StateManager(
            base_path=base_path,
            checkpoints_path=checkpoints_path,
            exports_path=exports_path,
        )

    @pytest.mark.asyncio
    async def test_create_checkpoint(self, state_manager):
        """Test checkpoint creation."""
        checkpoint = await state_manager.create_checkpoint(
            name="test_checkpoint",
            components=["config"],
            metadata={"test": True},
        )

        assert checkpoint is not None
        assert checkpoint.name == "test_checkpoint"
        assert "config" in checkpoint.components
        assert checkpoint.metadata.get("test") is True

    @pytest.mark.asyncio
    async def test_list_checkpoints(self, state_manager):
        """Test listing checkpoints."""
        await state_manager.create_checkpoint(
            name="cp1",
            components=["config"],
        )
        await state_manager.create_checkpoint(
            name="cp2",
            components=["config"],
        )

        checkpoints = await state_manager.list_checkpoints()

        assert len(checkpoints) == 2

    @pytest.mark.asyncio
    async def test_export_state(self, state_manager):
        """Test state export."""
        result = await state_manager.export_state(
            name="test_export",
            include_models=True,
            include_user_data=True,
            include_config=True,
        )

        assert "export_id" in result
        assert "path" in result
        assert Path(result["path"]).exists()
        assert result["path"].endswith(".zip")

    @pytest.mark.asyncio
    async def test_restore_from_export(self, state_manager, tmp_path):
        """Test state restoration."""
        # Create export
        export_result = await state_manager.export_state(
            name="restore_test",
            include_config=True,
        )

        # Clear config
        config_path = state_manager.base_path / "config" / "flashai.yaml"
        config_path.unlink()
        assert not config_path.exists()

        # Restore
        restore_result = await state_manager.restore_from_export(
            export_result["path"],
            restore_config=True,
        )

        assert restore_result["status"] == "restored"
        assert config_path.exists()

    @pytest.mark.asyncio
    async def test_clear_checkpoints(self, state_manager):
        """Test checkpoint clearing."""
        await state_manager.create_checkpoint(
            name="to_clear",
            components=["config"],
        )

        count_before = await state_manager.get_checkpoint_count()
        assert count_before > 0

        await state_manager.clear_checkpoints()

        count_after = await state_manager.get_checkpoint_count()
        assert count_after == 0

    @pytest.mark.asyncio
    async def test_delete_checkpoint(self, state_manager):
        """Test individual checkpoint deletion."""
        checkpoint = await state_manager.create_checkpoint(
            name="to_delete",
            components=["config"],
        )

        deleted = await state_manager.delete_checkpoint(checkpoint.checkpoint_id)
        assert deleted is True

        checkpoints = await state_manager.list_checkpoints()
        assert all(c["checkpoint_id"] != checkpoint.checkpoint_id for c in checkpoints)

    @pytest.mark.asyncio
    async def test_list_exports(self, state_manager):
        """Test listing exports."""
        await state_manager.export_state(name="export1")
        await state_manager.export_state(name="export2")

        exports = await state_manager.list_exports()

        assert len(exports) == 2
