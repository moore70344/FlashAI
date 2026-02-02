"""
State Manager - Handles save, export, reset, and restore operations.

Provides comprehensive state management for the portable AI system:
- Create checkpoints during operation
- Export full state to zip files
- Reset to initial state
- Restore from previous exports
"""

from __future__ import annotations

import asyncio
import json
import logging
import shutil
import zipfile
from datetime import datetime
from pathlib import Path
from typing import Any, Optional
from dataclasses import dataclass, field, asdict


logger = logging.getLogger(__name__)


@dataclass
class Checkpoint:
    """A saved checkpoint of system state."""

    checkpoint_id: str
    name: str
    created_at: str
    components: list[str]
    metadata: dict[str, Any] = field(default_factory=dict)
    size_bytes: int = 0

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "Checkpoint":
        return cls(**data)


@dataclass
class ExportManifest:
    """Manifest file for exported state."""

    export_id: str
    export_name: str
    created_at: str
    flashai_version: str
    system_id: str
    includes_models: bool
    includes_user_data: bool
    includes_config: bool
    files: list[str] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "ExportManifest":
        return cls(**data)


class StateManager:
    """
    Manages system state including checkpoints, exports, and resets.

    Key operations:
    - create_checkpoint: Save current state for later restoration
    - export_state: Create a complete zip archive of system state
    - restore_from_export: Restore system from a zip archive
    - reset: Clear all state and return to initial configuration
    """

    def __init__(
        self,
        base_path: Path,
        checkpoints_path: Path,
        exports_path: Path,
    ):
        self.base_path = Path(base_path)
        self.checkpoints_path = Path(checkpoints_path)
        self.exports_path = Path(exports_path)

        # Ensure directories exist
        self.checkpoints_path.mkdir(parents=True, exist_ok=True)
        self.exports_path.mkdir(parents=True, exist_ok=True)

        # Checkpoint registry
        self._checkpoints: dict[str, Checkpoint] = {}
        self._load_checkpoint_registry()

        logger.info(f"StateManager initialized at {base_path}")

    def _load_checkpoint_registry(self) -> None:
        """Load checkpoint registry from disk."""
        registry_path = self.checkpoints_path / "registry.json"
        if registry_path.exists():
            with open(registry_path, "r") as f:
                data = json.load(f)
                self._checkpoints = {
                    k: Checkpoint.from_dict(v) for k, v in data.items()
                }

    def _save_checkpoint_registry(self) -> None:
        """Save checkpoint registry to disk."""
        registry_path = self.checkpoints_path / "registry.json"
        data = {k: v.to_dict() for k, v in self._checkpoints.items()}
        with open(registry_path, "w") as f:
            json.dump(data, f, indent=2)

    async def create_checkpoint(
        self,
        name: str,
        components: list[str],
        metadata: Optional[dict[str, Any]] = None,
    ) -> Checkpoint:
        """
        Create a checkpoint of specified components.

        Args:
            name: Human-readable checkpoint name
            components: List of components to checkpoint (e.g., ["ebm_model", "user_profiles"])
            metadata: Optional metadata to include

        Returns:
            Created checkpoint object
        """
        checkpoint_id = f"{name}_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}"
        checkpoint_dir = self.checkpoints_path / checkpoint_id
        checkpoint_dir.mkdir(parents=True, exist_ok=True)

        total_size = 0

        # Copy component files
        for component in components:
            component_path = self._get_component_path(component)
            if component_path and component_path.exists():
                dest_path = checkpoint_dir / component
                if component_path.is_dir():
                    shutil.copytree(component_path, dest_path)
                else:
                    shutil.copy2(component_path, dest_path)
                total_size += self._get_size(dest_path)

        # Create checkpoint record
        checkpoint = Checkpoint(
            checkpoint_id=checkpoint_id,
            name=name,
            created_at=datetime.utcnow().isoformat(),
            components=components,
            metadata=metadata or {},
            size_bytes=total_size,
        )

        # Save checkpoint info
        info_path = checkpoint_dir / "checkpoint.json"
        with open(info_path, "w") as f:
            json.dump(checkpoint.to_dict(), f, indent=2)

        # Update registry
        self._checkpoints[checkpoint_id] = checkpoint
        self._save_checkpoint_registry()

        logger.info(f"Created checkpoint: {checkpoint_id}")
        return checkpoint

    def _get_component_path(self, component: str) -> Optional[Path]:
        """Get the path for a component."""
        component_paths = {
            "ebm_model": self.base_path / "models" / "ebm_model.pt",
            "user_profiles": self.base_path / "data" / "user_profiles",
            "config": self.base_path / "config",
            "logs": self.base_path / "data" / "logs",
        }
        return component_paths.get(component)

    def _get_size(self, path: Path) -> int:
        """Get total size of a path (file or directory)."""
        if path.is_file():
            return path.stat().st_size
        total = 0
        for p in path.rglob("*"):
            if p.is_file():
                total += p.stat().st_size
        return total

    async def export_state(
        self,
        name: str,
        include_models: bool = True,
        include_user_data: bool = True,
        include_config: bool = True,
    ) -> dict[str, Any]:
        """
        Export complete system state to a zip file.

        Args:
            name: Name for the export
            include_models: Include trained model weights
            include_user_data: Include user profiles and data
            include_config: Include configuration files

        Returns:
            Export information including file path
        """
        export_id = f"{name}_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}"
        export_path = self.exports_path / f"{export_id}.zip"

        # Create manifest
        manifest = ExportManifest(
            export_id=export_id,
            export_name=name,
            created_at=datetime.utcnow().isoformat(),
            flashai_version="1.0.0",
            system_id=self._get_system_id(),
            includes_models=include_models,
            includes_user_data=include_user_data,
            includes_config=include_config,
        )

        files_added = []

        with zipfile.ZipFile(export_path, "w", zipfile.ZIP_DEFLATED) as zf:
            # Add models
            if include_models:
                models_dir = self.base_path / "models"
                if models_dir.exists():
                    for model_file in models_dir.rglob("*"):
                        if model_file.is_file():
                            arcname = f"models/{model_file.relative_to(models_dir)}"
                            zf.write(model_file, arcname)
                            files_added.append(arcname)

            # Add user data
            if include_user_data:
                user_data_dir = self.base_path / "data" / "user_profiles"
                if user_data_dir.exists():
                    for data_file in user_data_dir.rglob("*"):
                        if data_file.is_file():
                            arcname = f"user_profiles/{data_file.relative_to(user_data_dir)}"
                            zf.write(data_file, arcname)
                            files_added.append(arcname)

            # Add config
            if include_config:
                config_dir = self.base_path / "config"
                if config_dir.exists():
                    for config_file in config_dir.rglob("*"):
                        if config_file.is_file():
                            arcname = f"config/{config_file.relative_to(config_dir)}"
                            zf.write(config_file, arcname)
                            files_added.append(arcname)

            # Add manifest
            manifest.files = files_added
            manifest_json = json.dumps(manifest.to_dict(), indent=2)
            zf.writestr("manifest.json", manifest_json)

        export_size = export_path.stat().st_size

        logger.info(f"Exported state to: {export_path} ({export_size} bytes)")

        return {
            "export_id": export_id,
            "path": str(export_path),
            "size_bytes": export_size,
            "files_count": len(files_added),
            "manifest": manifest.to_dict(),
        }

    def _get_system_id(self) -> str:
        """Get system ID from config."""
        config_path = self.base_path / "config" / "flashai.yaml"
        if config_path.exists():
            import yaml
            with open(config_path, "r") as f:
                config = yaml.safe_load(f) or {}
                return config.get("system_id", "unknown")
        return "unknown"

    async def restore_from_export(
        self,
        export_path: Path | str,
        restore_models: bool = True,
        restore_user_data: bool = True,
        restore_config: bool = True,
    ) -> dict[str, Any]:
        """
        Restore system state from an export zip file.

        Args:
            export_path: Path to the export zip file
            restore_models: Restore model weights
            restore_user_data: Restore user profiles
            restore_config: Restore configuration

        Returns:
            Restoration result
        """
        export_path = Path(export_path)
        if not export_path.exists():
            raise FileNotFoundError(f"Export file not found: {export_path}")

        restored_files = []

        with zipfile.ZipFile(export_path, "r") as zf:
            # Read manifest
            manifest_data = json.loads(zf.read("manifest.json"))
            manifest = ExportManifest.from_dict(manifest_data)

            # Extract files based on options
            for file_info in zf.infolist():
                if file_info.filename == "manifest.json":
                    continue

                should_extract = False
                dest_path = None

                if file_info.filename.startswith("models/") and restore_models:
                    should_extract = True
                    relative = file_info.filename[len("models/"):]
                    dest_path = self.base_path / "models" / relative

                elif file_info.filename.startswith("user_profiles/") and restore_user_data:
                    should_extract = True
                    relative = file_info.filename[len("user_profiles/"):]
                    dest_path = self.base_path / "data" / "user_profiles" / relative

                elif file_info.filename.startswith("config/") and restore_config:
                    should_extract = True
                    relative = file_info.filename[len("config/"):]
                    dest_path = self.base_path / "config" / relative

                if should_extract and dest_path:
                    dest_path.parent.mkdir(parents=True, exist_ok=True)
                    with zf.open(file_info) as src, open(dest_path, "wb") as dst:
                        dst.write(src.read())
                    restored_files.append(str(dest_path))

        logger.info(f"Restored {len(restored_files)} files from export")

        return {
            "status": "restored",
            "export_id": manifest.export_id,
            "original_system_id": manifest.system_id,
            "files_restored": len(restored_files),
            "manifest": manifest.to_dict(),
        }

    async def get_checkpoint_count(self) -> int:
        """Get number of saved checkpoints."""
        return len(self._checkpoints)

    async def list_checkpoints(self) -> list[dict[str, Any]]:
        """List all available checkpoints."""
        return [cp.to_dict() for cp in self._checkpoints.values()]

    async def list_exports(self) -> list[dict[str, Any]]:
        """List all available exports."""
        exports = []
        for export_file in self.exports_path.glob("*.zip"):
            try:
                with zipfile.ZipFile(export_file, "r") as zf:
                    manifest_data = json.loads(zf.read("manifest.json"))
                    exports.append({
                        "path": str(export_file),
                        "size_bytes": export_file.stat().st_size,
                        "manifest": manifest_data,
                    })
            except Exception as e:
                logger.warning(f"Failed to read export {export_file}: {e}")
        return exports

    async def clear_checkpoints(self) -> None:
        """Clear all checkpoints."""
        for checkpoint_id in list(self._checkpoints.keys()):
            checkpoint_dir = self.checkpoints_path / checkpoint_id
            if checkpoint_dir.exists():
                shutil.rmtree(checkpoint_dir)

        self._checkpoints.clear()
        self._save_checkpoint_registry()
        logger.info("All checkpoints cleared")

    async def restore_checkpoint(self, checkpoint_id: str) -> dict[str, Any]:
        """Restore system from a checkpoint."""
        if checkpoint_id not in self._checkpoints:
            raise ValueError(f"Checkpoint not found: {checkpoint_id}")

        checkpoint = self._checkpoints[checkpoint_id]
        checkpoint_dir = self.checkpoints_path / checkpoint_id

        restored_components = []

        for component in checkpoint.components:
            src_path = checkpoint_dir / component
            dest_path = self._get_component_path(component)

            if src_path.exists() and dest_path:
                # Remove existing
                if dest_path.exists():
                    if dest_path.is_dir():
                        shutil.rmtree(dest_path)
                    else:
                        dest_path.unlink()

                # Copy from checkpoint
                dest_path.parent.mkdir(parents=True, exist_ok=True)
                if src_path.is_dir():
                    shutil.copytree(src_path, dest_path)
                else:
                    shutil.copy2(src_path, dest_path)

                restored_components.append(component)

        logger.info(f"Restored checkpoint: {checkpoint_id}")

        return {
            "status": "restored",
            "checkpoint_id": checkpoint_id,
            "components_restored": restored_components,
        }

    async def delete_checkpoint(self, checkpoint_id: str) -> bool:
        """Delete a specific checkpoint."""
        if checkpoint_id not in self._checkpoints:
            return False

        checkpoint_dir = self.checkpoints_path / checkpoint_id
        if checkpoint_dir.exists():
            shutil.rmtree(checkpoint_dir)

        del self._checkpoints[checkpoint_id]
        self._save_checkpoint_registry()

        logger.info(f"Deleted checkpoint: {checkpoint_id}")
        return True

    async def delete_export(self, export_path: Path | str) -> bool:
        """Delete an export file."""
        export_path = Path(export_path)
        if export_path.exists():
            export_path.unlink()
            logger.info(f"Deleted export: {export_path}")
            return True
        return False
