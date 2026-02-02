"""
FlashAI Engine - Main orchestrator for the portable AI system.

Coordinates the Energy-Based Model reasoning, user adaptation,
state management, and GitHub integration.
"""

from __future__ import annotations

import asyncio
import hashlib
import logging
import uuid
from datetime import datetime
from pathlib import Path
from typing import Any, Optional, Callable

from flashai.core.config import FlashAIConfig, load_config_from_env
from flashai.models.ebm import EnergyBasedReasoningModel
from flashai.learning.user_profile import UserProfileManager
from flashai.learning.file_parser import LearningDataParser
from flashai.state.manager import StateManager


logger = logging.getLogger(__name__)


class FlashAIEngine:
    """
    Main FlashAI Engine that orchestrates all components.

    This engine manages:
    - Energy-Based Model reasoning
    - User profile adaptation
    - State persistence and reset
    - GitHub integration hooks
    """

    def __init__(
        self,
        base_path: Path | str | None = None,
        config: FlashAIConfig | None = None,
    ):
        """
        Initialize the FlashAI Engine.

        Args:
            base_path: Base path for the FlashAI installation (flash drive root).
                      If None, uses current working directory.
            config: Configuration object. If None, loads from default location.
        """
        self.base_path = Path(base_path) if base_path else Path.cwd()
        self.config = config or self._load_config()

        # Generate or load system ID
        if not self.config.system_id:
            self.config.system_id = self._generate_system_id()

        # Initialize paths
        self.data_path = self.base_path / self.config.data_dir
        self.models_path = self.base_path / self.config.models_dir
        self.config_path = self.base_path / self.config.config_dir

        # Ensure directories exist
        self._ensure_directories()

        # Initialize components (lazy loading)
        self._ebm_model: Optional[EnergyBasedReasoningModel] = None
        self._user_manager: Optional[UserProfileManager] = None
        self._state_manager: Optional[StateManager] = None

        # Runtime state
        self._initialized = False
        self._current_user_id: Optional[str] = None
        self._hooks: dict[str, list[Callable]] = {}

        # Setup logging
        self._setup_logging()

        logger.info(f"FlashAI Engine created with base path: {self.base_path}")

    def _load_config(self) -> FlashAIConfig:
        """Load configuration from file or environment."""
        config_file = self.base_path / "config" / "flashai.yaml"
        if config_file.exists():
            return FlashAIConfig.load(config_file)
        return load_config_from_env()

    def _generate_system_id(self) -> str:
        """Generate a unique system identifier."""
        # Create ID based on creation time and random component
        timestamp = datetime.utcnow().isoformat()
        random_part = uuid.uuid4().hex[:8]
        raw_id = f"flashai-{timestamp}-{random_part}"
        return hashlib.sha256(raw_id.encode()).hexdigest()[:16]

    def _ensure_directories(self) -> None:
        """Ensure all required directories exist."""
        directories = [
            self.data_path,
            self.data_path / "user_profiles",
            self.data_path / "checkpoints",
            self.data_path / "exports",
            self.data_path / "logs",
            self.models_path,
            self.config_path,
        ]
        for directory in directories:
            directory.mkdir(parents=True, exist_ok=True)

    def _setup_logging(self) -> None:
        """Configure logging for the engine."""
        log_level = getattr(logging, self.config.log_level.upper(), logging.INFO)
        log_file = self.data_path / "logs" / "flashai.log"

        logging.basicConfig(
            level=log_level,
            format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
            handlers=[
                logging.FileHandler(log_file),
                logging.StreamHandler(),
            ],
        )

    @property
    def ebm_model(self) -> EnergyBasedReasoningModel:
        """Get or initialize the Energy-Based Model."""
        if self._ebm_model is None:
            self._ebm_model = EnergyBasedReasoningModel(
                config=self.config.ebm,
                device=self.config.resolve_device(),
            )
            # Try to load saved model weights
            model_path = self.models_path / "ebm_model.pt"
            if model_path.exists():
                self._ebm_model.load(model_path)
                logger.info("Loaded existing EBM model weights")
        return self._ebm_model

    @property
    def user_manager(self) -> UserProfileManager:
        """Get or initialize the User Profile Manager."""
        if self._user_manager is None:
            self._user_manager = UserProfileManager(
                profiles_path=self.data_path / "user_profiles",
                config=self.config.user_adaptation,
            )
        return self._user_manager

    @property
    def state_manager(self) -> StateManager:
        """Get or initialize the State Manager."""
        if self._state_manager is None:
            self._state_manager = StateManager(
                base_path=self.base_path,
                checkpoints_path=self.data_path / "checkpoints",
                exports_path=self.data_path / "exports",
            )
        return self._state_manager

    async def initialize(self, user_identifier: Optional[str] = None) -> dict[str, Any]:
        """
        Initialize the FlashAI system for a user session.

        For the first user, this creates a new profile and begins learning.
        For returning users, this loads their profile and preferences.

        Args:
            user_identifier: Optional identifier for the user. If None,
                           generates a machine-based identifier.

        Returns:
            Initialization status and user profile summary.
        """
        logger.info("Initializing FlashAI Engine...")

        # Determine user identity
        if user_identifier is None:
            user_identifier = self._get_machine_identifier()

        self._current_user_id = user_identifier

        # Check if this is a new or returning user
        profile = await self.user_manager.get_or_create_profile(user_identifier)
        is_new_user = profile.interaction_count == 0

        # Initialize EBM model
        _ = self.ebm_model  # Trigger lazy initialization

        # If returning user, apply their preferences to the model
        if not is_new_user and profile.learned_preferences:
            await self._apply_user_preferences(profile)

        self._initialized = True

        # Trigger initialization hooks
        await self._trigger_hooks("on_initialize", {
            "user_id": user_identifier,
            "is_new_user": is_new_user,
            "profile": profile.to_summary(),
        })

        logger.info(f"FlashAI initialized for user: {user_identifier} (new: {is_new_user})")

        return {
            "status": "initialized",
            "system_id": self.config.system_id,
            "user_id": user_identifier,
            "is_new_user": is_new_user,
            "profile_summary": profile.to_summary(),
            "capabilities": self._get_capabilities(),
        }

    def _get_machine_identifier(self) -> str:
        """Generate a machine-based user identifier."""
        import platform
        machine_info = f"{platform.node()}-{platform.machine()}-{platform.system()}"
        return hashlib.sha256(machine_info.encode()).hexdigest()[:16]

    async def _apply_user_preferences(self, profile) -> None:
        """Apply user preferences to the EBM model."""
        if profile.learned_preferences:
            self.ebm_model.apply_preferences(profile.learned_preferences)

    def _get_capabilities(self) -> list[str]:
        """Get list of available capabilities."""
        return [
            "reasoning",
            "learning",
            "adaptation",
            "state_management",
            "github_integration",
            "export",
            "reset",
        ]

    async def reason(
        self,
        query: str,
        context: Optional[dict[str, Any]] = None,
        constraints: Optional[list[str]] = None,
    ) -> dict[str, Any]:
        """
        Perform energy-based reasoning on a query.

        Args:
            query: The query or problem to reason about.
            context: Optional context information.
            constraints: Optional constraints that must be satisfied.

        Returns:
            Reasoning result with energy scores and explanation.
        """
        if not self._initialized:
            raise RuntimeError("Engine not initialized. Call initialize() first.")

        logger.debug(f"Reasoning on query: {query[:100]}...")

        # Prepare context with user preferences
        full_context = context or {}
        if self._current_user_id:
            profile = await self.user_manager.get_profile(self._current_user_id)
            if profile:
                full_context["user_preferences"] = profile.learned_preferences

        # Perform reasoning
        result = await self.ebm_model.reason(
            query=query,
            context=full_context,
            constraints=constraints,
        )

        # Learn from interaction
        if self._current_user_id:
            await self.user_manager.record_interaction(
                user_id=self._current_user_id,
                interaction_type="reasoning",
                query=query,
                result=result,
            )

        # Trigger reasoning hooks
        await self._trigger_hooks("on_reason", {
            "query": query,
            "result": result,
        })

        return result

    async def learn(
        self,
        data: dict[str, Any],
        feedback: Optional[dict[str, Any]] = None,
    ) -> dict[str, Any]:
        """
        Learn from provided data or feedback.

        Args:
            data: Data to learn from.
            feedback: Optional feedback on previous reasoning.

        Returns:
            Learning result with updated model info.
        """
        if not self._initialized:
            raise RuntimeError("Engine not initialized. Call initialize() first.")

        logger.info("Learning from new data...")

        # Update EBM model
        learning_result = await self.ebm_model.learn(data, feedback)

        # Update user profile with learned patterns
        if self._current_user_id:
            await self.user_manager.update_learned_patterns(
                user_id=self._current_user_id,
                patterns=learning_result.get("patterns", {}),
            )

        # Save model checkpoint
        await self.state_manager.create_checkpoint(
            name="auto_learn",
            components=["ebm_model"],
            metadata={"trigger": "learn", "data_size": len(str(data))},
        )

        # Trigger learning hooks
        await self._trigger_hooks("on_learn", {
            "data": data,
            "result": learning_result,
        })

        return learning_result

    async def learn_from_file(
        self,
        file_path: Path | str,
        feedback: Optional[dict[str, Any]] = None,
    ) -> dict[str, Any]:
        """
        Learn from a data file.

        Supports JSON, JSONL, CSV, YAML, Markdown, and text files.

        Args:
            file_path: Path to the learning data file.
            feedback: Optional feedback on previous learning.

        Returns:
            Learning result with parsing and training info.
        """
        if not self._initialized:
            raise RuntimeError("Engine not initialized. Call initialize() first.")

        logger.info(f"Learning from file: {file_path}")

        # Parse the file
        parser = LearningDataParser()
        parse_result = parser.parse_file(file_path)

        if not parse_result.success:
            return {
                "status": "parse_error",
                "errors": parse_result.errors,
                "format_detected": parse_result.format_detected.value,
            }

        # Convert to learning data format
        learning_data = parse_result.to_learning_data()

        # Perform learning
        learning_result = await self.learn(data=learning_data, feedback=feedback)

        return {
            "status": "learned",
            "file": str(file_path),
            "format": parse_result.format_detected.value,
            "examples_parsed": len(parse_result.examples),
            "parse_warnings": parse_result.warnings,
            "learning_result": learning_result,
        }

    async def learn_from_screen(
        self,
        image_data: bytes,
        context: Optional[dict[str, Any]] = None,
    ) -> dict[str, Any]:
        """
        Learn from a screen capture image.

        Extracts text and visual patterns from the image for learning.

        Args:
            image_data: Raw image bytes (PNG, JPEG, etc.)
            context: Optional context about the screen capture.

        Returns:
            Learning result.
        """
        if not self._initialized:
            raise RuntimeError("Engine not initialized. Call initialize() first.")

        logger.info("Learning from screen capture...")

        # Check if screen learning is enabled in user settings
        if self._current_user_id:
            profile = await self.user_manager.get_profile(self._current_user_id)
            if profile and not profile.settings.get("screen_learning", False):
                return {
                    "status": "disabled",
                    "message": "Screen learning is disabled in user settings.",
                }

        # Try to extract text from image using OCR (if available)
        extracted_text = await self._extract_text_from_image(image_data)

        if not extracted_text:
            return {
                "status": "no_content",
                "message": "Could not extract content from image.",
            }

        # Create learning data from extracted text
        learning_data = {
            "examples": [],
            "screen_content": extracted_text,
            "context": context or {},
            "source": "screen_capture",
        }

        # Record as note if enabled
        if self._current_user_id:
            await self.user_manager.create_note(
                user_id=self._current_user_id,
                content=extracted_text[:2000],
                title=f"Screen capture - {datetime.utcnow().strftime('%Y-%m-%d %H:%M')}",
                tags=["screen_capture", "auto_generated"],
            )

        # Update model with screen content
        learning_result = await self.ebm_model.learn(learning_data, None)

        return {
            "status": "learned",
            "extracted_text_length": len(extracted_text),
            "learning_result": learning_result,
        }

    async def _extract_text_from_image(self, image_data: bytes) -> Optional[str]:
        """Extract text from image using OCR."""
        try:
            # Try using pytesseract if available
            import pytesseract
            from PIL import Image
            import io

            image = Image.open(io.BytesIO(image_data))
            text = pytesseract.image_to_string(image)
            return text.strip() if text.strip() else None

        except ImportError:
            logger.warning("pytesseract not available for OCR")
            # Return placeholder - in production, could use cloud OCR
            return None
        except Exception as e:
            logger.error(f"OCR failed: {e}")
            return None

    async def save_and_reset(
        self,
        export_name: Optional[str] = None,
        include_user_data: bool = True,
    ) -> dict[str, Any]:
        """
        Save current state to a zip file and reset to initial state.

        Args:
            export_name: Optional name for the export file.
            include_user_data: Whether to include user profile data.

        Returns:
            Export information and reset confirmation.
        """
        logger.info("Saving state and resetting...")

        if export_name is None:
            timestamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
            export_name = f"flashai_export_{timestamp}"

        # Create export
        export_result = await self.state_manager.export_state(
            name=export_name,
            include_models=True,
            include_user_data=include_user_data,
            include_config=True,
        )

        # Trigger pre-reset hooks
        await self._trigger_hooks("on_pre_reset", {
            "export_path": export_result["path"],
        })

        # Reset components
        await self._reset_to_initial_state()

        # Trigger post-reset hooks
        await self._trigger_hooks("on_post_reset", {
            "export_path": export_result["path"],
        })

        logger.info(f"State exported to: {export_result['path']}")

        return {
            "status": "reset_complete",
            "export": export_result,
            "message": "System has been reset to initial state. Export saved.",
        }

    async def _reset_to_initial_state(self) -> None:
        """Reset all components to initial state."""
        # Reset EBM model
        if self._ebm_model:
            self._ebm_model.reset()
            self._ebm_model = None

        # Clear user profiles
        await self.user_manager.clear_all_profiles()

        # Reset state
        self._initialized = False
        self._current_user_id = None

        # Clear checkpoints
        await self.state_manager.clear_checkpoints()

        logger.info("System reset to initial state")

    async def restore_from_export(self, export_path: Path | str) -> dict[str, Any]:
        """
        Restore system state from an exported zip file.

        Args:
            export_path: Path to the export zip file.

        Returns:
            Restoration result.
        """
        logger.info(f"Restoring from export: {export_path}")

        result = await self.state_manager.restore_from_export(export_path)

        # Reinitialize components
        self._ebm_model = None
        self._user_manager = None

        return result

    def register_hook(self, event: str, callback: Callable) -> None:
        """
        Register a callback for an event.

        Events:
        - on_initialize: When engine is initialized
        - on_reason: After reasoning is performed
        - on_learn: After learning from data
        - on_pre_reset: Before resetting to initial state
        - on_post_reset: After resetting to initial state

        Args:
            event: Event name to hook into.
            callback: Async callable to invoke.
        """
        if event not in self._hooks:
            self._hooks[event] = []
        self._hooks[event].append(callback)
        logger.debug(f"Registered hook for event: {event}")

    async def _trigger_hooks(self, event: str, data: dict[str, Any]) -> None:
        """Trigger all hooks for an event."""
        if event not in self._hooks:
            return

        for callback in self._hooks[event]:
            try:
                if asyncio.iscoroutinefunction(callback):
                    await callback(data)
                else:
                    callback(data)
            except Exception as e:
                logger.error(f"Hook error for {event}: {e}")

    async def get_status(self) -> dict[str, Any]:
        """Get current system status."""
        return {
            "system_id": self.config.system_id,
            "initialized": self._initialized,
            "current_user": self._current_user_id,
            "base_path": str(self.base_path),
            "device": self.config.resolve_device(),
            "ebm_loaded": self._ebm_model is not None,
            "users_count": await self.user_manager.get_user_count(),
            "checkpoints_count": await self.state_manager.get_checkpoint_count(),
        }

    async def shutdown(self) -> None:
        """Gracefully shutdown the engine."""
        logger.info("Shutting down FlashAI Engine...")

        # Save current state
        if self._ebm_model:
            model_path = self.models_path / "ebm_model.pt"
            self._ebm_model.save(model_path)

        # Save config
        config_file = self.config_path / "flashai.yaml"
        self.config.save(config_file)

        self._initialized = False
        logger.info("FlashAI Engine shutdown complete")
