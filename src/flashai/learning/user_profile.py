"""
User Profile Management and Learning System.

Manages user profiles for personalized AI adaptation. The system:
- Creates profiles for new users
- Tracks interaction history
- Learns preferences from behavior
- Encrypts sensitive data
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Any, Optional
from dataclasses import dataclass, field, asdict
import base64

from cryptography.fernet import Fernet
from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2HMAC

from flashai.core.config import UserAdaptationConfig


logger = logging.getLogger(__name__)


@dataclass
class Interaction:
    """A single user interaction record."""

    timestamp: str
    interaction_type: str
    query: str
    result_summary: str
    feedback: Optional[str] = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "Interaction":
        return cls(**data)


@dataclass
class UserProfile:
    """
    User profile containing preferences and interaction history.

    Attributes:
        user_id: Unique identifier for the user
        created_at: Profile creation timestamp
        last_active: Last activity timestamp
        interaction_count: Total number of interactions
        interactions: Recent interaction history
        learned_preferences: Learned preference embeddings and patterns
        settings: User-specific settings
        metadata: Additional profile metadata
    """

    user_id: str
    created_at: str = field(default_factory=lambda: datetime.utcnow().isoformat())
    last_active: str = field(default_factory=lambda: datetime.utcnow().isoformat())
    interaction_count: int = 0
    interactions: list[Interaction] = field(default_factory=list)
    learned_preferences: dict[str, Any] = field(default_factory=dict)
    settings: dict[str, Any] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)

    def add_interaction(self, interaction: Interaction, max_history: int = 1000) -> None:
        """Add an interaction to history, maintaining max size."""
        self.interactions.append(interaction)
        self.interaction_count += 1
        self.last_active = datetime.utcnow().isoformat()

        # Trim old interactions
        if len(self.interactions) > max_history:
            self.interactions = self.interactions[-max_history:]

    def to_summary(self) -> dict[str, Any]:
        """Get a summary of the profile (without full history)."""
        return {
            "user_id": self.user_id,
            "created_at": self.created_at,
            "last_active": self.last_active,
            "interaction_count": self.interaction_count,
            "recent_interactions": len(self.interactions),
            "has_preferences": bool(self.learned_preferences),
            "settings": self.settings,
        }

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "user_id": self.user_id,
            "created_at": self.created_at,
            "last_active": self.last_active,
            "interaction_count": self.interaction_count,
            "interactions": [i.to_dict() for i in self.interactions],
            "learned_preferences": self.learned_preferences,
            "settings": self.settings,
            "metadata": self.metadata,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "UserProfile":
        """Create from dictionary."""
        interactions = [
            Interaction.from_dict(i) for i in data.get("interactions", [])
        ]
        return cls(
            user_id=data["user_id"],
            created_at=data.get("created_at", datetime.utcnow().isoformat()),
            last_active=data.get("last_active", datetime.utcnow().isoformat()),
            interaction_count=data.get("interaction_count", 0),
            interactions=interactions,
            learned_preferences=data.get("learned_preferences", {}),
            settings=data.get("settings", {}),
            metadata=data.get("metadata", {}),
        )


class ProfileEncryption:
    """Handles encryption/decryption of user profile data."""

    def __init__(self, password: Optional[str] = None):
        """
        Initialize encryption with optional password.

        If no password provided, generates a device-specific key.
        """
        if password is None:
            password = self._get_device_key()

        self.fernet = self._create_fernet(password)

    def _get_device_key(self) -> str:
        """Generate a device-specific encryption key."""
        import platform
        device_info = f"{platform.node()}-{platform.machine()}-flashai-secret"
        return hashlib.sha256(device_info.encode()).hexdigest()

    def _create_fernet(self, password: str) -> Fernet:
        """Create Fernet cipher from password."""
        # Use PBKDF2 to derive key from password
        salt = b"flashai_salt_v1"  # Fixed salt for reproducibility
        kdf = PBKDF2HMAC(
            algorithm=hashes.SHA256(),
            length=32,
            salt=salt,
            iterations=100000,
        )
        key = base64.urlsafe_b64encode(kdf.derive(password.encode()))
        return Fernet(key)

    def encrypt(self, data: str) -> str:
        """Encrypt string data."""
        return self.fernet.encrypt(data.encode()).decode()

    def decrypt(self, encrypted_data: str) -> str:
        """Decrypt string data."""
        return self.fernet.decrypt(encrypted_data.encode()).decode()


class UserProfileManager:
    """
    Manages user profiles with persistence and encryption.

    Handles:
    - Profile creation and retrieval
    - Interaction tracking
    - Preference learning
    - Secure storage
    """

    def __init__(
        self,
        profiles_path: Path,
        config: UserAdaptationConfig,
        encryption_password: Optional[str] = None,
    ):
        self.profiles_path = Path(profiles_path)
        self.config = config
        self.profiles_path.mkdir(parents=True, exist_ok=True)

        # Initialize encryption if enabled
        self._encryption: Optional[ProfileEncryption] = None
        if config.encrypt_profile:
            self._encryption = ProfileEncryption(encryption_password)

        # In-memory cache
        self._profile_cache: dict[str, UserProfile] = {}

        logger.info(f"UserProfileManager initialized at {profiles_path}")

    def _get_profile_path(self, user_id: str) -> Path:
        """Get file path for a user profile."""
        # Hash user ID for filename
        filename = hashlib.sha256(user_id.encode()).hexdigest()[:16]
        return self.profiles_path / f"{filename}.profile"

    async def get_or_create_profile(self, user_id: str) -> UserProfile:
        """Get existing profile or create new one."""
        profile = await self.get_profile(user_id)
        if profile is None:
            profile = await self.create_profile(user_id)
        return profile

    async def create_profile(self, user_id: str) -> UserProfile:
        """Create a new user profile."""
        profile = UserProfile(user_id=user_id)
        await self.save_profile(profile)
        self._profile_cache[user_id] = profile
        logger.info(f"Created new profile for user: {user_id[:8]}...")
        return profile

    async def get_profile(self, user_id: str) -> Optional[UserProfile]:
        """Get a user profile by ID."""
        # Check cache first
        if user_id in self._profile_cache:
            return self._profile_cache[user_id]

        # Load from disk
        profile_path = self._get_profile_path(user_id)
        if not profile_path.exists():
            return None

        try:
            with open(profile_path, "r") as f:
                data = f.read()

            # Decrypt if encryption is enabled
            if self._encryption:
                data = self._encryption.decrypt(data)

            profile_dict = json.loads(data)
            profile = UserProfile.from_dict(profile_dict)

            # Cache it
            self._profile_cache[user_id] = profile
            return profile

        except Exception as e:
            logger.error(f"Failed to load profile: {e}")
            return None

    async def save_profile(self, profile: UserProfile) -> None:
        """Save a user profile to disk."""
        profile_path = self._get_profile_path(profile.user_id)

        data = json.dumps(profile.to_dict(), indent=2)

        # Encrypt if enabled
        if self._encryption:
            data = self._encryption.encrypt(data)

        with open(profile_path, "w") as f:
            f.write(data)

        # Update cache
        self._profile_cache[profile.user_id] = profile

    async def record_interaction(
        self,
        user_id: str,
        interaction_type: str,
        query: str,
        result: dict[str, Any],
        feedback: Optional[str] = None,
    ) -> None:
        """Record a user interaction."""
        profile = await self.get_profile(user_id)
        if profile is None:
            logger.warning(f"Profile not found for user: {user_id}")
            return

        # Create interaction record
        interaction = Interaction(
            timestamp=datetime.utcnow().isoformat(),
            interaction_type=interaction_type,
            query=query if not self.config.anonymize_logs else self._anonymize(query),
            result_summary=self._summarize_result(result),
            feedback=feedback,
            metadata={
                "energy": result.get("final_energy"),
                "confidence": result.get("confidence"),
            },
        )

        profile.add_interaction(
            interaction,
            max_history=self.config.max_interaction_history,
        )

        # Update learned patterns based on interaction
        await self._update_preferences_from_interaction(profile, interaction)

        # Save updated profile
        await self.save_profile(profile)

    def _anonymize(self, text: str) -> str:
        """Anonymize text by removing potentially identifying info."""
        # Simple anonymization - in production, use more sophisticated methods
        return f"[query_hash:{hashlib.md5(text.encode()).hexdigest()[:8]}]"

    def _summarize_result(self, result: dict[str, Any]) -> str:
        """Create a summary of a result for logging."""
        return f"energy={result.get('final_energy', 'N/A')}, confidence={result.get('confidence', 'N/A')}"

    async def _update_preferences_from_interaction(
        self,
        profile: UserProfile,
        interaction: Interaction,
    ) -> None:
        """Update learned preferences based on interaction."""
        # Track interaction patterns
        if "interaction_patterns" not in profile.learned_preferences:
            profile.learned_preferences["interaction_patterns"] = {}

        patterns = profile.learned_preferences["interaction_patterns"]

        # Count interaction types
        int_type = interaction.interaction_type
        patterns[int_type] = patterns.get(int_type, 0) + 1

        # Track average energy (quality indicator)
        if interaction.metadata.get("energy") is not None:
            if "avg_energy" not in profile.learned_preferences:
                profile.learned_preferences["avg_energy"] = interaction.metadata["energy"]
            else:
                # Exponential moving average
                alpha = self.config.adaptation_rate
                profile.learned_preferences["avg_energy"] = (
                    alpha * interaction.metadata["energy"]
                    + (1 - alpha) * profile.learned_preferences["avg_energy"]
                )

    async def update_learned_patterns(
        self,
        user_id: str,
        patterns: dict[str, Any],
    ) -> None:
        """Update learned patterns for a user."""
        profile = await self.get_profile(user_id)
        if profile is None:
            return

        # Merge patterns with decay for old values
        for key, value in patterns.items():
            if key in profile.learned_preferences:
                # Apply decay to old value and add new
                old_value = profile.learned_preferences[key]
                if isinstance(old_value, (int, float)) and isinstance(value, (int, float)):
                    profile.learned_preferences[key] = (
                        self.config.preference_decay * old_value + value
                    )
                else:
                    profile.learned_preferences[key] = value
            else:
                profile.learned_preferences[key] = value

        await self.save_profile(profile)

    async def get_user_count(self) -> int:
        """Get total number of user profiles."""
        return len(list(self.profiles_path.glob("*.profile")))

    async def clear_all_profiles(self) -> None:
        """Clear all user profiles (used during reset)."""
        for profile_path in self.profiles_path.glob("*.profile"):
            profile_path.unlink()

        self._profile_cache.clear()
        logger.info("All user profiles cleared")

    async def export_profile(self, user_id: str) -> Optional[dict[str, Any]]:
        """Export a user profile (decrypted) for backup."""
        profile = await self.get_profile(user_id)
        if profile is None:
            return None
        return profile.to_dict()

    async def import_profile(self, profile_data: dict[str, Any]) -> UserProfile:
        """Import a user profile from backup data."""
        profile = UserProfile.from_dict(profile_data)
        await self.save_profile(profile)
        return profile
