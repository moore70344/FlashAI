"""Tests for User Profile Management."""

import pytest

from flashai.core.config import UserAdaptationConfig
from flashai.learning.user_profile import (
    UserProfileManager,
    UserProfile,
    Interaction,
    ProfileEncryption,
)


class TestUserProfile:
    """Tests for UserProfile class."""

    def test_profile_creation(self):
        """Test creating a user profile."""
        profile = UserProfile(user_id="test-user-123")

        assert profile.user_id == "test-user-123"
        assert profile.interaction_count == 0
        assert len(profile.interactions) == 0

    def test_add_interaction(self):
        """Test adding interactions."""
        profile = UserProfile(user_id="test-user")

        interaction = Interaction(
            timestamp="2024-01-01T00:00:00",
            interaction_type="reasoning",
            query="Test query",
            result_summary="energy=0.5",
        )

        profile.add_interaction(interaction, max_history=100)

        assert profile.interaction_count == 1
        assert len(profile.interactions) == 1
        assert profile.interactions[0].query == "Test query"

    def test_interaction_history_limit(self):
        """Test that interaction history is limited."""
        profile = UserProfile(user_id="test-user")

        for i in range(20):
            interaction = Interaction(
                timestamp=f"2024-01-{i:02d}T00:00:00",
                interaction_type="reasoning",
                query=f"Query {i}",
                result_summary="result",
            )
            profile.add_interaction(interaction, max_history=10)

        assert len(profile.interactions) == 10
        assert profile.interaction_count == 20
        # Should keep most recent
        assert profile.interactions[-1].query == "Query 19"

    def test_profile_serialization(self):
        """Test profile to/from dict."""
        profile = UserProfile(user_id="test-user")
        profile.learned_preferences = {"theme": "dark"}

        data = profile.to_dict()
        restored = UserProfile.from_dict(data)

        assert restored.user_id == profile.user_id
        assert restored.learned_preferences == profile.learned_preferences

    def test_profile_summary(self):
        """Test profile summary."""
        profile = UserProfile(user_id="test-user")
        profile.interaction_count = 50

        summary = profile.to_summary()

        assert "user_id" in summary
        assert summary["interaction_count"] == 50


class TestProfileEncryption:
    """Tests for profile encryption."""

    def test_encryption_roundtrip(self):
        """Test encrypting and decrypting data."""
        encryption = ProfileEncryption(password="test-password")

        original = "Hello, World!"
        encrypted = encryption.encrypt(original)
        decrypted = encryption.decrypt(encrypted)

        assert encrypted != original
        assert decrypted == original

    def test_encryption_deterministic_key(self):
        """Test that same password produces same encryption."""
        enc1 = ProfileEncryption(password="same-password")
        enc2 = ProfileEncryption(password="same-password")

        data = "Test data"
        # Both should be able to decrypt each other's data
        encrypted = enc1.encrypt(data)
        assert enc2.decrypt(encrypted) == data


class TestUserProfileManager:
    """Tests for UserProfileManager."""

    @pytest.fixture
    def config(self):
        """Create test config."""
        return UserAdaptationConfig(
            encrypt_profile=False,
            anonymize_logs=False,
            max_interaction_history=100,
        )

    @pytest.fixture
    def manager(self, tmp_path, config):
        """Create test manager."""
        profiles_path = tmp_path / "profiles"
        profiles_path.mkdir()
        return UserProfileManager(
            profiles_path=profiles_path,
            config=config,
        )

    @pytest.mark.asyncio
    async def test_create_profile(self, manager):
        """Test profile creation."""
        profile = await manager.create_profile("new-user")

        assert profile is not None
        assert profile.user_id == "new-user"
        assert profile.interaction_count == 0

    @pytest.mark.asyncio
    async def test_get_profile(self, manager):
        """Test getting profile."""
        await manager.create_profile("test-user")

        profile = await manager.get_profile("test-user")

        assert profile is not None
        assert profile.user_id == "test-user"

    @pytest.mark.asyncio
    async def test_get_nonexistent_profile(self, manager):
        """Test getting nonexistent profile."""
        profile = await manager.get_profile("nonexistent")
        assert profile is None

    @pytest.mark.asyncio
    async def test_get_or_create_profile(self, manager):
        """Test get_or_create_profile."""
        # First call creates
        profile1 = await manager.get_or_create_profile("user-1")
        assert profile1.interaction_count == 0

        # Second call gets existing
        profile2 = await manager.get_or_create_profile("user-1")
        assert profile2.user_id == profile1.user_id

    @pytest.mark.asyncio
    async def test_record_interaction(self, manager):
        """Test recording interaction."""
        await manager.create_profile("test-user")

        await manager.record_interaction(
            user_id="test-user",
            interaction_type="reasoning",
            query="Test query",
            result={"final_energy": 0.5, "confidence": 0.8},
        )

        profile = await manager.get_profile("test-user")
        assert profile.interaction_count == 1

    @pytest.mark.asyncio
    async def test_user_count(self, manager):
        """Test user count."""
        await manager.create_profile("user-1")
        await manager.create_profile("user-2")
        await manager.create_profile("user-3")

        count = await manager.get_user_count()
        assert count == 3

    @pytest.mark.asyncio
    async def test_clear_all_profiles(self, manager):
        """Test clearing all profiles."""
        await manager.create_profile("user-1")
        await manager.create_profile("user-2")

        await manager.clear_all_profiles()

        count = await manager.get_user_count()
        assert count == 0

    @pytest.mark.asyncio
    async def test_encrypted_storage(self, tmp_path):
        """Test encrypted profile storage."""
        config = UserAdaptationConfig(encrypt_profile=True)
        profiles_path = tmp_path / "encrypted"
        profiles_path.mkdir()

        manager = UserProfileManager(
            profiles_path=profiles_path,
            config=config,
        )

        await manager.create_profile("encrypted-user")

        # Profile file should exist but not be readable as plain JSON
        profile_files = list(profiles_path.glob("*.profile"))
        assert len(profile_files) == 1

        content = profile_files[0].read_text()
        # Should not be valid JSON (encrypted)
        try:
            import json
            json.loads(content)
            # If we get here, it parsed as JSON which means it's not encrypted
            pytest.fail("Profile should be encrypted")
        except json.JSONDecodeError:
            pass  # Expected - content is encrypted
