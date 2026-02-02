"""
Agent Memory System

Provides memory capabilities for agents:
- Short-term working memory
- Long-term persistent memory
- Episodic memory for past interactions
- Semantic memory for learned concepts
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field, asdict
from datetime import datetime
from pathlib import Path
from typing import Any, Optional
import hashlib


logger = logging.getLogger(__name__)


@dataclass
class MemoryEntry:
    """A single memory entry."""

    entry_id: str
    content: Any
    entry_type: str
    timestamp: str
    importance: float = 0.5
    metadata: dict[str, Any] = field(default_factory=dict)
    embedding: Optional[list[float]] = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "MemoryEntry":
        return cls(**data)


class AgentMemory:
    """
    Memory system for agents.

    Provides:
    - Working memory (short-term, limited capacity)
    - Episodic memory (past interactions)
    - Semantic memory (learned concepts)
    - Persistence to disk
    """

    def __init__(
        self,
        agent_id: str,
        storage_path: Optional[Path] = None,
        working_memory_size: int = 20,
        persist: bool = True,
    ):
        self.agent_id = agent_id
        self.storage_path = Path(storage_path) if storage_path else None
        self.working_memory_size = working_memory_size
        self.persist = persist

        # Memory stores
        self._working_memory: list[MemoryEntry] = []
        self._episodic_memory: list[MemoryEntry] = []
        self._semantic_memory: dict[str, MemoryEntry] = {}

        # Load from disk if available
        if self.persist and self.storage_path:
            self._load_from_disk()

        logger.info(f"AgentMemory initialized for agent: {agent_id}")

    def _generate_id(self, content: str) -> str:
        """Generate a unique ID for content."""
        timestamp = datetime.utcnow().isoformat()
        raw = f"{self.agent_id}:{timestamp}:{content[:100]}"
        return hashlib.md5(raw.encode()).hexdigest()[:12]

    async def add_message(
        self,
        message: Any,
        importance: float = 0.5,
        metadata: Optional[dict[str, Any]] = None,
    ) -> MemoryEntry:
        """Add a message to working memory."""
        entry = MemoryEntry(
            entry_id=self._generate_id(str(message)),
            content=message if isinstance(message, (str, dict)) else str(message),
            entry_type="message",
            timestamp=datetime.utcnow().isoformat(),
            importance=importance,
            metadata=metadata or {},
        )

        self._working_memory.append(entry)

        # Trim working memory if needed
        while len(self._working_memory) > self.working_memory_size:
            # Remove least important entry
            min_importance_idx = min(
                range(len(self._working_memory)),
                key=lambda i: self._working_memory[i].importance
            )
            removed = self._working_memory.pop(min_importance_idx)
            # Optionally move to episodic memory
            if removed.importance > 0.3:
                self._episodic_memory.append(removed)

        if self.persist:
            await self._save_to_disk()

        return entry

    async def add_to_episodic(
        self,
        content: Any,
        importance: float = 0.5,
        metadata: Optional[dict[str, Any]] = None,
    ) -> MemoryEntry:
        """Add an entry to episodic memory."""
        entry = MemoryEntry(
            entry_id=self._generate_id(str(content)),
            content=content if isinstance(content, (str, dict)) else str(content),
            entry_type="episodic",
            timestamp=datetime.utcnow().isoformat(),
            importance=importance,
            metadata=metadata or {},
        )

        self._episodic_memory.append(entry)

        if self.persist:
            await self._save_to_disk()

        return entry

    async def add_concept(
        self,
        key: str,
        content: Any,
        importance: float = 0.5,
        metadata: Optional[dict[str, Any]] = None,
    ) -> MemoryEntry:
        """Add or update a concept in semantic memory."""
        entry = MemoryEntry(
            entry_id=self._generate_id(key),
            content=content if isinstance(content, (str, dict)) else str(content),
            entry_type="semantic",
            timestamp=datetime.utcnow().isoformat(),
            importance=importance,
            metadata=metadata or {},
        )

        self._semantic_memory[key] = entry

        if self.persist:
            await self._save_to_disk()

        return entry

    def get_working_memory(self) -> list[MemoryEntry]:
        """Get all entries in working memory."""
        return list(self._working_memory)

    def get_recent_messages(self, n: int = 10) -> list[MemoryEntry]:
        """Get the n most recent messages."""
        return self._working_memory[-n:]

    def get_concept(self, key: str) -> Optional[MemoryEntry]:
        """Get a concept from semantic memory."""
        return self._semantic_memory.get(key)

    def search_episodic(
        self,
        query: str,
        limit: int = 10,
    ) -> list[MemoryEntry]:
        """Search episodic memory by content."""
        query_lower = query.lower()
        matches = []

        for entry in self._episodic_memory:
            content_str = str(entry.content).lower()
            if query_lower in content_str:
                matches.append(entry)

        # Sort by importance and recency
        matches.sort(
            key=lambda e: (e.importance, e.timestamp),
            reverse=True,
        )

        return matches[:limit]

    def get_context_window(self, max_entries: int = 20) -> list[dict[str, Any]]:
        """
        Get a context window for the agent.

        Combines working memory with relevant episodic and semantic memories.
        """
        context = []

        # Add working memory
        for entry in self._working_memory[-max_entries:]:
            context.append({
                "type": "working",
                "content": entry.content,
                "timestamp": entry.timestamp,
            })

        return context

    async def clear_working_memory(self) -> None:
        """Clear working memory."""
        # Move important entries to episodic
        for entry in self._working_memory:
            if entry.importance > 0.5:
                self._episodic_memory.append(entry)

        self._working_memory.clear()

        if self.persist:
            await self._save_to_disk()

    async def consolidate(self) -> dict[str, int]:
        """
        Consolidate memory by:
        - Moving important working memories to episodic
        - Removing low-importance old episodic memories
        - Updating semantic memory based on patterns
        """
        stats = {
            "moved_to_episodic": 0,
            "removed_episodic": 0,
        }

        # Move high-importance working memories
        for entry in list(self._working_memory):
            if entry.importance > 0.7:
                self._episodic_memory.append(entry)
                stats["moved_to_episodic"] += 1

        # Prune old low-importance episodic memories
        if len(self._episodic_memory) > 1000:
            self._episodic_memory.sort(
                key=lambda e: (e.importance, e.timestamp)
            )
            removed = len(self._episodic_memory) - 1000
            self._episodic_memory = self._episodic_memory[-1000:]
            stats["removed_episodic"] = removed

        if self.persist:
            await self._save_to_disk()

        return stats

    def _get_storage_file(self) -> Path:
        """Get the storage file path."""
        if not self.storage_path:
            raise RuntimeError("No storage path configured")
        self.storage_path.mkdir(parents=True, exist_ok=True)
        return self.storage_path / f"memory_{self.agent_id}.json"

    async def _save_to_disk(self) -> None:
        """Save memory to disk."""
        if not self.storage_path:
            return

        data = {
            "agent_id": self.agent_id,
            "working_memory": [e.to_dict() for e in self._working_memory],
            "episodic_memory": [e.to_dict() for e in self._episodic_memory[-1000:]],
            "semantic_memory": {k: v.to_dict() for k, v in self._semantic_memory.items()},
            "saved_at": datetime.utcnow().isoformat(),
        }

        storage_file = self._get_storage_file()
        with open(storage_file, "w") as f:
            json.dump(data, f, indent=2)

    def _load_from_disk(self) -> None:
        """Load memory from disk."""
        if not self.storage_path:
            return

        storage_file = self._get_storage_file()
        if not storage_file.exists():
            return

        try:
            with open(storage_file, "r") as f:
                data = json.load(f)

            self._working_memory = [
                MemoryEntry.from_dict(e) for e in data.get("working_memory", [])
            ]
            self._episodic_memory = [
                MemoryEntry.from_dict(e) for e in data.get("episodic_memory", [])
            ]
            self._semantic_memory = {
                k: MemoryEntry.from_dict(v)
                for k, v in data.get("semantic_memory", {}).items()
            }

            logger.info(f"Loaded memory for agent {self.agent_id}")

        except Exception as e:
            logger.error(f"Failed to load memory: {e}")

    def get_stats(self) -> dict[str, int]:
        """Get memory statistics."""
        return {
            "working_memory_size": len(self._working_memory),
            "episodic_memory_size": len(self._episodic_memory),
            "semantic_memory_size": len(self._semantic_memory),
        }
