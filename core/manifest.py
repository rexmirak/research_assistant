"""Manifest manager for tracking papers and moves across categories."""

import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Set

from filelock import FileLock, Timeout

logger = logging.getLogger(__name__)


class ManifestEntry:
    """Single manifest entry for a paper."""

    def __init__(self, data: Dict):
        self.paper_id: str = data["paper_id"]
        self.title: str = data.get("title", "")  # Added title field
        self.path: str = data.get("path", data.get("current_path", data.get("original_path", "")))
        self.content_hash: str = data["content_hash"]
        self.classification_reasoning: Optional[str] = data.get("classification_reasoning")
        self.relevance_score: Optional[int] = data.get("relevance_score")  # Score for this category
        self.topic_relevance: Optional[int] = data.get("topic_relevance")  # Overall topic relevance
        self.analyzed: bool = data.get("analyzed", False)
        self.canonical_id: Optional[str] = data.get("canonical_id")  # For duplicates

        # Backward compatibility - maintain old fields if present
        self._legacy_fields = {}
        for key in ["status", "original_category", "moved_from", "moved_to", "moved_at", "reason",
                    "category", "original_path", "current_path"]:
            if key in data:
                self._legacy_fields[key] = data[key]

    def to_dict(self) -> Dict:
        """Convert to dictionary."""
        result = {
            "paper_id": self.paper_id,
            "title": self.title,
            "path": self.path,
            "content_hash": self.content_hash,
            "classification_reasoning": self.classification_reasoning,
            "relevance_score": self.relevance_score,
            "topic_relevance": self.topic_relevance,
            "analyzed": self.analyzed,
        }

        # Add canonical_id only if it's a duplicate
        if self.canonical_id:
            result["canonical_id"] = self.canonical_id

        return result


class CategoryManifest:
    """Manifest for a single category tracking all papers."""

    def __init__(self, category: str, manifest_dir: Path):
        """
        Initialize category manifest.

        Args:
            category: Category name
            manifest_dir: Directory to store manifests
        """
        self.category = category
        self.manifest_path = manifest_dir / f"{category}.manifest.json"
        self.entries: Dict[str, ManifestEntry] = {}
        self.content_hashes: Set[str] = set()

        self.load()

    def load(self):
        """Load manifest from disk with file lock."""
        if self.manifest_path.exists():
            lock_path = str(self.manifest_path) + ".lock"
            lock = FileLock(lock_path, timeout=30)
            try:
                with lock:
                    with open(self.manifest_path, "r") as f:
                        data = json.load(f)

                    for entry_data in data.get("entries", []):
                        entry = ManifestEntry(entry_data)
                        self.entries[entry.paper_id] = entry
                        self.content_hashes.add(entry.content_hash)

                    logger.info(f"Loaded manifest for {self.category}: {len(self.entries)} entries")
            except Timeout:
                logger.error(f"Timeout acquiring lock for manifest load: {self.manifest_path}")
            except Exception as e:
                logger.error(f"Failed to load manifest for {self.category}: {e}")

    def save(self):
        """Save manifest to disk with file lock."""
        lock_path = str(self.manifest_path) + ".lock"
        lock = FileLock(lock_path, timeout=30)
        try:
            self.manifest_path.parent.mkdir(parents=True, exist_ok=True)

            data = {
                "category": self.category,
                "updated_at": datetime.now().isoformat(),
                "entries": [e.to_dict() for e in self.entries.values()],
            }

            with lock:
                with open(self.manifest_path, "w") as f:
                    json.dump(data, f, indent=2)

            logger.debug(f"Saved manifest for {self.category}")
        except Timeout:
            logger.error(f"Timeout acquiring lock for manifest save: {self.manifest_path}")
        except Exception as e:
            logger.error(f"Failed to save manifest for {self.category}: {e}")

    def add_paper(
        self,
        paper_id: str,
        title: str,
        path: str,
        content_hash: str,
        classification_reasoning: Optional[str] = None,
        relevance_score: Optional[int] = None,
        topic_relevance: Optional[int] = None,
        analyzed: bool = True,
    ) -> ManifestEntry:
        """
        Add paper to manifest.

        Args:
            paper_id: Unique paper identifier
            title: Paper title
            path: Current file path
            content_hash: Content hash for dedup
            classification_reasoning: LLM explanation for classification
            relevance_score: Relevance score for THIS category (1-10)
            topic_relevance: Overall topic relevance (1-10)
            analyzed: Whether paper has been analyzed

        Returns:
            Manifest entry
        """
        entry = ManifestEntry(
            {
                "paper_id": paper_id,
                "title": title,
                "path": path,
                "content_hash": content_hash,
                "classification_reasoning": classification_reasoning,
                "relevance_score": relevance_score,
                "topic_relevance": topic_relevance,
                "analyzed": analyzed,
            }
        )

        self.entries[paper_id] = entry
        self.content_hashes.add(content_hash)
        return entry

    def remove_paper(self, paper_id: str):
        """Remove a paper from the manifest entirely."""
        if paper_id in self.entries:
            entry = self.entries.pop(paper_id)
            self.content_hashes.discard(entry.content_hash)

    def update_path(self, paper_id: str, new_path: str):
        """Update paper path (e.g., after moving)."""
        if paper_id in self.entries:
            self.entries[paper_id].path = new_path

    def mark_analyzed(self, paper_id: str):
        """Mark paper as analyzed."""
        if paper_id in self.entries:
            self.entries[paper_id].analyzed = True

    def mark_duplicate(self, paper_id: str, canonical_id: str):
        """Mark paper as duplicate of canonical paper."""
        if paper_id in self.entries:
            entry = self.entries[paper_id]
            entry.canonical_id = canonical_id

    def is_analyzed(self, paper_id: str) -> bool:
        """Check if paper was already analyzed."""
        return paper_id in self.entries and self.entries[paper_id].analyzed

    def should_skip(self, paper_id: str) -> bool:
        """Check if paper should be skipped (already analyzed or duplicate)."""
        if paper_id not in self.entries:
            return False

        entry = self.entries[paper_id]
        # Skip if already analyzed or is a duplicate
        return entry.analyzed or entry.canonical_id is not None

    def has_content_hash(self, content_hash: str) -> bool:
        """Check if content hash exists in manifest (for dedup)."""
        return content_hash in self.content_hashes

    def get_by_content_hash(self, content_hash: str) -> Optional[ManifestEntry]:
        """Get entry by content hash."""
        for entry in self.entries.values():
            if entry.content_hash == content_hash:
                return entry
        return None

    def get_active_papers(self) -> List[ManifestEntry]:
        """Get all active papers (excluding duplicates)."""
        return [e for e in self.entries.values() if e.canonical_id is None]


class ManifestManager:
    """Manager for all category manifests."""

    def __init__(self, manifest_dir: Path):
        """
        Initialize manifest manager.

        Args:
            manifest_dir: Directory to store manifests
        """
        self.manifest_dir = manifest_dir
        self.manifest_dir.mkdir(parents=True, exist_ok=True)
        self.manifests: Dict[str, CategoryManifest] = {}

    def get_manifest(self, category: str) -> CategoryManifest:
        """Get or create manifest for category."""
        if category not in self.manifests:
            self.manifests[category] = CategoryManifest(category, self.manifest_dir)
        return self.manifests[category]

    def record_move(
        self, paper_id: str, from_category: str, to_category: str, new_path: str, reason: str
    ):
        """
        Record a paper move between categories, removing from source manifest and adding to destination.
        """
        source_manifest = self.get_manifest(from_category)
        source_entry = source_manifest.entries.get(paper_id)
        if source_entry:
            # Remove from source manifest
            source_manifest.remove_paper(paper_id)
            source_manifest.save()
            # Add to destination manifest
            dest_manifest = self.get_manifest(to_category)
            dest_manifest.add_paper(
                paper_id=paper_id,
                title=source_entry.title,
                path=new_path,
                content_hash=source_entry.content_hash,
                classification_reasoning=f"Moved from {from_category}: {reason}",
                relevance_score=source_entry.relevance_score,
                topic_relevance=source_entry.topic_relevance,
            )
            dest_manifest.save()
            logger.info(f"Recorded move: {paper_id} from {from_category} to {to_category}")

    def save_all(self):
        """Save all manifests."""
        for manifest in list(self.manifests.values()):
            manifest.save()

    def get_all_content_hashes(self) -> Dict[str, tuple[str, str]]:
        """
        Get all content hashes across all manifests for global dedup.

        Returns:
            Dict mapping content_hash -> (category, paper_id)
        """
        all_hashes = {}
        for category, manifest in self.manifests.items():
            for entry in manifest.entries.values():
                # Skip duplicates (they have canonical_id set)
                if entry.canonical_id is None:
                    all_hashes[entry.content_hash] = (category, entry.paper_id)
        return all_hashes
