"""Manifest manager for tracking papers and moves across categories."""

import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Set

logger = logging.getLogger(__name__)


class ManifestEntry:
    """Single manifest entry for a paper."""

    def __init__(self, data: Dict):
        self.paper_id: str = data["paper_id"]
        self.original_path: str = data["original_path"]
        self.current_path: str = data.get("current_path", data["original_path"])
        self.status: str = data.get(
            "status", "active"
        )  # active, moved_out, moved_in, duplicate, quarantined
        self.category: str = data["category"]
        self.original_category: Optional[str] = data.get("original_category")
        self.moved_from: Optional[str] = data.get("moved_from")
        self.moved_to: Optional[str] = data.get("moved_to")
        self.moved_at: Optional[str] = data.get("moved_at")
        self.reason: Optional[str] = data.get("reason")
        self.analyzed: bool = data.get("analyzed", False)
        self.canonical_id: Optional[str] = data.get("canonical_id")  # For duplicates
        self.content_hash: str = data["content_hash"]

    def to_dict(self) -> Dict:
        """Convert to dictionary."""
        return {
            "paper_id": self.paper_id,
            "original_path": self.original_path,
            "current_path": self.current_path,
            "status": self.status,
            "category": self.category,
            "original_category": self.original_category,
            "moved_from": self.moved_from,
            "moved_to": self.moved_to,
            "moved_at": self.moved_at,
            "reason": self.reason,
            "analyzed": self.analyzed,
            "canonical_id": self.canonical_id,
            "content_hash": self.content_hash,
        }


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
        """Load manifest from disk."""
        if self.manifest_path.exists():
            try:
                with open(self.manifest_path, "r") as f:
                    data = json.load(f)

                for entry_data in data.get("entries", []):
                    entry = ManifestEntry(entry_data)
                    self.entries[entry.paper_id] = entry
                    self.content_hashes.add(entry.content_hash)

                logger.info(f"Loaded manifest for {self.category}: {len(self.entries)} entries")
            except Exception as e:
                logger.error(f"Failed to load manifest for {self.category}: {e}")

    def save(self):
        """Save manifest to disk."""
        try:
            self.manifest_path.parent.mkdir(parents=True, exist_ok=True)

            data = {
                "category": self.category,
                "updated_at": datetime.now().isoformat(),
                "entries": [e.to_dict() for e in self.entries.values()],
            }

            with open(self.manifest_path, "w") as f:
                json.dump(data, f, indent=2)

            logger.debug(f"Saved manifest for {self.category}")
        except Exception as e:
            logger.error(f"Failed to save manifest for {self.category}: {e}")

    def add_paper(
        self,
        paper_id: str,
        path: str,
        content_hash: str,
        status: str = "active",
        original_category: Optional[str] = None,
    ) -> ManifestEntry:
        """
        Add paper to manifest.

        Args:
            paper_id: Unique paper identifier
            path: Current file path
            content_hash: Content hash for dedup
            status: Paper status
            original_category: Original category if moved

        Returns:
            Manifest entry
        """
        entry = ManifestEntry(
            {
                "paper_id": paper_id,
                "original_path": path,
                "current_path": path,
                "status": status,
                "category": self.category,
                "original_category": original_category or self.category,
                "content_hash": content_hash,
                "analyzed": False,
            }
        )

        self.entries[paper_id] = entry
        self.content_hashes.add(content_hash)
        return entry

    def mark_moved_out(self, paper_id: str, to_category: str, reason: str):
        """Mark paper as moved out to another category."""
        if paper_id in self.entries:
            entry = self.entries[paper_id]
            entry.status = "moved_out"
            entry.moved_to = to_category
            entry.moved_at = datetime.now().isoformat()
            entry.reason = reason

    def mark_moved_in(self, paper_id: str, from_category: str, new_path: str, reason: str):
        """Mark paper as moved in from another category."""
        if paper_id in self.entries:
            entry = self.entries[paper_id]
            entry.status = "moved_in"
            entry.moved_from = from_category
            entry.current_path = new_path
            entry.moved_at = datetime.now().isoformat()
            entry.reason = reason

    def mark_analyzed(self, paper_id: str):
        """Mark paper as analyzed."""
        if paper_id in self.entries:
            self.entries[paper_id].analyzed = True

    def mark_duplicate(self, paper_id: str, canonical_id: str):
        """Mark paper as duplicate of canonical paper."""
        if paper_id in self.entries:
            entry = self.entries[paper_id]
            entry.status = "duplicate"
            entry.canonical_id = canonical_id

    def is_analyzed(self, paper_id: str) -> bool:
        """Check if paper was already analyzed."""
        return paper_id in self.entries and self.entries[paper_id].analyzed

    def should_skip(self, paper_id: str) -> bool:
        """Check if paper should be skipped (moved out or duplicate)."""
        if paper_id not in self.entries:
            return False

        status = self.entries[paper_id].status
        return status in ["moved_out", "duplicate"]

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
        """Get all active papers (not moved out or duplicates)."""
        return [e for e in self.entries.values() if e.status in ["active", "moved_in"]]


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
        Record a paper move between categories.

        Args:
            paper_id: Paper identifier
            from_category: Source category
            to_category: Destination category
            new_path: New file path
            reason: Reason for move
        """
        # Mark moved out in source
        source_manifest = self.get_manifest(from_category)
        source_manifest.mark_moved_out(paper_id, to_category, reason)
        source_manifest.save()

        # Mark moved in at destination
        dest_manifest = self.get_manifest(to_category)

        # Get original entry data
        source_entry = source_manifest.entries.get(paper_id)
        if source_entry:
            # Create entry in destination
            dest_manifest.add_paper(
                paper_id=paper_id,
                path=new_path,
                content_hash=source_entry.content_hash,
                status="moved_in",
                original_category=source_entry.original_category,
            )
            dest_manifest.mark_moved_in(paper_id, from_category, new_path, reason)

        dest_manifest.save()

        logger.info(f"Recorded move: {paper_id} from {from_category} to {to_category}")

    def save_all(self):
        """Save all manifests."""
        for manifest in self.manifests.values():
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
                if entry.status not in ["moved_out", "duplicate"]:  # Don't count moved/dup
                    all_hashes[entry.content_hash] = (category, entry.paper_id)
        return all_hashes
