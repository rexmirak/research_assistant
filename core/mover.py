"""File moving with manifest tracking."""

import logging
import shutil
from pathlib import Path
from typing import Optional

from core.manifest import ManifestManager

logger = logging.getLogger(__name__)


class FileMover:
    """Handle file moves with manifest tracking."""

    def __init__(
        self,
        root_dir: Path,
        manifest_manager: ManifestManager,
        dry_run: bool = False,
        create_symlinks: bool = False,
    ):
        """
        Initialize file mover.

        Args:
            root_dir: Root directory for paper categories
            manifest_manager: Manifest manager instance
            dry_run: If True, log moves but don't execute
            create_symlinks: Create symlinks in original location
        """
        self.root_dir = root_dir
        self.manifest_manager = manifest_manager
        self.dry_run = dry_run
        self.create_symlinks = create_symlinks

    def move_to_category(
        self, paper_id: str, current_path: Path, from_category: str, to_category: str, reason: str
    ) -> Optional[Path]:
        """
        Move paper to different category.

        Args:
            paper_id: Paper identifier
            current_path: Current file path
            from_category: Source category
            to_category: Destination category
            reason: Reason for move

        Returns:
            New file path or None if failed
        """
        # Calculate new path
        dest_dir = self.root_dir / to_category
        dest_dir.mkdir(parents=True, exist_ok=True)
        new_path = dest_dir / current_path.name

        # Handle name conflicts
        if new_path.exists():
            new_path = self._resolve_conflict(new_path)

        logger.info(f"Moving {current_path.name} from {from_category} to {to_category}: {reason}")

        if self.dry_run:
            logger.info(f"[DRY RUN] Would move: {current_path} -> {new_path}")
        else:
            try:
                # Move file
                shutil.move(str(current_path), str(new_path))

                # Create symlink if enabled
                if self.create_symlinks:
                    try:
                        current_path.symlink_to(new_path)
                    except Exception as e:
                        logger.warning(f"Failed to create symlink: {e}")

                logger.info(f"Moved: {current_path.name} -> {new_path}")

            except Exception as e:
                logger.error(f"Failed to move {current_path.name}: {e}")
                return None

        # Record move in manifests
        self.manifest_manager.record_move(
            paper_id=paper_id,
            from_category=from_category,
            to_category=to_category,
            new_path=str(new_path),
            reason=reason,
        )

        return new_path

    def move_to_repeated(
        self, paper_id: str, current_path: Path, from_category: str, canonical_id: str
    ) -> Optional[Path]:
        """
        Move duplicate paper to repeated folder.

        Args:
            paper_id: Paper identifier
            current_path: Current file path
            from_category: Source category
            canonical_id: ID of canonical paper

        Returns:
            New file path or None
        """
        return self.move_to_category(
            paper_id=paper_id,
            current_path=current_path,
            from_category=from_category,
            to_category="repeated",
            reason=f"Duplicate of {canonical_id}",
        )

    def move_to_quarantined(
        self, paper_id: str, current_path: Path, from_category: str, reason: str
    ) -> Optional[Path]:
        """
        Move unrelated paper to quarantined folder.

        Args:
            paper_id: Paper identifier
            current_path: Current file path
            from_category: Source category
            reason: Reason for quarantine

        Returns:
            New file path or None
        """
        return self.move_to_category(
            paper_id=paper_id,
            current_path=current_path,
            from_category=from_category,
            to_category="quarantined",
            reason=reason,
        )

    def _resolve_conflict(self, path: Path) -> Path:
        """Resolve filename conflict by adding suffix."""
        stem = path.stem
        suffix = path.suffix
        parent = path.parent

        counter = 1
        while True:
            new_path = parent / f"{stem}_{counter}{suffix}"
            if not new_path.exists():
                return new_path
            counter += 1
