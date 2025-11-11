"""Unit tests for core/mover.py: file moving logic (mocked, no real file operations)."""

from pathlib import Path
from unittest.mock import MagicMock

import pytest

from core.manifest import ManifestManager
from core.mover import FileMover


@pytest.fixture
def mover(tmp_path):
    manifest_manager = ManifestManager(tmp_path / "manifests")
    return FileMover(
        root_dir=tmp_path,
        manifest_manager=manifest_manager,
        dry_run=True,
        create_symlinks=False,
    )


def test_move_to_category_updates_manifest(mover, tmp_path):
    # Simulate a move
    mover.manifest_manager.get_manifest("catA").add_paper(
        paper_id="abc123", path=str(tmp_path / "catA" / "file.pdf"), content_hash="hash1"
    )
    mover.move_to_category = MagicMock(return_value=tmp_path / "catB" / "file.pdf")
    new_path = mover.move_to_category(
        paper_id="abc123",
        current_path=tmp_path / "catA" / "file.pdf",
        from_category="catA",
        to_category="catB",
        reason="Test move",
    )
    assert new_path == tmp_path / "catB" / "file.pdf"


def test_move_to_quarantined_returns_path(mover, tmp_path):
    mover.move_to_quarantined = MagicMock(return_value=tmp_path / "quarantined" / "file.pdf")
    new_path = mover.move_to_quarantined(
        paper_id="abc123",
        current_path=tmp_path / "catA" / "file.pdf",
        from_category="catA",
        reason="Test quarantine",
    )
    assert new_path == tmp_path / "quarantined" / "file.pdf"
