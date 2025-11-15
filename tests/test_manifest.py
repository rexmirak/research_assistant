"""Unit tests for core/manifest.py: Manifest tracking."""

from pathlib import Path

import pytest

from core.manifest import CategoryManifest, ManifestManager, ManifestEntry


@pytest.fixture
def temp_output_dir(tmp_path):
    """Create temporary output directory."""
    output_dir = tmp_path / "outputs" / "manifests"
    output_dir.mkdir(parents=True, exist_ok=True)
    return output_dir


class TestCategoryManifest:
    """Tests for CategoryManifest class."""

    def test_add_paper(self, temp_output_dir):
        """Test adding paper to manifest."""
        manifest = CategoryManifest(category="test_category", manifest_dir=temp_output_dir)
        
        entry = manifest.add_paper(
            paper_id="paper1",
            title="Test Paper",
            path="/test/path.pdf",
            content_hash="hash123",
            classification_reasoning="Good fit",
            relevance_score=9,
            topic_relevance=8,
        )
        
        assert entry.paper_id == "paper1"
        assert "paper1" in manifest.entries
        assert manifest.entries["paper1"].title == "Test Paper"

    def test_mark_duplicate(self, temp_output_dir):
        """Test marking paper as duplicate."""
        manifest = CategoryManifest(category="test_category", manifest_dir=temp_output_dir)
        
        manifest.add_paper(
            paper_id="canonical",
            title="Original Paper",
            path="/path1.pdf",
            content_hash="hash1",
        )
        
        manifest.add_paper(
            paper_id="duplicate",
            title="Duplicate Paper",
            path="/path2.pdf",
            content_hash="hash1",
        )
        
        manifest.mark_duplicate(paper_id="duplicate", canonical_id="canonical")
        
        assert manifest.entries["duplicate"].canonical_id == "canonical"

    def test_should_skip_duplicate(self, temp_output_dir):
        """Test should_skip returns True for duplicates."""
        manifest = CategoryManifest(category="test_category", manifest_dir=temp_output_dir)
        
        manifest.add_paper(
            paper_id="canonical",
            title="Original",
            path="/path1.pdf",
            content_hash="hash1",
        )
        
        manifest.add_paper(
            paper_id="duplicate",
            title="Duplicate",
            path="/path2.pdf",
            content_hash="hash1",
        )
        
        manifest.mark_duplicate(paper_id="duplicate", canonical_id="canonical")
        
        assert manifest.should_skip("duplicate") is True
        # Note: canonical is not marked analyzed yet, so should_skip returns False
        manifest.mark_analyzed("canonical")
        # After analyzing, canonical should not be skipped (it's the original)
        assert manifest.is_analyzed("canonical") is True
