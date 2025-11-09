"""Integration tests for full pipeline."""

import pytest
from pathlib import Path
import tempfile
import shutil
from unittest.mock import Mock, patch, MagicMock
from core.inventory import InventoryManager, PDFDocument
from core.manifest import ManifestManager
from core.dedup import DedupManager
from cache.cache_manager import CacheManager


@pytest.fixture
def temp_workspace():
    """Create temporary workspace with mock PDF structure."""
    temp_dir = Path(tempfile.mkdtemp())

    # Create category directories
    cat_a = temp_dir / "CategoryA"
    cat_b = temp_dir / "CategoryB"
    cat_a.mkdir()
    cat_b.mkdir()

    # Create mock PDF files
    (cat_a / "paper1.pdf").write_text("Mock PDF content 1")
    (cat_a / "paper2.pdf").write_text("Mock PDF content 2")
    (cat_b / "paper3.pdf").write_text("Mock PDF content 3")

    yield temp_dir

    shutil.rmtree(temp_dir)


@pytest.fixture
def cache_dir():
    """Create temporary cache directory."""
    temp_dir = Path(tempfile.mkdtemp())
    yield temp_dir
    shutil.rmtree(temp_dir)


@pytest.fixture
def output_dir():
    """Create temporary output directory."""
    temp_dir = Path(tempfile.mkdtemp())
    yield temp_dir
    shutil.rmtree(temp_dir)


def test_inventory_scanning(temp_workspace):
    """Test inventory manager scans PDFs correctly."""
    inventory = InventoryManager(temp_workspace)
    documents = inventory.scan()

    assert len(documents) == 3
    assert all(isinstance(doc, PDFDocument) for doc in documents)

    categories = inventory.get_categories()
    assert "CategoryA" in categories
    assert "CategoryB" in categories


def test_inventory_summary(temp_workspace):
    """Test inventory summary generation."""
    inventory = InventoryManager(temp_workspace)
    inventory.scan()

    summary = inventory.summary()

    assert summary["total_documents"] == 3
    assert summary["total_categories"] == 2
    assert "CategoryA" in summary["categories"]
    assert summary["categories"]["CategoryA"]["count"] == 2


def test_manifest_with_inventory(temp_workspace, output_dir):
    """Test manifest tracking with inventory."""
    # Scan PDFs
    inventory = InventoryManager(temp_workspace)
    documents = inventory.scan()

    # Create manifests
    manifest_dir = output_dir / "manifests"
    manifest_manager = ManifestManager(manifest_dir)

    # Add documents to manifests
    for doc in documents:
        manifest = manifest_manager.get_manifest(doc.category)
        manifest.add_paper(
            paper_id=doc.file_hash[:12], path=str(doc.file_path), content_hash=doc.file_hash
        )

    manifest_manager.save_all()

    # Verify manifests created
    cat_a_manifest = manifest_dir / "CategoryA.manifest.json"
    assert cat_a_manifest.exists()

    # Reload and verify
    new_manager = ManifestManager(manifest_dir)
    cat_a = new_manager.get_manifest("CategoryA")
    assert len(cat_a.entries) == 2


def test_move_tracking_integration(temp_workspace, output_dir):
    """Test move tracking prevents duplicate analysis."""
    inventory = InventoryManager(temp_workspace)
    documents = inventory.scan()

    manifest_dir = output_dir / "manifests"
    manifest_manager = ManifestManager(manifest_dir)

    # Add a paper to CategoryA
    doc = documents[0]
    paper_id = doc.file_hash[:12]

    manifest_a = manifest_manager.get_manifest("CategoryA")
    manifest_a.add_paper(paper_id=paper_id, path=str(doc.file_path), content_hash=doc.file_hash)
    manifest_a.mark_analyzed(paper_id)

    # Record move to CategoryB
    manifest_manager.record_move(
        paper_id=paper_id,
        from_category="CategoryA",
        to_category="CategoryB",
        new_path=str(temp_workspace / "CategoryB" / doc.file_name),
        reason="Better category fit",
    )

    # Verify source marked as moved_out
    assert manifest_a.should_skip(paper_id)
    assert manifest_a.entries[paper_id].status == "moved_out"

    # Verify destination has entry
    manifest_b = manifest_manager.get_manifest("CategoryB")
    assert paper_id in manifest_b.entries
    assert manifest_b.entries[paper_id].status == "moved_in"


def test_deduplication_integration(temp_workspace):
    """Test deduplication with inventory."""
    inventory = InventoryManager(temp_workspace)
    documents = inventory.scan()

    # Create duplicate (same content)
    cat_a = temp_workspace / "CategoryA"
    duplicate_path = cat_a / "paper1_duplicate.pdf"
    original_path = cat_a / "paper1.pdf"
    shutil.copy(original_path, duplicate_path)

    # Rescan
    inventory = InventoryManager(temp_workspace)
    documents = inventory.scan()

    # Find duplicates
    dedup_manager = DedupManager()
    exact_dups = dedup_manager.find_exact_duplicates(documents)

    assert len(exact_dups) > 0  # Should find duplicates


def test_cache_integration(cache_dir):
    """Test cache integration with multiple operations."""
    cache = CacheManager(cache_dir)

    # Simulate processing pipeline
    papers = [
        {"id": "paper1", "text": "Content 1", "embedding": [0.1, 0.2]},
        {"id": "paper2", "text": "Content 2", "embedding": [0.3, 0.4]},
        {"id": "paper3", "text": "Content 3", "embedding": [0.5, 0.6]},
    ]

    # Cache everything
    for paper in papers:
        cache.set_text(paper["id"], paper["text"], f"hash_{paper['id']}")
        cache.set_embedding(f"{paper['id']}_embed", paper["embedding"])
        cache.set_metadata(paper["id"], {"title": f"Paper {paper['id']}"})

    # Retrieve and verify
    for paper in papers:
        text, _ = cache.get_text(paper["id"])
        assert text == paper["text"]

        embedding = cache.get_embedding(f"{paper['id']}_embed")
        assert embedding == paper["embedding"]

        metadata = cache.get_metadata(paper["id"])
        assert metadata["title"] == f"Paper {paper['id']}"


def test_end_to_end_workflow(temp_workspace, cache_dir, output_dir):
    """Test end-to-end workflow from inventory to manifest."""
    # 1. Scan inventory
    inventory = InventoryManager(temp_workspace)
    documents = inventory.scan()
    assert len(documents) == 3

    # 2. Initialize components
    cache = CacheManager(cache_dir)
    manifest_manager = ManifestManager(output_dir / "manifests")

    # 3. Process each document
    for doc in documents:
        paper_id = doc.file_hash[:12]

        # Add to manifest
        manifest = manifest_manager.get_manifest(doc.category)
        manifest.add_paper(paper_id=paper_id, path=str(doc.file_path), content_hash=doc.file_hash)

        # Mock cache data
        cache.set_text(paper_id, f"Text content for {doc.file_name}", doc.file_hash)
        cache.set_metadata(paper_id, {"title": doc.file_name})

        # Mark as analyzed
        manifest.mark_analyzed(paper_id)

    # 4. Save manifests
    manifest_manager.save_all()

    # 5. Verify everything
    assert (output_dir / "manifests" / "CategoryA.manifest.json").exists()
    assert (output_dir / "manifests" / "CategoryB.manifest.json").exists()

    # 6. Reload and verify persistence
    new_manager = ManifestManager(output_dir / "manifests")
    cat_a = new_manager.get_manifest("CategoryA")
    assert len(cat_a.entries) == 2
    assert all(entry.analyzed for entry in cat_a.entries.values())


@patch("ollama.embeddings")
def test_mock_ollama_integration(mock_embeddings):
    """Test integration with mocked Ollama."""
    from core.embeddings import EmbeddingGenerator

    # Mock Ollama response
    mock_embeddings.return_value = {"embedding": [0.1] * 768}

    generator = EmbeddingGenerator()
    embedding = generator.embed("Test text")

    assert embedding is not None
    assert len(embedding) == 768
    mock_embeddings.assert_called_once()


def test_scoring_with_embeddings():
    """Test scoring with mock embeddings."""
    from core.scoring import ScoringEngine

    class MockEmbedding:
        def embed(self, text):
            return [0.5] * 768

        @staticmethod
        def cosine_similarity(v1, v2):
            return 0.75  # Mock similarity

    engine = ScoringEngine(
        topic="Test topic", embedding_generator=MockEmbedding(), relevance_threshold=6.0
    )

    paper_embedding = [0.5] * 768
    score, include = engine.score_paper(paper_embedding)

    assert 0 <= score <= 10
    assert isinstance(include, bool)
    assert score > 0  # Should have positive score


def test_full_pipeline_with_mocks(temp_workspace, cache_dir, output_dir):
    """Test full pipeline with mocked external services."""
    # This tests the integration without requiring actual services

    # Inventory
    inventory = InventoryManager(temp_workspace)
    documents = inventory.scan()

    # Cache
    cache = CacheManager(cache_dir)

    # Manifests
    manifest_manager = ManifestManager(output_dir / "manifests")

    # Dedup
    dedup = DedupManager()

    # Process simulation
    processed_papers = {}

    for doc in documents:
        paper_id = doc.file_hash[:12]

        # Mock text extraction
        text = f"Mock text for {doc.file_name}"
        cache.set_text(paper_id, text, doc.file_hash)

        # Mock metadata
        metadata = {"title": doc.file_name, "authors": ["Mock Author"], "year": "2023"}
        cache.set_metadata(paper_id, metadata)

        # Mock embedding
        embedding = [0.1] * 768
        cache.set_embedding(f"{paper_id}_embed", embedding)

        # Add to manifest
        manifest = manifest_manager.get_manifest(doc.category)
        manifest.add_paper(paper_id, str(doc.file_path), doc.file_hash)
        manifest.mark_analyzed(paper_id)

        processed_papers[paper_id] = {
            "doc": doc,
            "text": text,
            "metadata": metadata,
            "embedding": embedding,
        }

    # Save manifests
    manifest_manager.save_all()

    # Verify results
    assert len(processed_papers) == 3
    assert all(cache.get_text(pid) is not None for pid in processed_papers.keys())
    assert (output_dir / "manifests").exists()
