"""Unit tests for cache manager."""

import shutil
import tempfile
from pathlib import Path

import pytest

from utils.cache_manager import CacheManager


@pytest.fixture
def cache_dir():
    """Create temporary cache directory."""
    temp_dir = Path(tempfile.mkdtemp())
    yield temp_dir
    shutil.rmtree(temp_dir)


@pytest.fixture
def cache_manager(cache_dir):
    """Create cache manager instance."""
    return CacheManager(cache_dir, ttl_days=90)


def test_cache_initialization(cache_manager, cache_dir):
    """Test cache manager initializes correctly."""
    assert cache_manager.cache_dir.exists()
    assert cache_manager.db_path.exists()


def test_embedding_cache(cache_manager):
    """Test embedding caching."""
    key = "test_embedding"
    embedding = [0.1, 0.2, 0.3, 0.4]

    # Set embedding
    cache_manager.set_embedding(key, embedding)

    # Get embedding
    retrieved = cache_manager.get_embedding(key)
    assert retrieved == embedding


def test_embedding_cache_miss(cache_manager):
    """Test cache miss returns None."""
    retrieved = cache_manager.get_embedding("nonexistent")
    assert retrieved is None


def test_metadata_cache(cache_manager):
    """Test metadata caching."""
    paper_id = "paper123"
    metadata = {"title": "Test Paper", "authors": ["Author One", "Author Two"], "year": "2023"}

    cache_manager.set_metadata(paper_id, metadata)
    retrieved = cache_manager.get_metadata(paper_id)

    assert retrieved == metadata
    assert retrieved["title"] == "Test Paper"


def test_text_cache(cache_manager):
    """Test text extract caching."""
    paper_id = "paper456"
    text = "This is the extracted text content"
    text_hash = "abc123hash"

    cache_manager.set_text(paper_id, text, text_hash)
    retrieved_text, retrieved_hash = cache_manager.get_text(paper_id)

    assert retrieved_text == text
    assert retrieved_hash == text_hash


def test_text_cache_miss(cache_manager):
    """Test text cache miss returns None."""
    result = cache_manager.get_text("nonexistent")
    assert result is None


def test_processing_state_cache(cache_manager):
    """Test processing state caching."""
    paper_id = "paper789"
    stage = "embeddings"
    state = {"completed": True, "score": 7.5, "timestamp": "2023-01-01"}

    cache_manager.set_processing_state(paper_id, stage, state)
    retrieved = cache_manager.get_processing_state(paper_id, stage)

    assert retrieved == state
    assert retrieved["score"] == 7.5


def test_processing_state_update(cache_manager):
    """Test updating processing state."""
    paper_id = "paper_update"
    stage = "scoring"

    state1 = {"score": 5.0}
    cache_manager.set_processing_state(paper_id, stage, state1)

    state2 = {"score": 8.0}
    cache_manager.set_processing_state(paper_id, stage, state2)

    retrieved = cache_manager.get_processing_state(paper_id, stage)
    assert retrieved["score"] == 8.0  # Should be updated


def test_multiple_papers_cache(cache_manager):
    """Test caching multiple papers."""
    for i in range(5):
        paper_id = f"paper{i}"
        metadata = {"title": f"Paper {i}"}
        cache_manager.set_metadata(paper_id, metadata)

    # Retrieve all
    for i in range(5):
        paper_id = f"paper{i}"
        metadata = cache_manager.get_metadata(paper_id)
        assert metadata["title"] == f"Paper {i}"
