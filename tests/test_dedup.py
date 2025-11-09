"""Test deduplication functionality."""

from core.dedup import DedupManager


def test_minhash_initialization():
    """Test MinHash initialization."""
    dedup = DedupManager(similarity_threshold=0.9, num_perm=128)

    assert dedup.similarity_threshold == 0.9
    assert dedup.num_perm == 128


def test_compute_minhash():
    """Test MinHash computation."""
    dedup = DedupManager()

    text1 = "This is a sample paper about machine learning"
    text2 = "This is a sample paper about deep learning"
    text3 = "Completely different content about biology"

    mh1 = dedup._compute_minhash(text1)
    mh2 = dedup._compute_minhash(text2)
    mh3 = dedup._compute_minhash(text3)

    # Similar texts should have higher Jaccard similarity
    sim_12 = mh1.jaccard(mh2)
    sim_13 = mh1.jaccard(mh3)

    assert sim_12 > sim_13
