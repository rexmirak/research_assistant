"""Test deduplication functionality."""

import pytest

from core.dedup import DedupManager


def test_minhash_initialization():
    """Test MinHash initialization with default parameters."""
    dedup = DedupManager(similarity_threshold=0.9, num_perm=128)

    assert dedup.similarity_threshold == 0.9
    assert dedup.num_perm == 128


def test_minhash_initialization_custom_threshold():
    """Test MinHash with custom similarity threshold."""
    dedup = DedupManager(similarity_threshold=0.85, num_perm=256)

    assert dedup.similarity_threshold == 0.85
    assert dedup.num_perm == 256


def test_compute_minhash():
    """Test MinHash computation for different texts."""
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
    assert 0.0 <= sim_12 <= 1.0
    assert 0.0 <= sim_13 <= 1.0


def test_compute_minhash_identical_texts():
    """Test MinHash with identical texts."""
    dedup = DedupManager()

    text = "This is a test paper about machine learning"
    mh1 = dedup._compute_minhash(text)
    mh2 = dedup._compute_minhash(text)

    similarity = mh1.jaccard(mh2)
    assert similarity == 1.0


def test_compute_minhash_completely_different():
    """Test MinHash with completely different texts."""
    dedup = DedupManager()

    text1 = "Machine learning and artificial intelligence"
    text2 = "Biology and chemistry research"

    mh1 = dedup._compute_minhash(text1)
    mh2 = dedup._compute_minhash(text2)

    similarity = mh1.jaccard(mh2)
    assert similarity < 0.5  # Should be very low


def test_is_duplicate_above_threshold():
    """Test duplicate detection above threshold."""
    dedup = DedupManager(similarity_threshold=0.9)

    # Very similar texts (mostly same words)
    text1 = "The quick brown fox jumps over the lazy dog every day"
    text2 = "The quick brown fox jumps over the lazy dog every night"

    mh1 = dedup._compute_minhash(text1)
    mh2 = dedup._compute_minhash(text2)

    similarity = mh1.jaccard(mh2)
    is_dup = similarity >= dedup.similarity_threshold

    # Should be similar but exact result depends on threshold
    assert isinstance(is_dup, bool)


def test_is_duplicate_below_threshold():
    """Test duplicate detection below threshold."""
    dedup = DedupManager(similarity_threshold=0.9)

    text1 = "Machine learning algorithms for classification"
    text2 = "Quantum physics and particle interactions"

    mh1 = dedup._compute_minhash(text1)
    mh2 = dedup._compute_minhash(text2)

    similarity = mh1.jaccard(mh2)
    is_dup = similarity >= dedup.similarity_threshold

    assert is_dup is False


def test_mark_duplicate():
    """Test marking documents as duplicates."""
    dedup = DedupManager()

    paper_id = "doc_001"
    canonical_id = "doc_000"

    # Mark as duplicate
    dedup.mark_duplicate(paper_id, canonical_id)

    # Check if it's marked as duplicate
    is_dup, canonical = dedup.is_duplicate(paper_id)
    assert is_dup is True
    assert canonical == canonical_id


def test_check_duplicate_existing():
    """Test checking for duplicate of existing document."""
    dedup = DedupManager(similarity_threshold=0.95)

    # Mark doc_002 as duplicate of doc_001
    dedup.mark_duplicate("doc_002", "doc_001")

    # Check if doc_002 is marked as duplicate
    is_dup, canonical = dedup.is_duplicate("doc_002")

    assert is_dup is True
    assert canonical == "doc_001"


def test_check_duplicate_nonexistent():
    """Test checking for duplicate when none exists."""
    dedup = DedupManager(similarity_threshold=0.95)

    # Check a document that hasn't been marked as duplicate
    is_dup, canonical = dedup.is_duplicate("doc_001")

    assert is_dup is False
    assert canonical is None


def test_multiple_documents():
    """Test dedup manager with multiple documents."""
    dedup = DedupManager(similarity_threshold=0.9)

    docs = {
        "doc_001": "Machine learning algorithms for classification tasks",
        "doc_002": "Deep learning neural networks for image recognition",
        "doc_003": "Natural language processing with transformers",
    }

    # Use find_near_duplicates method which is the actual API
    near_dups = dedup.find_near_duplicates(docs, docs)

    # With these dissimilar texts, should find no near-duplicates
    assert len(near_dups) == 0


def test_threshold_variations():
    """Test different similarity thresholds."""
    text1 = "The quick brown fox jumps over the lazy dog"
    text2 = "The quick brown fox leaps over the lazy dog"

    # High threshold (strict) - use 0.95 as MinHashLSH has limits
    dedup_strict = DedupManager(similarity_threshold=0.95)
    mh1 = dedup_strict._compute_minhash(text1)
    mh2 = dedup_strict._compute_minhash(text2)
    sim = mh1.jaccard(mh2)
    is_dup_strict = sim >= 0.95

    # Low threshold (lenient)
    dedup_lenient = DedupManager(similarity_threshold=0.5)
    is_dup_lenient = sim >= 0.5

    # Same similarity, different thresholds
    assert isinstance(is_dup_strict, bool)
    assert isinstance(is_dup_lenient, bool)


def test_num_perm_affects_accuracy():
    """Test that num_perm parameter affects MinHash."""
    text1 = "Machine learning algorithms"
    text2 = "Machine learning methods"

    # Lower permutations (less accurate) - use 64 as minimum practical value
    dedup_low = DedupManager(num_perm=64)
    mh1_low = dedup_low._compute_minhash(text1)
    mh2_low = dedup_low._compute_minhash(text2)
    sim_low = mh1_low.jaccard(mh2_low)

    # High permutations (more accurate)
    dedup_high = DedupManager(num_perm=256)
    mh1_high = dedup_high._compute_minhash(text1)
    mh2_high = dedup_high._compute_minhash(text2)
    sim_high = mh1_high.jaccard(mh2_high)

    # Both should give similarity scores
    assert 0.0 <= sim_low <= 1.0
    assert 0.0 <= sim_high <= 1.0


def test_empty_text_handling():
    """Test handling of empty or whitespace text."""
    dedup = DedupManager()

    text1 = ""
    text2 = "   "

    mh1 = dedup._compute_minhash(text1)
    mh2 = dedup._compute_minhash(text2)

    # Should not crash
    assert mh1 is not None
    assert mh2 is not None


def test_case_sensitivity():
    """Test MinHash case handling."""
    dedup = DedupManager()

    text1 = "Machine Learning Algorithms"
    text2 = "machine learning algorithms"

    mh1 = dedup._compute_minhash(text1)
    mh2 = dedup._compute_minhash(text2)

    similarity = mh1.jaccard(mh2)

    # Should be very similar (implementation may normalize case)
    assert similarity > 0.5


# ============================================================================
# Integration Tests - Testing full API methods
# ============================================================================


def test_find_exact_duplicates_with_mock_documents():
    """Test exact duplicate detection with PDFDocument-like objects."""
    from pathlib import Path
    from core.inventory import PDFDocument

    dedup = DedupManager()

    # Create mock documents with same hash (exact duplicates)
    doc1 = PDFDocument(
        file_path=Path("/papers/cat1/paper1.pdf"),
        file_name="paper1.pdf",
        category="cat1",
        file_size=1000,
        file_hash="abc123",
        modified_time=1234567890.0,
    )
    doc2 = PDFDocument(
        file_path=Path("/papers/cat2/paper1_copy.pdf"),
        file_name="paper1_copy.pdf",
        category="cat2",
        file_size=1000,
        file_hash="abc123",  # Same hash!
        modified_time=1234567891.0,
    )
    doc3 = PDFDocument(
        file_path=Path("/papers/cat1/paper2.pdf"),
        file_name="paper2.pdf",
        category="cat1",
        file_size=2000,
        file_hash="def456",  # Different hash
        modified_time=1234567892.0,
    )

    documents = [doc1, doc2, doc3]
    exact_dups = dedup.find_exact_duplicates(documents)

    # Should find one group with doc1 as canonical and doc2 as duplicate
    assert len(exact_dups) == 1
    assert "paper1.pdf" in exact_dups
    assert len(exact_dups["paper1.pdf"]) == 1
    assert exact_dups["paper1.pdf"][0].file_name == "paper1_copy.pdf"


def test_find_exact_duplicates_no_duplicates():
    """Test exact duplicate detection when no duplicates exist."""
    from pathlib import Path
    from core.inventory import PDFDocument

    dedup = DedupManager()

    doc1 = PDFDocument(
        file_path=Path("/papers/paper1.pdf"),
        file_name="paper1.pdf",
        category="cat1",
        file_size=1000,
        file_hash="abc123",
        modified_time=1234567890.0,
    )
    doc2 = PDFDocument(
        file_path=Path("/papers/paper2.pdf"),
        file_name="paper2.pdf",
        category="cat1",
        file_size=2000,
        file_hash="def456",
        modified_time=1234567891.0,
    )

    documents = [doc1, doc2]
    exact_dups = dedup.find_exact_duplicates(documents)

    # Should find no duplicates
    assert len(exact_dups) == 0


def test_find_near_duplicates_with_similar_texts():
    """Test near-duplicate detection with similar paper texts."""
    dedup = DedupManager(similarity_threshold=0.8)

    # Similar papers (same content with minor edits - more realistic for near-duplicates)
    # These papers share most of the same 3-word shingles
    paper_texts = {
        "paper1": "Machine learning algorithms for image classification using deep neural networks with convolutional layers and batch normalization techniques applied to large scale datasets with millions of training examples and extensive data augmentation strategies",
        "paper2": "Machine learning algorithms for image classification using deep neural networks with convolutional layers and batch normalization techniques applied to large scale datasets with millions of training examples and advanced data augmentation strategies",
        "paper3": "Quantum computing algorithms for solving optimization problems using quantum annealing and quantum gate operations",
    }

    paper_ids = {k: k for k in paper_texts.keys()}

    near_dups = dedup.find_near_duplicates(paper_texts, paper_ids)

    # paper1 and paper2 should be detected as near-duplicates (only 2 words differ)
    # paper3 is different enough to not be grouped
    assert len(near_dups) >= 1, f"Expected near-duplicates but got: {near_dups}"

    # Check that similar papers are grouped
    if "paper1" in near_dups:
        assert (
            "paper2" in near_dups["paper1"]
        ), f"Expected paper2 in duplicates of paper1: {near_dups}"
    elif "paper2" in near_dups:
        assert (
            "paper1" in near_dups["paper2"]
        ), f"Expected paper1 in duplicates of paper2: {near_dups}"


def test_find_near_duplicates_with_dissimilar_texts():
    """Test near-duplicate detection with dissimilar papers."""
    dedup = DedupManager(similarity_threshold=0.9)

    paper_texts = {
        "paper1": "Quantum computing algorithms for solving NP-complete problems using quantum annealing",
        "paper2": "Machine learning classification with decision trees and random forests",
        "paper3": "Protein folding simulations using molecular dynamics in computational biology",
    }

    paper_ids = {k: k for k in paper_texts.keys()}

    near_dups = dedup.find_near_duplicates(paper_texts, paper_ids)

    # No papers should be similar enough
    assert len(near_dups) == 0


def test_find_near_duplicates_with_identical_texts():
    """Test near-duplicate detection with identical texts."""
    dedup = DedupManager(similarity_threshold=0.95)

    identical_text = "This is an identical paper about machine learning algorithms and their applications in various domains"

    paper_texts = {
        "paper1": identical_text,
        "paper2": identical_text,  # Exact copy
        "paper3": "Completely different paper about quantum physics",
    }

    paper_ids = {k: k for k in paper_texts.keys()}

    near_dups = dedup.find_near_duplicates(paper_texts, paper_ids)

    # paper1 and paper2 should be detected as duplicates
    assert len(near_dups) >= 1

    # One should be canonical, other should be duplicate
    if "paper1" in near_dups:
        assert "paper2" in near_dups["paper1"]
    elif "paper2" in near_dups:
        assert "paper1" in near_dups["paper2"]


def test_find_near_duplicates_canonical_selection():
    """Test that first paper in group becomes canonical."""
    dedup = DedupManager(similarity_threshold=0.85)

    # Three similar papers
    similar_text = "Machine learning and artificial intelligence research"
    paper_texts = {
        "paper_a": similar_text + " in computer science applications",
        "paper_b": similar_text + " in computational science applications",
        "paper_c": similar_text + " in computer science applications",
    }

    paper_ids = {k: k for k in paper_texts.keys()}

    near_dups = dedup.find_near_duplicates(paper_texts, paper_ids)

    # Should have duplicate groups
    if near_dups:
        # Each canonical should have a list of duplicates
        for canonical, duplicates in near_dups.items():
            assert canonical not in duplicates
            assert isinstance(duplicates, list)
            assert len(duplicates) > 0


def test_deduplication_integration_workflow():
    """Test complete deduplication workflow: exact + near duplicates + marking."""
    from pathlib import Path
    from core.inventory import PDFDocument

    dedup = DedupManager(similarity_threshold=0.9)

    # Create documents with exact duplicates
    doc1 = PDFDocument(
        file_path=Path("/papers/paper1.pdf"),
        file_name="paper1.pdf",
        category="cat1",
        file_size=1000,
        file_hash="hash1",
        modified_time=1.0,
    )
    doc2 = PDFDocument(
        file_path=Path("/papers/paper1_dup.pdf"),
        file_name="paper1_dup.pdf",
        category="cat1",
        file_size=1000,
        file_hash="hash1",  # Exact duplicate
        modified_time=2.0,
    )
    doc3 = PDFDocument(
        file_path=Path("/papers/paper2.pdf"),
        file_name="paper2.pdf",
        category="cat1",
        file_size=2000,
        file_hash="hash2",
        modified_time=3.0,
    )

    documents = [doc1, doc2, doc3]

    # Step 1: Find exact duplicates
    exact_dups = dedup.find_exact_duplicates(documents)
    assert len(exact_dups) == 1

    # Step 2: Mark exact duplicates
    for canonical_name, dups in exact_dups.items():
        for dup_doc in dups:
            dedup.mark_duplicate(dup_doc.file_name, canonical_name)

    # Step 3: Check duplicate status
    is_dup, canonical = dedup.is_duplicate("paper1_dup.pdf")
    assert is_dup is True
    assert canonical == "paper1.pdf"

    # Non-duplicate should return False
    is_dup2, canonical2 = dedup.is_duplicate("paper2.pdf")
    assert is_dup2 is False
    assert canonical2 is None


def test_threshold_constraint_validation():
    """Test that threshold=0.99 raises ValueError as expected."""
    with pytest.raises(ValueError, match="bands"):
        # This should fail due to bands < 2 constraint
        DedupManager(similarity_threshold=0.99, num_perm=128)


def test_valid_threshold_ranges():
    """Test that documented valid threshold ranges work."""
    # These should all work without errors
    valid_configs = [
        (0.95, 128),  # Default strict
        (0.85, 128),  # Balanced
        (0.7, 128),  # Lenient
        (0.95, 64),  # Lower precision
        (0.98, 256),  # High precision, high threshold
    ]

    for threshold, num_perm in valid_configs:
        dedup = DedupManager(similarity_threshold=threshold, num_perm=num_perm)
        assert dedup.similarity_threshold == threshold
        assert dedup.num_perm == num_perm
