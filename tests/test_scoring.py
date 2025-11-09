"""Test scoring functionality."""

from core.embeddings import EmbeddingGenerator
from core.scoring import ScoringEngine


def test_score_calculation():
    """Test basic score calculation."""

    # Mock embedding generator
    class MockEmbedding:
        def embed(self, text):
            return [0.1] * 768

        @staticmethod
        def cosine_similarity(v1, v2):
            return 0.8

    mock_gen = MockEmbedding()

    engine = ScoringEngine(
        topic="Machine learning in healthcare",
        embedding_generator=mock_gen,
        relevance_threshold=6.5,
    )

    # Test scoring
    paper_embedding = [0.1] * 768
    score, include = engine.score_paper(paper_embedding)

    assert 0 <= score <= 10
    assert isinstance(include, bool)


def test_statistics():
    """Test statistics calculation."""

    class MockEmbedding:
        def embed(self, text):
            return [0.1] * 768

        @staticmethod
        def cosine_similarity(v1, v2):
            return 0.7

    mock_gen = MockEmbedding()

    engine = ScoringEngine(
        topic="Test topic", embedding_generator=mock_gen, relevance_threshold=6.0
    )

    scores = {"paper1": (7.5, True), "paper2": (4.2, False), "paper3": (8.9, True)}

    stats = engine.get_statistics(scores)

    assert stats["total_papers"] == 3
    assert stats["included"] == 2
    assert stats["excluded"] == 1
    assert "mean_score" in stats
