"""Relevance scoring and ranking."""

import logging
from typing import List, Optional, Dict, Tuple
import numpy as np
from core.embeddings import EmbeddingGenerator

logger = logging.getLogger(__name__)


class ScoringEngine:
    """Score papers for relevance to research topic."""

    def __init__(
        self,
        topic: str,
        embedding_generator: EmbeddingGenerator,
        min_score: float = 0.0,
        max_score: float = 10.0,
        relevance_threshold: float = 6.5,
    ):
        """
        Initialize scoring engine.

        Args:
            topic: Research topic description
            embedding_generator: Embedding generator instance
            min_score: Minimum score
            max_score: Maximum score
            relevance_threshold: Threshold for inclusion
        """
        self.topic = topic
        self.embedding_generator = embedding_generator
        self.min_score = min_score
        self.max_score = max_score
        self.relevance_threshold = relevance_threshold

        # Generate topic embedding
        logger.info("Generating topic embedding...")
        self.topic_embedding = self.embedding_generator.embed(topic)

        if not self.topic_embedding:
            raise ValueError("Failed to generate topic embedding")

    def score_paper(self, paper_embedding: List[float]) -> Tuple[float, bool]:
        """
        Score a paper's relevance to the topic.

        Args:
            paper_embedding: Paper embedding vector

        Returns:
            Tuple of (score 0-10, include boolean)
        """
        # Compute cosine similarity
        similarity = self.embedding_generator.cosine_similarity(
            self.topic_embedding, paper_embedding
        )

        # Map similarity [-1, 1] to score [0, 10]
        # Focus on positive similarities
        normalized_sim = max(0, similarity)  # Clip negative similarities
        score = normalized_sim * self.max_score

        # Round to 1 decimal
        score = round(score, 1)

        # Determine inclusion
        include = score >= self.relevance_threshold

        return score, include

    def score_papers_batch(
        self, paper_embeddings: Dict[str, List[float]]
    ) -> Dict[str, Tuple[float, bool]]:
        """
        Score multiple papers.

        Args:
            paper_embeddings: Dictionary mapping paper_id to embedding

        Returns:
            Dictionary mapping paper_id to (score, include)
        """
        scores = {}

        for paper_id, embedding in paper_embeddings.items():
            score, include = self.score_paper(embedding)
            scores[paper_id] = (score, include)

        logger.info(f"Scored {len(scores)} papers")
        return scores

    def calibrate_threshold(self, scores: List[float], target_percentile: float = 0.6) -> float:
        """
        Calibrate threshold based on score distribution.

        Args:
            scores: List of scores
            target_percentile: Target percentile for threshold

        Returns:
            Calibrated threshold
        """
        if not scores:
            return self.relevance_threshold

        threshold = np.percentile(scores, target_percentile * 100)
        logger.info(f"Calibrated threshold at {target_percentile:.0%} percentile: {threshold:.2f}")

        return float(threshold)

    def get_statistics(self, scores: Dict[str, Tuple[float, bool]]) -> Dict:
        """
        Get scoring statistics.

        Args:
            scores: Dictionary of paper scores

        Returns:
            Statistics dictionary
        """
        score_values = [s[0] for s in scores.values()]
        include_count = sum(1 for s in scores.values() if s[1])

        return {
            "total_papers": len(scores),
            "included": include_count,
            "excluded": len(scores) - include_count,
            "mean_score": float(np.mean(score_values)) if score_values else 0,
            "median_score": float(np.median(score_values)) if score_values else 0,
            "min_score": float(np.min(score_values)) if score_values else 0,
            "max_score": float(np.max(score_values)) if score_values else 0,
            "std_score": float(np.std(score_values)) if score_values else 0,
        }
