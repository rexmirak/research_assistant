"""Embedding generation using Ollama."""

import logging
from typing import List, Optional, Dict
import numpy as np

logger = logging.getLogger(__name__)


class EmbeddingGenerator:
    """Generate embeddings using Ollama."""

    def __init__(
        self,
        model: str = "nomic-embed-text",
        base_url: str = "http://localhost:11434",
        batch_size: int = 64,
    ):
        """
        Initialize embedding generator.

        Args:
            model: Ollama embedding model name
            base_url: Ollama API base URL
            batch_size: Batch size for embedding generation
        """
        self.model = model
        self.base_url = base_url
        self.batch_size = batch_size

    def embed(self, text: str) -> Optional[List[float]]:
        """
        Generate embedding for text.

        Args:
            text: Input text

        Returns:
            Embedding vector or None
        """
        try:
            import ollama

            response = ollama.embeddings(model=self.model, prompt=text)

            return response["embedding"]

        except Exception as e:
            logger.error(f"Embedding generation failed: {e}")
            return None

    def embed_batch(self, texts: List[str]) -> List[Optional[List[float]]]:
        """
        Generate embeddings for batch of texts.

        Args:
            texts: List of input texts

        Returns:
            List of embedding vectors
        """
        embeddings = []

        # Process in batches
        for i in range(0, len(texts), self.batch_size):
            batch = texts[i : i + self.batch_size]
            logger.debug(
                f"Processing embedding batch {i//self.batch_size + 1}/{(len(texts)-1)//self.batch_size + 1}"
            )

            for text in batch:
                embedding = self.embed(text)
                embeddings.append(embedding)

        return embeddings

    def embed_paper(
        self, title: str, abstract: Optional[str] = None, intro: Optional[str] = None
    ) -> Optional[List[float]]:
        """
        Generate embedding for paper using title, abstract, and intro.

        Args:
            title: Paper title
            abstract: Paper abstract
            intro: Paper introduction

        Returns:
            Combined embedding vector
        """
        # Combine available text
        parts = [title]
        if abstract:
            parts.append(abstract)
        elif intro:
            # Use intro if no abstract
            parts.append(intro[:1000])  # Limit intro length

        text = " ".join(parts)
        return self.embed(text)

    @staticmethod
    def cosine_similarity(vec1: List[float], vec2: List[float]) -> float:
        """
        Compute cosine similarity between two vectors.

        Args:
            vec1: First vector
            vec2: Second vector

        Returns:
            Cosine similarity score
        """
        v1 = np.array(vec1)
        v2 = np.array(vec2)

        dot_product = np.dot(v1, v2)
        norm1 = np.linalg.norm(v1)
        norm2 = np.linalg.norm(v2)

        if norm1 == 0 or norm2 == 0:
            return 0.0

        return float(dot_product / (norm1 * norm2))

    def test_connection(self) -> bool:
        """Test connection to Ollama service."""
        try:
            import ollama

            # Try to list models
            models = ollama.list()

            # Check if our model is available
            model_names = [m["name"] for m in models.get("models", [])]
            if self.model not in model_names:
                logger.warning(f"Model {self.model} not found. Available models: {model_names}")
                return False

            logger.info(f"Ollama connection successful, using model: {self.model}")
            return True

        except Exception as e:
            logger.error(f"Ollama connection failed: {e}")
            return False
