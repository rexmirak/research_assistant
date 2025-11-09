"""Category classification and validation using LLM."""

import json
import logging
from typing import Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)


class CategoryClassifier:
    """Classify and validate paper categories using LLM."""

    def __init__(
        self,
        model: str = "llama3.1:8b",
        temperature: float = 0.2,
        confidence_threshold: float = 0.7,
    ):
        """
        Initialize classifier.

        Args:
            model: Ollama model name
            temperature: Sampling temperature
            confidence_threshold: Minimum confidence for recategorization
        """
        self.model = model
        self.temperature = temperature
        self.confidence_threshold = confidence_threshold

    def classify_paper(
        self,
        title: str,
        abstract: Optional[str],
        current_category: str,
        available_categories: List[str],
        topic: str,
    ) -> Tuple[str, float, str]:
        """
        Classify paper into appropriate category.

        Args:
            title: Paper title
            abstract: Paper abstract
            current_category: Current assigned category
            available_categories: List of available categories
            topic: Research topic

        Returns:
            Tuple of (recommended_category, confidence, reason)
        """
        try:
            import ollama

            # Build prompt
            prompt = self._build_classification_prompt(
                title, abstract, current_category, available_categories, topic
            )

            # Call LLM
            response = ollama.generate(
                model=self.model, prompt=prompt, options={"temperature": self.temperature}
            )

            # Parse response
            result = self._parse_classification_response(response["response"])

            return result

        except Exception as e:
            logger.error(f"Classification failed: {e}")
            return current_category, 0.0, "Classification error"

    def should_recategorize(
        self, current_category: str, recommended_category: str, confidence: float
    ) -> bool:
        """
        Determine if paper should be moved to new category.

        Args:
            current_category: Current category
            recommended_category: Recommended category
            confidence: Classification confidence

        Returns:
            True if should move
        """
        return current_category != recommended_category and confidence >= self.confidence_threshold

    def is_relevant(
        self, title: str, abstract: Optional[str], topic: str
    ) -> Tuple[bool, float, str]:
        """
        Determine if paper is relevant to research topic.

        Args:
            title: Paper title
            abstract: Paper abstract
            topic: Research topic

        Returns:
            Tuple of (is_relevant, confidence, reason)
        """
        try:
            import ollama

            prompt = f"""Given the research topic and a paper, determine if the paper is relevant.

Research Topic:
{topic}

Paper Title: {title}

Paper Abstract: {abstract or 'Not available'}

Is this paper relevant to the research topic?
Respond in JSON format:
{{
  "relevant": true/false,
  "confidence": 0.0-1.0,
  "reason": "brief explanation"
}}
"""

            response = ollama.generate(
                model=self.model, prompt=prompt, options={"temperature": self.temperature}
            )

            # Parse response
            result = self._parse_json_response(response["response"])

            return (
                result.get("relevant", True),
                result.get("confidence", 0.5),
                result.get("reason", ""),
            )

        except Exception as e:
            logger.error(f"Relevance check failed: {e}")
            return True, 0.0, "Check failed"

    def _build_classification_prompt(
        self,
        title: str,
        abstract: Optional[str],
        current_category: str,
        available_categories: List[str],
        topic: str,
    ) -> str:
        """Build classification prompt."""
        categories_str = ", ".join(available_categories)

        return f"""Given a research paper and available categories, determine the most appropriate category.

Research Topic Context:
{topic}

Paper Title: {title}

Paper Abstract: {abstract or 'Not available'}

Current Category: {current_category}

Available Categories: {categories_str}

Which category best fits this paper? Consider the paper's primary focus and the research topic.

Respond in JSON format:
{{
  "category": "category_name",
  "confidence": 0.0-1.0,
  "reason": "brief explanation"
}}
"""

    def _parse_classification_response(self, response: str) -> Tuple[str, float, str]:
        """Parse LLM classification response."""
        try:
            # Try to extract JSON
            result = self._parse_json_response(response)

            return (
                result.get("category", ""),
                float(result.get("confidence", 0.0)),
                result.get("reason", ""),
            )
        except Exception as e:
            logger.error(f"Failed to parse classification response: {e}")
            return "", 0.0, "Parse error"

    def _parse_json_response(self, response: str) -> Dict:
        """Parse JSON from LLM response."""
        # Try to find JSON in response
        start = response.find("{")
        end = response.rfind("}") + 1

        if start >= 0 and end > start:
            json_str = response[start:end]
            return json.loads(json_str)

        raise ValueError("No JSON found in response")
