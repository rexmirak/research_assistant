"""LLM-based taxonomy generation for research paper categorization."""

import json
import logging
from pathlib import Path
from typing import Dict, List, Optional

from utils.llm_provider import llm_generate
from core.category_validator import CategoryValidator

logger = logging.getLogger(__name__)


class TaxonomyGenerator:
    """Generates research paper categories using LLM based on topic."""

    def __init__(
        self,
        cache_dir: Path,
        output_dir: Path,
        min_categories: int = 3,
        max_categories: int = 25,
    ):
        """
        Initialize taxonomy generator.

        Args:
            cache_dir: Directory to cache generated taxonomies
            output_dir: Directory to save output taxonomies
            min_categories: Minimum number of categories (default: 3)
            max_categories: Maximum number of categories (default: 25)
        """
        self.cache_dir = cache_dir
        self.output_dir = output_dir
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.validator = CategoryValidator(
            min_categories=min_categories, max_categories=max_categories
        )

    def generate_categories(
        self, topic: str, force_regenerate: bool = False
    ) -> Dict[str, str]:
        """
        Generate research paper categories for a given topic using LLM.

        Args:
            topic: Research topic (e.g., "Prompt Injection Attacks")
            force_regenerate: If True, regenerate even if cached

        Returns:
            Dictionary mapping category names to their definitions
            Example: {
                "attack_vectors": "Papers describing methods and techniques...",
                "defense_mechanisms": "Research on mitigation strategies...",
                ...
            }
        """
        cache_file = self.cache_dir / "categories.json"
        output_file = self.output_dir / "categories.json"

        # Check cache unless force regenerate
        if not force_regenerate and cache_file.exists():
            logger.info(f"Loading cached taxonomy from {cache_file}")
            try:
                with open(cache_file, "r") as f:
                    cached_data = json.load(f)
                    if cached_data.get("topic") == topic:
                        categories = cached_data.get("categories", {})
                        logger.info(
                            f"Using cached taxonomy with {len(categories)} categories"
                        )
                        # Also save to output dir
                        self._save_taxonomy(output_file, topic, categories)
                        return categories
                    else:
                        logger.info(
                            f"Cached topic '{cached_data.get('topic')}' doesn't match '{topic}', regenerating"
                        )
            except Exception as e:
                logger.warning(f"Failed to load cached taxonomy: {e}")

        # Generate new taxonomy
        logger.info(f"Generating taxonomy for topic: {topic}")
        categories = self._generate_with_llm(topic)

        # Save to both cache and output
        self._save_taxonomy(cache_file, topic, categories)
        self._save_taxonomy(output_file, topic, categories)

        logger.info(f"Generated {len(categories)} categories for topic: {topic}")
        return categories

    def _generate_with_llm(self, topic: str) -> Dict[str, str]:
        """
        Use LLM to generate category taxonomy.

        Args:
            topic: Research topic

        Returns:
            Dictionary of category names to definitions
        """
        prompt = f"""You are a research librarian organizing academic papers on the topic: "{topic}"

Your task is to generate a comprehensive taxonomy of research paper categories that would be relevant to this topic.

Guidelines:
1. Focus on distinct research areas, methodologies, and application domains
2. Categories should be mutually exclusive where possible
3. Each category should have a clear, specific definition
4. Aim for 5-15 categories (be comprehensive but not excessive)
5. Use snake_case for category names (e.g., "attack_vectors", "defense_mechanisms")
6. Definitions should be 1-2 sentences explaining what papers belong in this category

Question: If I am researching "{topic}", what categories of papers am I looking for?

Return your response as a JSON object with this exact structure:
{{
    "category_name_1": "Definition of what papers belong in this category...",
    "category_name_2": "Definition of what papers belong in this category...",
    ...
}}

Think carefully about the natural divisions in this research area. Consider:
- Different types of approaches or methodologies
- Different application domains or contexts
- Different stages of research (foundational, applied, evaluation)
- Different perspectives (offensive, defensive, analytical)

Return ONLY the JSON object, no other text."""

        try:
            from config import Config
            import re

            cfg = Config()
            provider = getattr(cfg, "llm_provider", "ollama")
            
            response = llm_generate(
                prompt=prompt,
                model=None,  # Use default model from config
                options={"temperature": 0.1},
            )
            
            # Extract response text
            response_text = response["response"].strip()
            
            # Clean response (remove markdown code blocks if present)
            if response_text.startswith("```json"):
                response_text = response_text[7:]
            if response_text.startswith("```"):
                response_text = response_text[3:]
            if response_text.endswith("```"):
                response_text = response_text[:-3]
            response_text = response_text.strip()

            # Parse JSON
            raw_categories = json.loads(response_text)

            if not isinstance(raw_categories, dict):
                raise ValueError("LLM response is not a dictionary")

            if len(raw_categories) == 0:
                raise ValueError("LLM generated zero categories")

            # Validate and sanitize categories
            logger.info(
                f"Validating {len(raw_categories)} LLM-generated categories..."
            )
            sanitized, warnings, errors = self.validator.validate_and_sanitize(
                raw_categories
            )

            # Log warnings
            for warning in warnings:
                logger.warning(f"Category validation: {warning}")

            # Check for critical errors
            if errors:
                error_msg = "; ".join(errors)
                logger.error(f"Category validation failed: {error_msg}")
                raise ValueError(f"Category validation failed: {error_msg}")

            logger.info(
                f"Validated taxonomy: {len(sanitized)} categories after sanitization"
            )
            return sanitized

        except json.JSONDecodeError as e:
            logger.error(f"Failed to parse LLM response as JSON: {e}")
            logger.error(f"Response was: {response[:500]}...")
            raise
        except Exception as e:
            logger.error(f"Failed to generate taxonomy with LLM: {e}")
            raise

    def _save_taxonomy(self, path: Path, topic: str, categories: Dict[str, str]):
        """
        Save taxonomy to JSON file.

        Args:
            path: File path to save to
            topic: Research topic
            categories: Category dictionary
        """
        data = {
            "topic": topic,
            "categories": categories,
            "num_categories": len(categories),
            "category_names": list(categories.keys()),
        }

        try:
            with open(path, "w") as f:
                json.dump(data, f, indent=2)
            logger.info(f"Saved taxonomy to {path}")
        except Exception as e:
            logger.error(f"Failed to save taxonomy to {path}: {e}")
            raise

    def load_categories(self, source: str = "cache") -> Optional[Dict[str, str]]:
        """
        Load categories from file.

        Args:
            source: "cache" or "output"

        Returns:
            Dictionary of categories or None if not found
        """
        if source == "cache":
            path = self.cache_dir / "categories.json"
        else:
            path = self.output_dir / "categories.json"

        if not path.exists():
            return None

        try:
            with open(path, "r") as f:
                data = json.load(f)
                return data.get("categories", {})
        except Exception as e:
            logger.error(f"Failed to load categories from {path}: {e}")
            return None
