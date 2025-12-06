"""Unit tests for core/taxonomy.py"""

import json
import os
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from core.taxonomy import TaxonomyGenerator


@pytest.fixture
def temp_dirs(tmp_path):
    """Create temporary cache and output directories."""
    cache_dir = tmp_path / "cache"
    output_dir = tmp_path / "outputs"
    cache_dir.mkdir(exist_ok=True)
    output_dir.mkdir(exist_ok=True)
    return cache_dir, output_dir


@pytest.fixture
def generator(temp_dirs):
    """Create TaxonomyGenerator with temp directories."""
    cache_dir, output_dir = temp_dirs
    return TaxonomyGenerator(cache_dir=cache_dir, output_dir=output_dir)


@pytest.fixture
def mock_llm_response():
    """Mock LLM response with valid categories."""
    return {
        "response": json.dumps({
            "attack_vectors": "Methods and techniques for attacking LLMs",
            "defense_mechanisms": "Strategies for defending against attacks",
            "evaluation_frameworks": "Methods for evaluating security",
            "prompt_injection_fundamentals": "Core concepts of prompt injection",
        })
    }


class TestTaxonomyGenerator:
    """Tests for TaxonomyGenerator class."""

    def test_initialization(self, temp_dirs):
        """Test generator initializes correctly."""
        cache_dir, output_dir = temp_dirs
        generator = TaxonomyGenerator(
            cache_dir=cache_dir,
            output_dir=output_dir,
            min_categories=5,
            max_categories=20
        )

        assert generator.cache_dir == cache_dir
        assert generator.output_dir == output_dir
        assert generator.validator.min_categories == 5
        assert generator.validator.max_categories == 20

    def test_directories_created(self, temp_dirs):
        """Test that directories are created if they don't exist."""
        cache_dir, output_dir = temp_dirs
        # Remove directories
        cache_dir.rmdir()
        output_dir.rmdir()

        generator = TaxonomyGenerator(cache_dir=cache_dir, output_dir=output_dir)

        assert cache_dir.exists()
        assert output_dir.exists()

    def test_generate_categories_with_mock(self, generator, mock_llm_response, temp_dirs):
        """Test category generation with mocked LLM."""
        cache_dir, output_dir = temp_dirs
        topic = "Prompt Injection Attacks"

        with patch("core.taxonomy.llm_generate", return_value=mock_llm_response):
            categories = generator.generate_categories(topic=topic, force_regenerate=True)

        assert isinstance(categories, dict)
        assert len(categories) >= 3  # Min categories
        assert "attack_vectors" in categories
        assert "defense_mechanisms" in categories

        # Check cache file was created
        cache_file = cache_dir / "categories.json"
        assert cache_file.exists()

        # Check output file was created
        output_file = output_dir / "categories.json"
        assert output_file.exists()

    def test_cache_loading(self, generator, mock_llm_response, temp_dirs):
        """Test that categories are loaded from cache."""
        cache_dir, output_dir = temp_dirs
        topic = "Test Topic"

        # First generation - should call LLM
        with patch("core.taxonomy.llm_generate", return_value=mock_llm_response) as mock_llm:
            categories1 = generator.generate_categories(topic=topic, force_regenerate=True)
            assert mock_llm.call_count == 1

        # Second generation - should use cache
        with patch("core.taxonomy.llm_generate", return_value=mock_llm_response) as mock_llm:
            categories2 = generator.generate_categories(topic=topic, force_regenerate=False)
            assert mock_llm.call_count == 0  # Should not call LLM

        assert categories1 == categories2

    def test_force_regenerate_bypasses_cache(self, generator, mock_llm_response, temp_dirs):
        """Test force_regenerate bypasses cache."""
        topic = "Test Topic"

        # First generation
        with patch("core.taxonomy.llm_generate", return_value=mock_llm_response) as mock_llm:
            generator.generate_categories(topic=topic, force_regenerate=True)
            assert mock_llm.call_count == 1

        # Second generation with force_regenerate
        with patch("core.taxonomy.llm_generate", return_value=mock_llm_response) as mock_llm:
            generator.generate_categories(topic=topic, force_regenerate=True)
            assert mock_llm.call_count == 1  # Should call LLM again

    def test_category_validation(self, generator):
        """Test that categories are validated and sanitized."""
        topic = "Test Topic"

        # Mock response with mix of valid and invalid categories
        mixed_response = {
            "response": json.dumps({
                "Valid Category One": "Good category",
                "Valid Category Two": "Another good one",
                "Valid Category Three": "Third one",
                "quarantined": "Reserved name - should be rejected",
                "a": "Too short - should be rejected",
            })
        }

        with patch("core.taxonomy.llm_generate", return_value=mixed_response):
            categories = generator.generate_categories(topic=topic, force_regenerate=True)

        # Valid categories should remain
        assert "valid_category_one" in categories
        assert "valid_category_two" in categories
        assert "valid_category_three" in categories
        # Invalid ones should be filtered
        assert "quarantined" not in categories
        assert "a" not in categories

    def test_minimum_categories_validation(self, generator):
        """Test that minimum category count is enforced."""
        topic = "Test Topic"

        # Mock response with too few categories
        insufficient_response = {
            "response": json.dumps({
                "category_one": "First category",
                "category_two": "Second category",
            })
        }

        with patch("core.taxonomy.llm_generate", return_value=insufficient_response):
            with pytest.raises(ValueError, match="minimum"):
                generator.generate_categories(topic=topic, force_regenerate=True)

    def test_llm_json_parsing_error(self, generator):
        """Test handling of invalid JSON from LLM."""
        topic = "Test Topic"

        # Mock response with invalid JSON
        invalid_json_response = {
            "response": "This is not valid JSON"
        }

        with patch("core.taxonomy.llm_generate", return_value=invalid_json_response):
            with pytest.raises((json.JSONDecodeError, ValueError, KeyError)):
                # Might raise JSONDecodeError or ValueError depending on error handling
                generator.generate_categories(topic=topic, force_regenerate=True)

    def test_cached_topic_mismatch(self, generator, mock_llm_response, temp_dirs):
        """Test that cache is regenerated if topic doesn't match."""
        cache_dir, _ = temp_dirs

        # Generate for topic 1
        with patch("core.taxonomy.llm_generate", return_value=mock_llm_response):
            generator.generate_categories(topic="Topic 1", force_regenerate=True)

        # Request for topic 2 - should regenerate
        with patch("core.taxonomy.llm_generate", return_value=mock_llm_response) as mock_llm:
            generator.generate_categories(topic="Topic 2", force_regenerate=False)
            assert mock_llm.call_count == 1  # Should call LLM for new topic


@pytest.mark.integration
@pytest.mark.skipif(not os.getenv("GEMINI_API_KEY"), reason="No Gemini API key")
def test_real_taxonomy_generation(temp_dirs):
    """Integration test with real LLM (requires API key)."""
    os.environ["LLM_PROVIDER"] = "gemini"
    cache_dir, output_dir = temp_dirs

    generator = TaxonomyGenerator(cache_dir=cache_dir, output_dir=output_dir)
    topic = "Prompt Injection Attacks in Large Language Models"

    categories = generator.generate_categories(topic=topic, force_regenerate=True)

    assert isinstance(categories, dict)
    assert len(categories) >= 3
    # Check that all values are strings (definitions)
    assert all(isinstance(v, str) for v in categories.values())
    # Check that all keys are sanitized
    assert all(k.islower() and "_" in k or k.isalnum() for k in categories.keys())
