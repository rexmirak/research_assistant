"""Unit tests for LLM provider routing (Ollama/Gemini)."""

import os
from unittest.mock import MagicMock, patch

import pytest

from config import Config
from utils.llm_provider import llm_generate


@pytest.fixture
def mock_config_ollama():
    """Mock config for Ollama provider."""
    config = Config()
    config.llm_provider = "ollama"
    config.ollama.summarize_model = "deepseek-r1:8b"
    return config


@pytest.fixture
def mock_config_gemini():
    """Mock config for Gemini provider."""
    config = Config()
    config.llm_provider = "gemini"
    config.gemini.api_key = "test_api_key"
    # Note: model and temperature are not stored in GeminiConfig,
    # they're passed as parameters or use defaults
    return config


def test_llm_generate_ollama_success(mock_config_ollama):
    """Test successful Ollama generation."""
    with patch("utils.llm_provider.Config", return_value=mock_config_ollama):
        with patch("ollama.generate") as mock_ollama:
            mock_ollama.return_value = {"response": "Test response from Ollama"}

            result = llm_generate("Test prompt", model="deepseek-r1:8b")

            assert result["response"] == "Test response from Ollama"
            mock_ollama.assert_called_once()
            call_args = mock_ollama.call_args
            assert call_args[1]["model"] == "deepseek-r1:8b"
            assert call_args[1]["prompt"] == "Test prompt"


def test_llm_generate_ollama_error(mock_config_ollama):
    """Test Ollama error handling."""
    with patch("utils.llm_provider.Config", return_value=mock_config_ollama):
        with patch("ollama.generate") as mock_ollama:
            mock_ollama.side_effect = Exception("Connection error")

            with pytest.raises(Exception, match="Connection error"):
                llm_generate("Test prompt")


def test_llm_generate_gemini_success(mock_config_gemini):
    """Test successful Gemini generation."""
    with patch("utils.llm_provider.Config", return_value=mock_config_gemini):
        with patch("utils.llm_provider.gemini_generate") as mock_gemini:
            mock_gemini.return_value = "Test response from Gemini"

            result = llm_generate("Test prompt")

            assert result["response"] == "Test response from Gemini"
            mock_gemini.assert_called_once()
            call_args = mock_gemini.call_args
            assert call_args[1]["api_key"] == "test_api_key"
            # Model and temperature should use defaults since not in config
            assert call_args[1]["model"] == "gemini-2.0-flash-exp"  # default
            assert call_args[1]["temperature"] == 0.1  # default


def test_llm_generate_gemini_json_mode(mock_config_gemini):
    """Test Gemini JSON generation with schema."""
    with patch("utils.llm_provider.Config", return_value=mock_config_gemini):
        with patch("utils.llm_provider.gemini_generate_json") as mock_gemini_json:
            test_schema = {"type": "object", "properties": {"title": {"type": "string"}}}
            mock_gemini_json.return_value = {"title": "Test Title"}

            result = llm_generate("Test prompt", options={"schema": test_schema})

            assert result["response"]["title"] == "Test Title"
            mock_gemini_json.assert_called_once()
            call_args = mock_gemini_json.call_args
            assert call_args[1]["schema"] == test_schema


def test_llm_generate_gemini_no_api_key():
    """Test Gemini with missing API key."""
    config = Config()
    config.llm_provider = "gemini"
    config.gemini.api_key = None

    with patch("utils.llm_provider.Config", return_value=config):
        with pytest.raises(ValueError, match="GEMINI_API_KEY not set"):
            llm_generate("Test prompt")


def test_llm_generate_gemini_error(mock_config_gemini):
    """Test Gemini error handling."""
    with patch("utils.llm_provider.Config", return_value=mock_config_gemini):
        with patch("utils.llm_provider.gemini_generate") as mock_gemini:
            mock_gemini.side_effect = Exception("API rate limit exceeded")

            with pytest.raises(Exception, match="API rate limit exceeded"):
                llm_generate("Test prompt")


def test_llm_generate_ollama_removes_schema_option(mock_config_ollama):
    """Test that schema option is removed for Ollama calls."""
    with patch("utils.llm_provider.Config", return_value=mock_config_ollama):
        with patch("ollama.generate") as mock_ollama:
            mock_ollama.return_value = {"response": "Test response"}
            test_schema = {"type": "object"}

            llm_generate("Test prompt", options={"schema": test_schema, "temperature": 0.5})

            call_args = mock_ollama.call_args
            # Schema should not be in options passed to Ollama
            assert "schema" not in call_args[1]["options"]
            # But temperature should be present
            assert call_args[1]["options"]["temperature"] == 0.5


def test_llm_generate_default_model_ollama(mock_config_ollama):
    """Test default model selection for Ollama."""
    with patch("utils.llm_provider.Config", return_value=mock_config_ollama):
        with patch("ollama.generate") as mock_ollama:
            mock_ollama.return_value = {"response": "Test response"}

            llm_generate("Test prompt")

            call_args = mock_ollama.call_args
            # Should use default summarize_model from config
            assert call_args[1]["model"] == "deepseek-r1:8b"


@pytest.mark.integration
def test_llm_generate_real_ollama():
    """Integration test with real Ollama service."""
    # Skip if Ollama is not available
    try:
        import ollama

        ollama.list()
    except Exception:
        pytest.skip("Ollama service not available")

    config = Config()
    config.llm_provider = "ollama"
    config.ollama.summarize_model = "deepseek-r1:8b"

    with patch("utils.llm_provider.Config", return_value=config):
        result = llm_generate("Say 'Hello'", model="deepseek-r1:8b")

        assert "response" in result
        assert isinstance(result["response"], str)
        assert len(result["response"]) > 0
