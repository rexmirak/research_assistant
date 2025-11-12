"""Unit tests for Gemini API client."""

import json
from unittest.mock import MagicMock, patch

import pytest
from pydantic import BaseModel

from utils.gemini_client import gemini_generate, gemini_generate_json


class PaperSchema(BaseModel):
    """Test Pydantic schema for papers."""

    title: str
    score: int


@pytest.fixture
def mock_genai():
    """Mock google.generativeai module."""
    with patch("utils.gemini_client.genai") as mock:
        yield mock


def test_gemini_generate_success(mock_genai):
    """Test successful text generation."""
    mock_model = MagicMock()
    mock_response = MagicMock()
    mock_response.text = "Test response from Gemini"
    mock_model.generate_content.return_value = mock_response
    mock_genai.GenerativeModel.return_value = mock_model

    result = gemini_generate(
        prompt="Test prompt", api_key="test_api_key", model="gemini-2.0-flash-exp"
    )

    assert result == "Test response from Gemini"
    mock_genai.configure.assert_called_once_with(api_key="test_api_key")
    mock_genai.GenerativeModel.assert_called_once_with(
        model_name="gemini-2.0-flash-exp", generation_config={"temperature": 0.1}
    )
    mock_model.generate_content.assert_called_once()


def test_gemini_generate_no_api_key(mock_genai):
    """Test error when API key is missing."""
    with patch.dict("os.environ", {}, clear=True):  # Clear all env vars
        with pytest.raises(ValueError, match="GEMINI_API_KEY is not set"):
            gemini_generate(prompt="Test prompt", api_key=None)


def test_gemini_generate_rate_limit_retry(mock_genai):
    """Test retry logic on rate limit errors."""
    from google.api_core.exceptions import ResourceExhausted

    mock_model = MagicMock()
    mock_response = MagicMock()
    mock_response.text = "Success after retry"

    # First call raises rate limit, second succeeds
    mock_model.generate_content.side_effect = [
        ResourceExhausted("Rate limit exceeded"),
        mock_response,
    ]
    mock_genai.GenerativeModel.return_value = mock_model

    with patch("utils.gemini_client.time.sleep"):  # Patch the correct module path
        result = gemini_generate(prompt="Test prompt", api_key="test_api_key", max_retries=2)

    assert result == "Success after retry"
    assert mock_model.generate_content.call_count == 2


def test_gemini_generate_max_retries_exceeded(mock_genai):
    """Test failure when max retries is exceeded."""
    from google.api_core.exceptions import ResourceExhausted

    mock_model = MagicMock()
    mock_model.generate_content.side_effect = ResourceExhausted("Rate limit exceeded")
    mock_genai.GenerativeModel.return_value = mock_model

    with patch("utils.gemini_client.time.sleep"):  # Patch the correct module path
        with pytest.raises(ResourceExhausted, match="Rate limit exceeded"):
            gemini_generate(prompt="Test prompt", api_key="test_api_key", max_retries=3)

    assert mock_model.generate_content.call_count == 3


def test_gemini_generate_json_success(mock_genai):
    """Test successful JSON generation with schema."""
    mock_model = MagicMock()
    mock_response = MagicMock()
    mock_response.text = '{"title": "Test Paper", "score": 8}'
    mock_model.generate_content.return_value = mock_response
    mock_genai.GenerativeModel.return_value = mock_model

    result = gemini_generate_json(
        prompt="Extract metadata", schema=PaperSchema, api_key="test_api_key"
    )

    # gemini_generate_json returns a dict, not a Pydantic instance
    assert isinstance(result, dict)
    assert result["title"] == "Test Paper"
    assert result["score"] == 8

    # But it can be validated against the schema if needed
    validated = PaperSchema(**result)
    assert validated.title == "Test Paper"
    assert validated.score == 8


def test_gemini_generate_json_with_dict_schema(mock_genai):
    """Test JSON generation with dictionary schema."""
    mock_model = MagicMock()
    mock_response = MagicMock()
    mock_response.text = '{"title": "Test", "score": 5}'
    mock_model.generate_content.return_value = mock_response
    mock_genai.GenerativeModel.return_value = mock_model

    schema_dict = {
        "type": "object",
        "properties": {"title": {"type": "string"}, "score": {"type": "integer"}},
    }

    result = gemini_generate_json(
        prompt="Extract metadata", schema=schema_dict, api_key="test_api_key"
    )

    assert isinstance(result, dict)
    assert result["title"] == "Test"
    assert result["score"] == 5


def test_gemini_generate_json_invalid_json(mock_genai):
    """Test handling of invalid JSON response."""
    mock_model = MagicMock()
    mock_response = MagicMock()
    mock_response.text = "Not valid JSON"
    mock_model.generate_content.return_value = mock_response
    mock_genai.GenerativeModel.return_value = mock_model

    with pytest.raises(json.JSONDecodeError):
        gemini_generate_json(prompt="Extract metadata", schema=PaperSchema, api_key="test_api_key")


def test_gemini_generate_json_schema_validation_error(mock_genai):
    """Test handling of schema validation errors."""
    from pydantic import ValidationError

    mock_model = MagicMock()
    mock_response = MagicMock()
    # Missing required 'score' field
    mock_response.text = '{"title": "Test Paper"}'
    mock_model.generate_content.return_value = mock_response
    mock_genai.GenerativeModel.return_value = mock_model

    with pytest.raises(ValidationError):
        gemini_generate_json(prompt="Extract metadata", schema=PaperSchema, api_key="test_api_key")


def test_gemini_generate_json_dict_schema_no_validation(mock_genai):
    """Test that dict schemas don't trigger validation."""
    mock_model = MagicMock()
    mock_response = MagicMock()
    # Missing fields in response, but dict schema doesn't validate
    mock_response.text = '{"title": "Test"}'
    mock_model.generate_content.return_value = mock_response
    mock_genai.GenerativeModel.return_value = mock_model

    schema_dict = {
        "type": "object",
        "properties": {"title": {"type": "string"}, "score": {"type": "integer"}},
        "required": ["title", "score"],
    }

    # Should not raise ValidationError since dict schemas aren't validated
    result = gemini_generate_json(
        prompt="Extract metadata", schema=schema_dict, api_key="test_api_key"
    )

    assert isinstance(result, dict)
    assert result["title"] == "Test"
    assert "score" not in result  # Missing field is ok for dict schema


def test_gemini_generate_custom_temperature(mock_genai):
    """Test custom temperature parameter."""
    mock_model = MagicMock()
    mock_response = MagicMock()
    mock_response.text = "Response"
    mock_model.generate_content.return_value = mock_response
    mock_genai.GenerativeModel.return_value = mock_model

    gemini_generate(prompt="Test prompt", api_key="test_api_key", temperature=0.8)

    # Check that GenerativeModel was called with correct generation_config
    call_args = mock_genai.GenerativeModel.call_args
    gen_config = call_args[1]["generation_config"]
    assert gen_config["temperature"] == 0.8


def test_gemini_generate_json_custom_temperature(mock_genai):
    """Test JSON generation with custom temperature."""
    mock_model = MagicMock()
    mock_response = MagicMock()
    mock_response.text = '{"title": "Test", "score": 7}'
    mock_model.generate_content.return_value = mock_response
    mock_genai.GenerativeModel.return_value = mock_model

    result = gemini_generate_json(
        prompt="Extract metadata", schema=PaperSchema, api_key="test_api_key", temperature=0.9
    )

    # Verify the result
    assert isinstance(result, dict)
    assert result["title"] == "Test"
    assert result["score"] == 7

    # Verify temperature was passed correctly
    call_args = mock_genai.GenerativeModel.call_args
    gen_config = call_args[1]["generation_config"]
    assert gen_config["temperature"] == 0.9


def test_gemini_generate_json_schema_conversion(mock_genai):
    """Test that Pydantic schema is properly converted and passed to Gemini."""
    mock_model = MagicMock()
    mock_response = MagicMock()
    mock_response.text = '{"title": "Test", "score": 5}'
    mock_model.generate_content.return_value = mock_response
    mock_genai.GenerativeModel.return_value = mock_model

    result = gemini_generate_json(
        prompt="Extract metadata", schema=PaperSchema, api_key="test_api_key"
    )

    # Verify the result
    assert isinstance(result, dict)
    assert result["title"] == "Test"

    # Verify GenerativeModel was called with proper config
    call_args = mock_genai.GenerativeModel.call_args
    gen_config = call_args[1]["generation_config"]
    assert "response_mime_type" in gen_config
    assert gen_config["response_mime_type"] == "application/json"
    assert "response_schema" in gen_config
    assert isinstance(gen_config["response_schema"], dict)


def test_gemini_generate_json_rate_limit_retry(mock_genai):
    """Test JSON generation with rate limit retry."""
    from google.api_core.exceptions import ResourceExhausted

    mock_model = MagicMock()
    mock_response = MagicMock()
    mock_response.text = '{"title": "Success", "score": 9}'

    # First call raises rate limit, second succeeds
    mock_model.generate_content.side_effect = [
        ResourceExhausted("Rate limit exceeded"),
        mock_response,
    ]
    mock_genai.GenerativeModel.return_value = mock_model

    with patch("utils.gemini_client.time.sleep"):
        result = gemini_generate_json(
            prompt="Extract metadata", schema=PaperSchema, api_key="test_api_key", max_retries=2
        )

    assert result["title"] == "Success"
    assert result["score"] == 9
    assert mock_model.generate_content.call_count == 2


def test_gemini_generate_infinite_retries(mock_genai):
    """Test infinite retry mode (max_retries=None)."""
    from google.api_core.exceptions import ResourceExhausted

    mock_model = MagicMock()
    mock_response = MagicMock()
    mock_response.text = "Success"

    # Fail 5 times, then succeed
    mock_model.generate_content.side_effect = [
        ResourceExhausted("Rate limit"),
        ResourceExhausted("Rate limit"),
        ResourceExhausted("Rate limit"),
        ResourceExhausted("Rate limit"),
        ResourceExhausted("Rate limit"),
        mock_response,
    ]
    mock_genai.GenerativeModel.return_value = mock_model

    with patch("utils.gemini_client.time.sleep"):  # Patch the correct module path
        result = gemini_generate(prompt="Test prompt", api_key="test_api_key", max_retries=None)

    assert result == "Success"
    assert mock_model.generate_content.call_count == 6


@pytest.mark.integration
def test_gemini_generate_real_api():
    """Integration test with real Gemini API (requires valid API key)."""
    import os

    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        pytest.skip("GEMINI_API_KEY not set")

    result = gemini_generate(
        prompt="Say 'Hello' in one word.",
        api_key=api_key,
        model="gemini-2.0-flash-exp",
        temperature=0.0,
    )

    assert isinstance(result, str)
    assert len(result) > 0
    assert "hello" in result.lower()
