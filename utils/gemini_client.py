"""Gemini API client for LLM completions using google.generativeai SDK, with support for structured JSON output."""

import json
import logging
import os
import time
from typing import Optional, Type

import google.generativeai as genai
from google.api_core import exceptions as google_exceptions

logger = logging.getLogger(__name__)


def gemini_generate(
    prompt: str,
    api_key: Optional[str] = None,
    model: str = "gemini-2.0-flash-exp",
    temperature: float = 0.1,
    max_retries: Optional[int] = None,
) -> str:
    """
    Generate text using Gemini API with automatic retry on rate limits.

    Args:
        prompt: Input prompt for text generation
        api_key: Gemini API key (defaults to GEMINI_API_KEY env var)
        model: Gemini model name
        temperature: Sampling temperature (0.0-1.0)
        max_retries: Maximum retry attempts (None for infinite retries)

    Returns:
        Generated text response

    Raises:
        ValueError: If API key is not set
        ResourceExhausted: If max_retries is exceeded
    """
    api_key = api_key or os.getenv("GEMINI_API_KEY")
    if not api_key:
        raise ValueError("GEMINI_API_KEY is not set.")
    genai.configure(api_key=api_key)

    generation_config = {
        "temperature": temperature,
    }

    model_instance = genai.GenerativeModel(
        model_name=model, generation_config=generation_config  # type: ignore[arg-type]
    )

    attempt = 0
    while True:
        try:
            response = model_instance.generate_content(prompt)
            return response.text  # type: ignore[no-any-return]
        except google_exceptions.ResourceExhausted as e:
            attempt += 1
            if max_retries is not None and attempt >= max_retries:
                logger.error(f"Max retries ({max_retries}) exceeded")
                raise
            retry_delay = 10  # 10 second delay to respect rate limits
            logger.warning(f"Rate limit hit, retrying in {retry_delay}s (attempt {attempt})")
            time.sleep(retry_delay)


def _clean_schema_for_gemini(schema_dict: dict) -> dict:
    """
    Clean Pydantic JSON schema to only include fields Gemini SDK accepts.
    Removes 'title', '$defs', and other non-standard fields.
    """
    cleaned = {}

    # Only keep fields that Gemini accepts
    allowed_fields = {"type", "properties", "required", "items", "enum", "description"}

    for key, value in schema_dict.items():
        if key in allowed_fields:
            if key == "properties" and isinstance(value, dict):
                # Recursively clean nested properties
                cleaned[key] = {
                    k: _clean_schema_for_gemini(v) if isinstance(v, dict) else v
                    for k, v in value.items()
                }
            elif key == "items" and isinstance(value, dict):
                # Recursively clean array items
                cleaned[key] = _clean_schema_for_gemini(value)
            else:
                cleaned[key] = value

    return cleaned


def gemini_generate_json(
    prompt: str,
    schema: Type,
    api_key: Optional[str] = None,
    model: str = "gemini-2.0-flash-exp",
    temperature: float = 0.1,
    max_retries: Optional[int] = None,
) -> dict:
    """
    Generate structured JSON output using Gemini SDK and a Pydantic schema.

    Automatically retries on rate limits. Converts Pydantic schema to JSON schema
    dict format compatible with Gemini SDK. Validates the returned JSON against
    the provided Pydantic schema.

    Args:
        prompt: Input prompt for JSON generation
        schema: Pydantic BaseModel class defining the output structure
        api_key: Gemini API key (defaults to GEMINI_API_KEY env var)
        model: Gemini model name
        temperature: Sampling temperature (0.0-1.0)
        max_retries: Maximum retry attempts (None for infinite retries)

    Returns:
        Parsed JSON dict matching the schema

    Raises:
        ValueError: If API key is not set
        ValidationError: If the generated JSON doesn't match the schema
        ResourceExhausted: If max_retries is exceeded
    """
    api_key = api_key or os.getenv("GEMINI_API_KEY")
    if not api_key:
        raise ValueError("GEMINI_API_KEY is not set.")
    genai.configure(api_key=api_key)

    # Convert Pydantic schema to JSON schema dict
    # Gemini SDK expects a dict-based JSON schema, not a Pydantic class
    json_schema: dict
    try:
        from pydantic import BaseModel

        if isinstance(schema, type) and issubclass(schema, BaseModel):
            # Use Pydantic's model_json_schema() to get JSON schema
            json_schema = schema.model_json_schema()
            # Clean the schema to remove fields Gemini doesn't accept
            json_schema = _clean_schema_for_gemini(json_schema)
        else:
            # Assume it's already a dict schema
            json_schema = schema  # type: ignore[assignment]
    except Exception:
        # Fallback: assume it's already a dict schema
        json_schema = schema  # type: ignore[assignment]

    generation_config = {
        "temperature": temperature,
        "response_mime_type": "application/json",
        "response_schema": json_schema,
    }

    model_instance = genai.GenerativeModel(
        model_name=model, generation_config=generation_config  # type: ignore[arg-type]
    )

    attempt = 0
    while True:
        try:
            response = model_instance.generate_content(prompt)
            parsed_json = json.loads(response.text)

            # Validate against schema if it's a Pydantic model
            from pydantic import BaseModel

            if isinstance(schema, type) and issubclass(schema, BaseModel):
                # Validate the parsed JSON against the Pydantic schema
                schema.model_validate(parsed_json)

            return parsed_json  # type: ignore[no-any-return]
        except google_exceptions.ResourceExhausted as e:
            attempt += 1
            if max_retries is not None and attempt >= max_retries:
                raise
            retry_delay = 10  # 10 second delay to respect rate limits
            logger.warning(f"Rate limit hit, retrying in {retry_delay}s (attempt {attempt})")
            time.sleep(retry_delay)
