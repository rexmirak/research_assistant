"""Unit tests for configuration management."""

import os
import tempfile
from pathlib import Path
from unittest.mock import patch

import pytest
import yaml

from config import (
    CacheConfig,
    Config,
    CrossrefConfig,
    DedupConfig,
    GeminiConfig,
    MoveConfig,
    OllamaConfig,
    ProcessingConfig,
    ScoringConfig,
)


def test_default_config_initialization():
    """Test default configuration values."""
    config = Config()

    # llm_provider may be set by environment variable, so check it's either ollama or gemini
    assert config.llm_provider in ["ollama", "gemini"]
    assert isinstance(config.ollama, OllamaConfig)
    assert isinstance(config.gemini, GeminiConfig)
    assert isinstance(config.crossref, CrossrefConfig)
    assert isinstance(config.dedup, DedupConfig)
    assert isinstance(config.scoring, ScoringConfig)
    assert isinstance(config.move, MoveConfig)
    assert isinstance(config.cache, CacheConfig)
    assert isinstance(config.processing, ProcessingConfig)


def test_ollama_config_defaults():
    """Test Ollama configuration defaults."""
    config = OllamaConfig()

    assert config.summarize_model == "deepseek-r1:8b"
    assert config.embed_model == "nomic-embed-text"
    assert config.classify_model == "deepseek-r1:8b"
    assert config.temperature == 0.1
    assert config.base_url == "http://localhost:11434"


def test_gemini_config_defaults():
    """Test Gemini configuration defaults."""
    config = GeminiConfig()

    assert config.api_key is None


def test_dedup_config_defaults():
    """Test deduplication configuration defaults."""
    config = DedupConfig()

    assert config.similarity_threshold == 0.95
    assert config.use_minhash is True
    assert config.num_perm == 128


def test_scoring_config_defaults():
    """Test scoring configuration defaults."""
    config = ScoringConfig()

    assert config.relevance_threshold == 6.5
    assert config.min_score == 0.0
    assert config.max_score == 10.0
    assert config.use_abstract_only is False


def test_config_from_yaml_empty():
    """Test loading config from non-existent YAML file."""
    config = Config.from_yaml(Path("/non/existent/config.yaml"))

    # Should return default config (llm_provider may be set by env var)
    assert config.llm_provider in ["ollama", "gemini"]
    assert config.ollama.summarize_model == "deepseek-r1:8b"


def test_config_from_yaml_with_data():
    """Test loading config from YAML file with data."""
    with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
        yaml_data = {
            "llm_provider": "gemini",
            "ollama": {"summarize_model": "custom-model", "temperature": 0.5},
            "scoring": {"relevance_threshold": 7.0},
        }
        yaml.dump(yaml_data, f)
        yaml_path = Path(f.name)

    try:
        config = Config.from_yaml(yaml_path)

        assert config.llm_provider == "gemini"
        assert config.ollama.summarize_model == "custom-model"
        assert config.ollama.temperature == 0.5
        assert config.scoring.relevance_threshold == 7.0
        # Other values should be defaults
        assert config.ollama.embed_model == "nomic-embed-text"
    finally:
        yaml_path.unlink()


def test_config_to_yaml():
    """Test saving config to YAML file."""
    config = Config()
    config.llm_provider = "gemini"
    config.ollama.summarize_model = "custom-model"
    config.scoring.relevance_threshold = 8.0

    with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
        yaml_path = Path(f.name)

    try:
        config.to_yaml(yaml_path)

        with open(yaml_path, "r") as f:
            saved_data = yaml.safe_load(f)

        assert saved_data["llm_provider"] == "gemini"
        assert saved_data["ollama"]["summarize_model"] == "custom-model"
        assert saved_data["scoring"]["relevance_threshold"] == 8.0
        # Runtime parameters should be excluded
        assert "root_dir" not in saved_data
        assert "topic" not in saved_data
        assert "dry_run" not in saved_data
    finally:
        yaml_path.unlink()


def test_config_load_env_gemini_api_key():
    """Test loading Gemini API key from environment."""
    with patch.dict(os.environ, {"GEMINI_API_KEY": "test_key_123"}):
        config = Config()
        config.load_env()

        assert config.gemini.api_key == "test_key_123"


def test_config_load_env_llm_provider():
    """Test loading LLM provider from environment."""
    with patch.dict(os.environ, {"LLM_PROVIDER": "gemini"}):
        config = Config()
        config.load_env()

        assert config.llm_provider == "gemini"


def test_config_model_post_init_calls_load_env():
    """Test that model_post_init automatically calls load_env."""
    with patch.dict(os.environ, {"GEMINI_API_KEY": "auto_loaded_key"}):
        config = Config()

        # load_env should be called automatically in model_post_init
        assert config.gemini.api_key == "auto_loaded_key"


def test_config_setup_directories(tmp_path):
    """Test directory creation."""
    config = Config()
    config.output_dir = tmp_path / "outputs"
    config.cache_dir = tmp_path / "cache"

    config.setup_directories()

    assert (tmp_path / "outputs").exists()
    assert (tmp_path / "outputs" / "summaries").exists()
    assert (tmp_path / "outputs" / "logs").exists()
    assert (tmp_path / "outputs" / "manifests").exists()
    assert (tmp_path / "cache").exists()
    assert (tmp_path / "cache" / "ocr").exists()
    assert (tmp_path / "cache" / "embeddings").exists()


def test_config_runtime_parameters():
    """Test runtime parameters are separate from config."""
    config = Config()
    config.root_dir = Path("/test/root")
    config.topic = "AI Security"
    config.dry_run = True
    config.resume = True

    assert config.root_dir == Path("/test/root")
    assert config.topic == "AI Security"
    assert config.dry_run is True
    assert config.resume is True


def test_crossref_config_defaults():
    """Test Crossref configuration defaults."""
    config = CrossrefConfig()

    assert config.enabled is True
    assert config.email is None
    assert config.timeout == 10


def test_move_config_defaults():
    """Test file moving configuration defaults."""
    config = MoveConfig()

    assert config.enabled is True
    assert config.track_manifest is True
    assert config.create_symlinks is False


def test_cache_config_defaults():
    """Test cache configuration defaults."""
    config = CacheConfig()

    assert config.enabled is True
    assert config.backend == "sqlite"
    assert config.ttl_days == 90


def test_processing_config_defaults():
    """Test processing configuration defaults."""
    config = ProcessingConfig()

    assert config.workers == 4
    assert config.batch_size == 32
    assert config.max_pages_for_intro == 10
    assert config.ocr_language == "eng"
    assert config.skip_ocr_if_text_exists is True


def test_config_partial_yaml_override():
    """Test that YAML only overrides specified values."""
    with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
        yaml_data = {"scoring": {"relevance_threshold": 7.5}}
        yaml.dump(yaml_data, f)
        yaml_path = Path(f.name)

    try:
        config = Config.from_yaml(yaml_path)

        # Override should work
        assert config.scoring.relevance_threshold == 7.5
        # Other scoring values should be defaults
        assert config.scoring.min_score == 0.0
        assert config.scoring.max_score == 10.0
        # Other configs should be defaults
        assert config.llm_provider == "gemini"
        assert config.dedup.similarity_threshold == 0.95
    finally:
        yaml_path.unlink()


def test_config_env_file_loading():
    """Test loading from .env file."""
    with tempfile.NamedTemporaryFile(mode="w", suffix=".env", delete=False) as f:
        f.write("GEMINI_API_KEY=env_file_key\n")
        f.write("LLM_PROVIDER=gemini\n")
        env_path = Path(f.name)

    try:
        with patch("config.load_dotenv") as mock_load_dotenv:
            with patch.dict(
                os.environ, {"GEMINI_API_KEY": "env_file_key", "LLM_PROVIDER": "gemini"}
            ):
                config = Config()
                config.load_env()

                assert config.gemini.api_key == "env_file_key"
                assert config.llm_provider == "gemini"
    finally:
        env_path.unlink()
