"""Configuration management for research assistant."""

import os
from pathlib import Path
from typing import Optional

import yaml
from pydantic import BaseModel, Field


class OllamaConfig(BaseModel):
    """Ollama configuration."""

    summarize_model: str = "deepseek-r1:8b"
    embed_model: str = "nomic-embed-text"
    classify_model: str = "deepseek-r1:8b"
    temperature: float = 0.1
    base_url: str = "http://localhost:11434"


class CrossrefConfig(BaseModel):
    """Crossref API configuration."""

    enabled: bool = True
    email: Optional[str] = None  # For polite pool
    timeout: int = 10


class DedupConfig(BaseModel):
    """Deduplication configuration."""

    similarity_threshold: float = 0.95
    use_minhash: bool = True
    num_perm: int = 128  # MinHash permutations


class ScoringConfig(BaseModel):
    """Relevance scoring configuration."""

    relevance_threshold: float = 6.5  # Include papers >= this score
    min_score: float = 0.0
    max_score: float = 10.0
    use_abstract_only: bool = False  # If False, use title+abstract+intro


class MoveConfig(BaseModel):
    """File moving configuration."""

    enabled: bool = True
    track_manifest: bool = True
    create_symlinks: bool = False  # Create symlinks in original location


class CacheConfig(BaseModel):
    """Cache configuration."""

    enabled: bool = True
    backend: str = "sqlite"  # sqlite or json
    ttl_days: int = 90  # Cache expiration


class ProcessingConfig(BaseModel):
    """Processing configuration."""

    workers: int = 4
    batch_size: int = 32
    max_pages_for_intro: int = 10  # Extract intro from first N pages
    ocr_language: str = "eng"
    skip_ocr_if_text_exists: bool = True


class Config(BaseModel):
    """Main configuration."""

    ollama: OllamaConfig = Field(default_factory=OllamaConfig)
    crossref: CrossrefConfig = Field(default_factory=CrossrefConfig)
    dedup: DedupConfig = Field(default_factory=DedupConfig)
    scoring: ScoringConfig = Field(default_factory=ScoringConfig)
    move: MoveConfig = Field(default_factory=MoveConfig)
    cache: CacheConfig = Field(default_factory=CacheConfig)
    processing: ProcessingConfig = Field(default_factory=ProcessingConfig)

    # Runtime parameters (set via CLI)
    root_dir: Optional[Path] = None
    topic: Optional[str] = None
    output_dir: Optional[Path] = None
    cache_dir: Optional[Path] = None
    dry_run: bool = False
    resume: bool = False

    @classmethod
    def from_yaml(cls, path: Path) -> "Config":
        """Load configuration from YAML file."""
        if not path.exists():
            return cls()

        with open(path, "r") as f:
            data = yaml.safe_load(f)

        return cls(**data)

    def to_yaml(self, path: Path):
        """Save configuration to YAML file."""
        with open(path, "w") as f:
            yaml.dump(
                self.model_dump(
                    exclude={"root_dir", "topic", "output_dir", "cache_dir", "dry_run", "resume"}
                ),
                f,
                default_flow_style=False,
            )

    def setup_directories(self):
        """Create necessary directories."""
        if self.output_dir:
            self.output_dir.mkdir(parents=True, exist_ok=True)
            (self.output_dir / "summaries").mkdir(exist_ok=True)
            (self.output_dir / "logs").mkdir(exist_ok=True)
            (self.output_dir / "manifests").mkdir(exist_ok=True)

        if self.cache_dir:
            self.cache_dir.mkdir(parents=True, exist_ok=True)
            (self.cache_dir / "ocr").mkdir(exist_ok=True)
            (self.cache_dir / "embeddings").mkdir(exist_ok=True)


# Default config instance
default_config = Config()
