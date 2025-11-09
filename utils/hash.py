"""Hash utilities for content identification and deduplication."""

import hashlib
from pathlib import Path


def file_hash(path: Path, algorithm: str = "sha256") -> str:
    """
    Compute hash of file contents.

    Args:
        path: Path to file
        algorithm: Hash algorithm (sha256, md5, etc.)

    Returns:
        Hex digest of file hash
    """
    hasher = hashlib.new(algorithm)
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            hasher.update(chunk)
    return hasher.hexdigest()


def text_hash(text: str, algorithm: str = "sha256") -> str:
    """
    Compute hash of normalized text content.

    Args:
        text: Text to hash
        algorithm: Hash algorithm

    Returns:
        Hex digest of text hash
    """
    # Normalize whitespace and encoding for consistent hashing
    normalized = " ".join(text.split()).lower()
    hasher = hashlib.new(algorithm)
    hasher.update(normalized.encode("utf-8"))
    return hasher.hexdigest()


def stable_paper_id(title: str, authors: list[str], year: str = "") -> str:
    """
    Generate stable paper ID from metadata.

    Args:
        title: Paper title
        authors: List of author names
        year: Publication year

    Returns:
        Short stable ID (first 12 chars of hash)
    """
    # Combine metadata for stable ID
    parts = [title.lower().strip(), "|".join(sorted(a.lower().strip() for a in authors)), str(year)]
    content = "|".join(p for p in parts if p)
    return text_hash(content, "sha256")[:12]
