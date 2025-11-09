"""Text processing and normalization utilities."""

import re
from typing import Optional


def normalize_text(text: str) -> str:
    """
    Normalize text for consistent processing.

    Args:
        text: Raw text

    Returns:
        Normalized text
    """
    # Remove extra whitespace
    text = re.sub(r"\s+", " ", text)
    # Remove special characters but keep basic punctuation
    text = re.sub(r"[^\w\s.,;:!?()\[\]{}\-\'\"]+", "", text)
    return text.strip()


def extract_abstract(text: str, max_length: int = 2000) -> Optional[str]:
    """
    Extract abstract from paper text.

    Args:
        text: Full paper text
        max_length: Maximum abstract length

    Returns:
        Abstract text or None
    """
    # Common abstract markers
    patterns = [
        r"(?i)abstract[\s\n]+(.+?)(?:\n\s*\n|introduction|keywords)",
        r"(?i)summary[\s\n]+(.+?)(?:\n\s*\n|introduction)",
        r"(?i)^(.+?)(?:\n\s*\n|introduction)",  # First paragraph fallback
    ]

    for pattern in patterns:
        match = re.search(pattern, text, re.DOTALL)
        if match:
            abstract = match.group(1).strip()
            if 50 < len(abstract) < max_length:
                return normalize_text(abstract)

    return None


def extract_introduction(text: str, max_length: int = 5000) -> Optional[str]:
    """
    Extract introduction section from paper text.

    Args:
        text: Full paper text
        max_length: Maximum intro length

    Returns:
        Introduction text or None
    """
    # Look for introduction section
    # Allow optional whitespace before the next section header (common in PDFs)
    pattern = r"(?i)introduction[\s\n]+(.+?)(?:\n\s*\n\s*(?:related work|background|method|approach|experiment))"
    match = re.search(pattern, text, re.DOTALL)

    if match:
        intro = match.group(1).strip()
        if len(intro) > 100:
            return normalize_text(intro[:max_length])

    return None


def clean_title(title: str) -> str:
    """
    Clean and normalize paper title.

    Args:
        title: Raw title

    Returns:
        Cleaned title
    """
    # Remove line breaks and extra spaces
    title = " ".join(title.split())
    # Remove trailing punctuation except question marks
    title = re.sub(r"[.,;:]$", "", title)
    return title.strip()


def truncate_text(text: str, max_chars: int = 1000, suffix: str = "...") -> str:
    """
    Truncate text to maximum length.

    Args:
        text: Text to truncate
        max_chars: Maximum characters
        suffix: Suffix to add if truncated

    Returns:
        Truncated text
    """
    if len(text) <= max_chars:
        return text

    return text[: max_chars - len(suffix)].rsplit(" ", 1)[0] + suffix


def create_bibtex_key(authors: list[str], year: str, title: str) -> str:
    """
    Create BibTeX citation key.

    Args:
        authors: List of author names
        year: Publication year
        title: Paper title

    Returns:
        BibTeX key (e.g., smith2023neural)
    """
    # Get first author last name
    if authors:
        first_author = authors[0].split()[-1].lower()
    else:
        first_author = "unknown"

    # Get first significant word from title
    title_words = [w.lower() for w in re.findall(r"\b\w+\b", title) if len(w) > 3]
    title_word = title_words[0] if title_words else "paper"

    # Clean and combine
    first_author = re.sub(r"[^\w]", "", first_author)
    title_word = re.sub(r"[^\w]", "", title_word)

    return f"{first_author}{year}{title_word}"
