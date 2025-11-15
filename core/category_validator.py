"""Category name validation and sanitization for filesystem safety and consistency."""

import logging
import re
from difflib import SequenceMatcher
from typing import Dict, List, Tuple

logger = logging.getLogger(__name__)


class CategoryValidator:
    """Validates and sanitizes category names for filesystem safety and consistency."""

    # Reserved filesystem names
    RESERVED_NAMES = {
        "con", "prn", "aux", "nul",
        "com1", "com2", "com3", "com4", "com5", "com6", "com7", "com8", "com9",
        "lpt1", "lpt2", "lpt3", "lpt4", "lpt5", "lpt6", "lpt7", "lpt8", "lpt9",
        "quarantined", "repeated", "need_human_element",  # Special system folders
    }

    # Dangerous patterns
    DANGEROUS_PATTERNS = [
        r"\.\.",  # Path traversal
        r"[/\\]",  # Path separators
        r"[:*?\"<>|]",  # Filesystem-unsafe characters
        r"^[~.]",  # Hidden files or home directory
    ]

    def __init__(
        self,
        min_categories: int = 3,
        max_categories: int = 25,
        similarity_threshold: float = 0.8,
    ):
        """
        Initialize category validator.

        Args:
            min_categories: Minimum number of valid categories required
            max_categories: Maximum number of categories allowed
            similarity_threshold: Fuzzy matching threshold (0.0-1.0)
        """
        self.min_categories = min_categories
        self.max_categories = max_categories
        self.similarity_threshold = similarity_threshold

    def sanitize_name(self, name: str) -> str:
        """
        Sanitize a category name to be filesystem-safe.

        Args:
            name: Raw category name

        Returns:
            Sanitized name (lowercase, underscores, alphanumeric only)

        Examples:
            "Attack Vectors" -> "attack_vectors"
            "Defense & Mitigation" -> "defense_mitigation"
            "../../etc/passwd" -> "etc_passwd"
        """
        if not name or not isinstance(name, str):
            return ""

        # Lowercase and strip
        name = name.lower().strip()

        # Replace spaces, dashes, and other separators with underscore
        name = re.sub(r'[\s\-&/\\]+', '_', name)

        # Remove all non-alphanumeric except underscore
        name = re.sub(r'[^\w_]', '', name)

        # Replace multiple underscores with single
        name = re.sub(r'_+', '_', name)

        # Remove leading/trailing underscores
        name = name.strip('_')

        # Truncate if too long (max 50 chars for filesystem)
        if len(name) > 50:
            name = name[:50].rstrip('_')

        return name

    def is_valid_name(self, name: str) -> Tuple[bool, str]:
        """
        Check if a name is valid after sanitization.

        Args:
            name: Sanitized category name

        Returns:
            Tuple of (is_valid, error_message)
        """
        if not name:
            return False, "Empty category name"

        if name in self.RESERVED_NAMES:
            return False, f"Reserved system name: {name}"

        # Check for dangerous patterns
        for pattern in self.DANGEROUS_PATTERNS:
            if re.search(pattern, name):
                return False, f"Contains dangerous pattern: {pattern}"

        # Must contain at least one letter
        if not re.search(r'[a-z]', name):
            return False, "Must contain at least one letter"

        # Should be reasonable length
        if len(name) < 2:
            return False, "Too short (minimum 2 characters)"

        if len(name) > 50:
            return False, "Too long (maximum 50 characters)"

        return True, ""

    def match_to_existing(
        self,
        name: str,
        existing_categories: List[str],
    ) -> str | None:
        """
        Fuzzy match a category name to existing categories.

        Handles:
        - Exact matches (case-insensitive)
        - Singular/plural variations
        - Typos and spelling mistakes
        - Similar names with high similarity score

        Args:
            name: Category name to match
            existing_categories: List of existing category names

        Returns:
            Matched category name, or None if no match found

        Examples:
            "attack vector" matches "attack_vectors"
            "defenses" matches "defense_mechanisms"
        """
        if not name or not existing_categories:
            return None

        # Sanitize input name
        sanitized = self.sanitize_name(name)
        if not sanitized:
            return None

        # Check for exact match first
        for existing in existing_categories:
            if sanitized == existing.lower():
                return existing

        # Check for singular/plural variations
        for existing in existing_categories:
            existing_lower = existing.lower()

            # Remove trailing 's' or 'es' for plural matching
            singular_input = re.sub(r'e?s$', '', sanitized)
            singular_existing = re.sub(r'e?s$', '', existing_lower)

            if singular_input == singular_existing:
                return existing

        # Fuzzy matching for typos and similar names
        best_match = None
        best_score = 0.0

        for existing in existing_categories:
            score = SequenceMatcher(None, sanitized, existing.lower()).ratio()
            if score > best_score and score >= self.similarity_threshold:
                best_score = score
                best_match = existing

        if best_match:
            logger.debug(
                f"Fuzzy matched '{name}' to '{best_match}' (score: {best_score:.2f})"
            )
            return best_match

        return None

    def validate_and_sanitize(
        self, categories: Dict[str, str]
    ) -> Tuple[Dict[str, str], List[str], List[str]]:
        """
        Validate and sanitize a dictionary of categories.

        Args:
            categories: Dictionary mapping category names to definitions

        Returns:
            Tuple of:
            - Sanitized categories (valid only)
            - List of warnings (non-fatal issues)
            - List of errors (fatal issues)

        Examples:
            Input: {"Attack Vectors": "...", "Defense & Mitigation": "..."}
            Output: (
                {"attack_vectors": "...", "defense_mitigation": "..."},
                ["Sanitized 'Attack Vectors' to 'attack_vectors'"],
                []
            )
        """
        warnings: list[str] = []
        errors: list[str] = []
        sanitized: dict[str, str] = {}
        seen_names = set()

        if not categories:
            errors.append("No categories provided")
            return {}, warnings, errors

        if len(categories) > self.max_categories:
            warnings.append(
                f"Too many categories ({len(categories)}), maximum is {self.max_categories}"
            )

        for original_name, definition in categories.items():
            # Sanitize the name
            clean_name = self.sanitize_name(original_name)

            if not clean_name:
                warnings.append(f"Skipping invalid category: '{original_name}'")
                continue

            # Check if valid
            is_valid, error_msg = self.is_valid_name(clean_name)
            if not is_valid:
                warnings.append(
                    f"Skipping invalid category '{original_name}': {error_msg}"
                )
                continue

            # Check for duplicates
            if clean_name in seen_names:
                warnings.append(
                    f"Duplicate category detected: '{original_name}' -> '{clean_name}'"
                )
                continue

            # Warn if name was changed
            if clean_name != original_name:
                warnings.append(f"Sanitized '{original_name}' to '{clean_name}'")

            # Add to sanitized dict
            sanitized[clean_name] = definition
            seen_names.add(clean_name)

        # Check minimum count
        if len(sanitized) < self.min_categories:
            errors.append(
                f"Only {len(sanitized)} valid categories (minimum is {self.min_categories})"
            )

        return sanitized, warnings, errors

    def validate_category_set(
        self, categories: List[str]
    ) -> Tuple[List[str], List[str], List[str]]:
        """
        Validate a list of category names (for existing taxonomies).

        Args:
            categories: List of category names

        Returns:
            Tuple of (valid_categories, warnings, errors)
        """
        # Convert to dict format for validation
        category_dict = {name: f"Category: {name}" for name in categories}

        sanitized_dict, warnings, errors = self.validate_and_sanitize(category_dict)

        return list(sanitized_dict.keys()), warnings, errors
