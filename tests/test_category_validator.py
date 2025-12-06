"""Unit tests for core/category_validator.py"""

import pytest
from core.category_validator import CategoryValidator


@pytest.fixture
def validator():
    """Create validator with default settings."""
    return CategoryValidator(min_categories=3, max_categories=25, similarity_threshold=0.8)


class TestSanitizeName:
    """Tests for sanitize_name() method."""

    def test_basic_sanitization(self, validator):
        """Test basic name sanitization."""
        assert validator.sanitize_name("Attack Vectors") == "attack_vectors"
        assert validator.sanitize_name("Defense Mechanisms") == "defense_mechanisms"

    def test_special_characters_removed(self, validator):
        """Test that special characters are removed."""
        assert validator.sanitize_name("Defense & Mitigation") == "defense_mitigation"
        assert validator.sanitize_name("Attack/Defense") == "attack_defense"
        assert validator.sanitize_name("Test:Category") == "testcategory"  # Colon removed, not replaced

    def test_multiple_separators(self, validator):
        """Test multiple separators collapsed to single underscore."""
        assert validator.sanitize_name("Attack   Vectors") == "attack_vectors"
        assert validator.sanitize_name("Defense - - Mechanisms") == "defense_mechanisms"
        assert validator.sanitize_name("Test__Category") == "test_category"

    def test_path_traversal_removed(self, validator):
        """Test that path traversal attempts are sanitized."""
        assert validator.sanitize_name("../../etc/passwd") == "etc_passwd"
        assert validator.sanitize_name("..\\..\\windows\\system32") == "windows_system32"

    def test_leading_trailing_underscores(self, validator):
        """Test leading/trailing underscores are removed."""
        assert validator.sanitize_name("_category_") == "category"
        assert validator.sanitize_name("__test__") == "test"

    def test_length_truncation(self, validator):
        """Test names longer than 50 chars are truncated."""
        long_name = "a" * 100
        result = validator.sanitize_name(long_name)
        assert len(result) <= 50

    def test_empty_string(self, validator):
        """Test empty strings return empty."""
        assert validator.sanitize_name("") == ""
        assert validator.sanitize_name("   ") == ""

    def test_non_string_input(self, validator):
        """Test non-string input returns empty."""
        assert validator.sanitize_name(None) == ""
        assert validator.sanitize_name(123) == ""


class TestIsValidName:
    """Tests for is_valid_name() method."""

    def test_valid_names(self, validator):
        """Test valid category names."""
        valid, msg = validator.is_valid_name("attack_vectors")
        assert valid is True
        assert msg == ""

        valid, msg = validator.is_valid_name("defense_mechanisms")
        assert valid is True

    def test_empty_name_invalid(self, validator):
        """Test empty name is invalid."""
        valid, msg = validator.is_valid_name("")
        assert valid is False
        assert "empty" in msg.lower()

    def test_reserved_names_invalid(self, validator):
        """Test reserved system names are invalid."""
        reserved = ["con", "prn", "aux", "nul", "quarantined", "repeated"]
        for name in reserved:
            valid, msg = validator.is_valid_name(name)
            assert valid is False
            assert "reserved" in msg.lower()

    def test_too_short_invalid(self, validator):
        """Test names shorter than 2 chars are invalid."""
        valid, msg = validator.is_valid_name("a")
        assert valid is False
        assert "short" in msg.lower()

    def test_too_long_invalid(self, validator):
        """Test names longer than 50 chars are invalid."""
        long_name = "a" * 51
        valid, msg = validator.is_valid_name(long_name)
        assert valid is False
        assert "long" in msg.lower()

    def test_must_contain_letter(self, validator):
        """Test name must contain at least one letter."""
        valid, msg = validator.is_valid_name("123456")
        assert valid is False
        assert "letter" in msg.lower()

    def test_dangerous_patterns_invalid(self, validator):
        """Test dangerous patterns are caught."""
        # Path traversal
        valid, msg = validator.is_valid_name("../etc")
        assert valid is False

        # Path separators
        valid, msg = validator.is_valid_name("test/path")
        assert valid is False


class TestMatchToExisting:
    """Tests for match_to_existing() method."""

    def test_exact_match(self, validator):
        """Test exact matches are found."""
        existing = ["attack_vectors", "defense_mechanisms"]
        assert validator.match_to_existing("attack_vectors", existing) == "attack_vectors"

    def test_case_insensitive_match(self, validator):
        """Test case-insensitive matching."""
        existing = ["attack_vectors", "defense_mechanisms"]
        assert validator.match_to_existing("Attack_Vectors", existing) == "attack_vectors"
        assert validator.match_to_existing("DEFENSE_MECHANISMS", existing) == "defense_mechanisms"

    def test_singular_plural_matching(self, validator):
        """Test singular/plural variations are matched."""
        existing = ["attack_vectors", "defense_mechanisms"]
        assert validator.match_to_existing("attack_vector", existing) == "attack_vectors"
        assert validator.match_to_existing("defense_mechanism", existing) == "defense_mechanisms"

    def test_fuzzy_matching_typos(self, validator):
        """Test fuzzy matching catches typos."""
        existing = ["attack_vectors", "defense_mechanisms"]
        # "attack_vectros" (typo) should match "attack_vectors"
        result = validator.match_to_existing("attack_vectros", existing)
        assert result == "attack_vectors"

    def test_fuzzy_matching_threshold(self, validator):
        """Test fuzzy matching respects similarity threshold."""
        existing = ["attack_vectors"]
        # Very different word should not match
        result = validator.match_to_existing("zebra_category", existing)
        assert result is None

    def test_no_match_returns_none(self, validator):
        """Test no match returns None."""
        existing = ["attack_vectors"]
        assert validator.match_to_existing("completely_different", existing) is None

    def test_empty_inputs(self, validator):
        """Test empty inputs return None."""
        assert validator.match_to_existing("", ["attack_vectors"]) is None
        assert validator.match_to_existing("test", []) is None
        assert validator.match_to_existing("", []) is None

    def test_best_match_selection(self, validator):
        """Test best match is selected when multiple are similar."""
        existing = ["attack_techniques", "attack_vectors"]
        # Should match to most similar
        result = validator.match_to_existing("attack_technique", existing)
        assert result == "attack_techniques"


class TestValidateAndSanitize:
    """Tests for validate_and_sanitize() method."""

    def test_valid_categories(self, validator):
        """Test validation of valid categories."""
        categories = {
            "Attack Vectors": "Methods of attack",
            "Defense Mechanisms": "Defense strategies",
            "System Security": "Overall security",
        }
        sanitized, warnings, errors = validator.validate_and_sanitize(categories)

        assert len(sanitized) == 3
        assert "attack_vectors" in sanitized
        assert "defense_mechanisms" in sanitized
        assert "system_security" in sanitized
        assert len(errors) == 0

    def test_sanitization_warnings(self, validator):
        """Test warnings are generated for sanitized names."""
        categories = {
            "Attack Vectors": "Test",
            "Defense & Mitigation": "Test",
        }
        sanitized, warnings, errors = validator.validate_and_sanitize(categories)

        assert len(warnings) >= 2
        assert any("attack_vectors" in w.lower() for w in warnings)
        assert any("defense_mitigation" in w.lower() for w in warnings)

    def test_reserved_names_skipped(self, validator):
        """Test reserved names are skipped with warnings."""
        categories = {
            "quarantined": "Test",
            "Attack Vectors": "Test",
        }
        sanitized, warnings, errors = validator.validate_and_sanitize(categories)

        assert "quarantined" not in sanitized
        assert "attack_vectors" in sanitized
        assert len(warnings) >= 1
        assert any("reserved" in w.lower() for w in warnings)

    def test_duplicate_detection(self, validator):
        """Test duplicate categories are detected."""
        categories = {
            "Attack Vectors": "Test 1",
            "Attack_Vectors": "Test 2",  # Duplicate after sanitization
        }
        sanitized, warnings, errors = validator.validate_and_sanitize(categories)

        assert len(sanitized) == 1
        assert "attack_vectors" in sanitized
        assert any("duplicate" in w.lower() for w in warnings)

    def test_minimum_categories_error(self, validator):
        """Test error when below minimum categories."""
        categories = {
            "Category 1": "Test",
            "Category 2": "Test",
        }
        sanitized, warnings, errors = validator.validate_and_sanitize(categories)

        assert len(errors) > 0
        assert any("minimum" in e.lower() for e in errors)

    def test_maximum_categories_warning(self):
        """Test warning when exceeding maximum categories."""
        validator = CategoryValidator(max_categories=5)
        categories = {f"Category {i}": "Test" for i in range(10)}

        sanitized, warnings, errors = validator.validate_and_sanitize(categories)

        assert any("many" in w.lower() for w in warnings)

    def test_empty_categories(self, validator):
        """Test error when no categories provided."""
        sanitized, warnings, errors = validator.validate_and_sanitize({})

        assert len(errors) > 0
        assert any("no categories" in e.lower() for e in errors)

    def test_invalid_category_names_skipped(self, validator):
        """Test invalid names are skipped with warnings."""
        categories = {
            "Valid Category": "Test",
            "": "Empty name",
            "a": "Too short",
            "123": "No letters",
        }
        sanitized, warnings, errors = validator.validate_and_sanitize(categories)

        assert len(sanitized) == 1
        assert "valid_category" in sanitized
        assert len(warnings) >= 3  # Three invalid categories


class TestValidateCategorySet:
    """Tests for validate_category_set() method."""

    def test_valid_category_list(self, validator):
        """Test validation of valid category list."""
        categories = ["attack_vectors", "defense_mechanisms", "system_security"]
        valid, warnings, errors = validator.validate_category_set(categories)

        assert len(valid) == 3
        assert "attack_vectors" in valid
        assert len(errors) == 0

    def test_invalid_categories_filtered(self, validator):
        """Test invalid categories are filtered out."""
        categories = ["attack_vectors", "quarantined", "a", "123"]
        valid, warnings, errors = validator.validate_category_set(categories)

        assert len(valid) == 1
        assert "attack_vectors" in valid
        assert len(warnings) > 0


class TestEdgeCases:
    """Test edge cases and boundary conditions."""

    def test_unicode_characters(self, validator):
        """Test unicode characters are handled."""
        result = validator.sanitize_name("Défense Mécanisms")
        assert result  # Should produce some sanitized output
        valid, _ = validator.is_valid_name(result)
        assert valid or not result  # Either valid or empty

    def test_very_long_category_name(self, validator):
        """Test extremely long names are handled."""
        long_name = "a" * 200
        result = validator.sanitize_name(long_name)
        assert len(result) <= 50

    def test_all_special_characters(self, validator):
        """Test name with only special characters."""
        result = validator.sanitize_name("!@#$%^&*()")
        assert result == ""

    def test_custom_thresholds(self):
        """Test validator with custom min/max settings."""
        validator = CategoryValidator(min_categories=1, max_categories=5)
        assert validator.min_categories == 1
        assert validator.max_categories == 5

        # Should accept 1 category
        categories = {"test_category": "Test"}
        sanitized, warnings, errors = validator.validate_and_sanitize(categories)
        assert len(errors) == 0
        assert len(sanitized) == 1

    def test_similarity_threshold_affects_matching(self):
        """Test different similarity thresholds affect fuzzy matching."""
        # Strict threshold
        validator_strict = CategoryValidator(similarity_threshold=0.9)
        # Lenient threshold
        validator_lenient = CategoryValidator(similarity_threshold=0.6)

        existing = ["attack_vectors"]
        test_name = "attack_vektors"  # Similar but not very close

        # Strict might not match
        result_strict = validator_strict.match_to_existing(test_name, existing)

        # Lenient should match
        result_lenient = validator_lenient.match_to_existing(test_name, existing)

        # At least one should behave differently or lenient should match
        assert result_lenient == "attack_vectors" or result_strict != result_lenient
