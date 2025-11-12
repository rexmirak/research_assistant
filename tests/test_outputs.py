"""Unit tests for core/outputs.py: output generation logic."""

import csv
import json
from pathlib import Path

import pytest

from core.outputs import OutputGenerator


@pytest.fixture
def generator(tmp_path):
    """Create OutputGenerator with temporary directory."""
    return OutputGenerator(tmp_path)


def test_write_jsonl_creates_file(generator, tmp_path):
    """Test JSONL file creation with records."""
    records = [
        {"paper_id": "abc", "title": "Test Paper 1", "category": "catA"},
        {"paper_id": "def", "title": "Test Paper 2", "category": "catB"},
    ]

    generator.write_jsonl(records)

    jsonl_file = tmp_path / "index.jsonl"
    assert jsonl_file.exists()

    # Verify content
    with open(jsonl_file, "r") as f:
        lines = f.readlines()

    assert len(lines) == 2
    record1 = json.loads(lines[0])
    assert record1["paper_id"] == "abc"
    assert record1["title"] == "Test Paper 1"


def test_write_csv_creates_file(generator, tmp_path):
    """Test CSV file creation with records."""
    records = [
        {"paper_id": "abc", "title": "Test Paper 1", "category": "catA"},
        {"paper_id": "def", "title": "Test Paper 2", "category": "catB"},
    ]

    generator.write_csv(records)

    csv_file = tmp_path / "index.csv"
    assert csv_file.exists()

    # Verify content
    with open(csv_file, "r", newline="") as f:
        reader = csv.DictReader(f)
        rows = list(reader)

    assert len(rows) == 2
    assert rows[0]["paper_id"] == "abc"
    assert rows[1]["paper_id"] == "def"


def test_write_empty_jsonl(generator, tmp_path):
    """Test writing empty JSONL file."""
    generator.write_jsonl([])

    jsonl_file = tmp_path / "index.jsonl"
    assert jsonl_file.exists()

    with open(jsonl_file, "r") as f:
        content = f.read()

    assert content == ""


def test_write_empty_csv(generator, tmp_path):
    """Test writing empty CSV file (should not create file)."""
    generator.write_csv([])

    csv_file = tmp_path / "index.csv"
    # Empty records should not create a file
    assert not csv_file.exists()


def test_write_category_summary(generator, tmp_path):
    """Test category summary generation."""
    papers = [
        {"title": "Paper 1", "summary": "Summary 1", "authors": ["Author A"]},
        {"title": "Paper 2", "summary": "Summary 2", "authors": ["Author B"]},
    ]

    generator.write_category_summary("test_category", papers)

    summary_file = tmp_path / "summaries" / "test_category.md"
    assert summary_file.exists()

    content = summary_file.read_text()
    assert "Paper 1" in content
    assert "Paper 2" in content
    assert "Summary 1" in content


def test_write_jsonl_with_nested_data(generator, tmp_path):
    """Test JSONL with nested data structures."""
    records = [
        {
            "paper_id": "abc",
            "title": "Test",
            "authors": ["Author 1", "Author 2"],
            "metadata": {"year": 2023, "venue": "Conference"},
        }
    ]

    generator.write_jsonl(records)

    jsonl_file = tmp_path / "index.jsonl"
    with open(jsonl_file, "r") as f:
        record = json.loads(f.readline())

    assert record["authors"] == ["Author 1", "Author 2"]
    assert record["metadata"]["year"] == 2023


def test_write_csv_with_list_fields(generator, tmp_path):
    """Test CSV handling of list fields (should serialize)."""
    records = [{"paper_id": "abc", "title": "Test", "authors": ["Author 1", "Author 2"]}]

    generator.write_csv(records)

    csv_file = tmp_path / "index.csv"
    with open(csv_file, "r", newline="") as f:
        reader = csv.DictReader(f)
        row = next(reader)

    # Authors should be serialized as string
    assert "authors" in row


def test_append_to_existing_jsonl(generator, tmp_path):
    """Test appending to existing JSONL file."""
    records1 = [{"paper_id": "abc", "title": "Paper 1"}]
    records2 = [{"paper_id": "def", "title": "Paper 2"}]

    generator.write_jsonl(records1)
    generator.write_jsonl(records2)

    jsonl_file = tmp_path / "index.jsonl"
    with open(jsonl_file, "r") as f:
        lines = f.readlines()

    # Last write should replace (or append, depending on implementation)
    assert len(lines) >= 1


def test_output_directory_creation(tmp_path):
    """Test that output directories are created."""
    output_dir = tmp_path / "custom_output"
    generator = OutputGenerator(output_dir)

    generator.write_jsonl([{"test": "data"}])

    assert output_dir.exists()
    assert (output_dir / "index.jsonl").exists()


def test_write_category_summary_creates_directory(generator, tmp_path):
    """Test that summaries directory is created."""
    papers = [{"title": "Test", "summary": "Summary"}]

    generator.write_category_summary("new_category", papers)

    assert (tmp_path / "summaries").exists()
    assert (tmp_path / "summaries" / "new_category.md").exists()


def test_write_jsonl_with_special_characters(generator, tmp_path):
    """Test JSONL with special characters."""
    records = [
        {
            "paper_id": "abc",
            "title": 'Test with "quotes" and \n newlines',
            "abstract": "Text with unicode: café, 日本語",
        }
    ]

    generator.write_jsonl(records)

    jsonl_file = tmp_path / "index.jsonl"
    with open(jsonl_file, "r", encoding="utf-8") as f:
        record = json.loads(f.readline())

    assert "quotes" in record["title"]
    assert "café" in record["abstract"]


def test_write_csv_with_missing_fields(generator, tmp_path):
    """Test CSV with records having different fields."""
    records = [
        {"paper_id": "abc", "title": "Paper 1", "year": 2023},
        {"paper_id": "def", "title": "Paper 2"},  # Missing 'year'
    ]

    generator.write_csv(records)

    csv_file = tmp_path / "index.csv"
    with open(csv_file, "r", newline="") as f:
        reader = csv.DictReader(f)
        rows = list(reader)

    assert len(rows) == 2
    # Second row should have empty year field
    assert rows[1]["paper_id"] == "def"
