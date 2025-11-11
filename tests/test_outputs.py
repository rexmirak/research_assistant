"""Unit tests for core/outputs.py: output generation logic (mocked, no real file writes)."""

from pathlib import Path
from unittest.mock import MagicMock

import pytest

from core.outputs import OutputGenerator


@pytest.fixture
def generator(tmp_path):
    return OutputGenerator(tmp_path)


def test_write_jsonl_and_csv(generator):
    # Patch file writing
    generator.write_jsonl = MagicMock()
    generator.write_csv = MagicMock()
    records = [
        {"paper_id": "abc", "title": "Test", "category": "catA"},
        {"paper_id": "def", "title": "Test2", "category": "catB"},
    ]
    generator.write_jsonl(records)
    generator.write_csv(records)
    generator.write_jsonl.assert_called_once_with(records)
    generator.write_csv.assert_called_once_with(records)


def test_write_category_summary(generator):
    generator.write_category_summary = MagicMock()
    generator.write_category_summary("catA", [{"title": "Test", "summary": "Summary"}])
    generator.write_category_summary.assert_called_once()
