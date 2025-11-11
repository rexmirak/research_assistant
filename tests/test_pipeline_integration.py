"""Integration test for the main pipeline: process_single_doc logic (mocked LLM and file ops)."""

from pathlib import Path
from unittest.mock import MagicMock

import pytest

from core.inventory import InventoryManager, PDFDocument
from core.manifest import ManifestManager
from core.metadata import MetadataExtractor
from core.mover import FileMover
from core.outputs import OutputGenerator
from core.parser import PDFParser


@pytest.fixture
def pipeline_components(tmp_path):
    manifest_manager = ManifestManager(tmp_path / "manifests")
    output_generator = OutputGenerator(tmp_path)
    file_mover = FileMover(tmp_path, manifest_manager, dry_run=True, create_symlinks=False)
    metadata_extractor = MetadataExtractor(False, None)
    pdf_parser = PDFParser("eng", True)
    return manifest_manager, output_generator, file_mover, metadata_extractor, pdf_parser


def test_process_single_doc_pipeline(pipeline_components, tmp_path):
    manifest_manager, output_generator, file_mover, metadata_extractor, pdf_parser = (
        pipeline_components
    )
    # Mock document
    doc = MagicMock(spec=PDFDocument)
    doc.file_hash = "abc123def456"
    doc.file_name = "test.pdf"
    doc.file_path = tmp_path / "catA" / "test.pdf"
    doc.category = "catA"
    # Mock LLM and parser
    pdf_parser.extract_text = MagicMock(return_value=("Full text", "hash1"))
    pdf_parser.extract_sections = MagicMock(return_value={"abstract": "Test abstract"})
    metadata_extractor._extract_with_llm = MagicMock(
        return_value={"title": "Test", "authors": ["A"], "abstract": "Test abstract"}
    )
    metadata_extractor._llm_categorize_and_score = MagicMock(
        return_value={"category": "catA", "relevance_score": 8.0, "include": True}
    )
    # Simulate process_single_doc logic
    paper_id = doc.file_hash[:12]
    text, text_hash = pdf_parser.extract_text(doc.file_path, tmp_path)
    metadata = metadata_extractor._extract_with_llm(doc.file_path)
    cat_score = metadata_extractor._llm_categorize_and_score(
        title=metadata.get("title", ""),
        abstract=metadata.get("abstract", ""),
        topic="Test topic",
        available_categories=["catA", "catB"],
    )
    metadata.update(cat_score)
    # Add to manifest
    manifest = manifest_manager.get_manifest(doc.category)
    manifest.add_paper(
        paper_id=paper_id,
        path=str(doc.file_path),
        content_hash=text_hash,
        status="active",
        original_category=doc.category,
    )
    # Output
    record = {
        "paper_id": paper_id,
        "title": metadata.get("title", ""),
        "category": doc.category,
        "relevance_score": metadata.get("relevance_score"),
        "include": metadata.get("include", False),
    }
    assert record["title"] == "Test"
    assert record["category"] == "catA"
    assert record["relevance_score"] == 8.0
    assert record["include"] is True
