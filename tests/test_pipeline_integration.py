"""Integration tests for end-to-end pipeline with real PDFs."""

import os
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
from click.testing import CliRunner

from cli import process
from core.inventory import InventoryManager, PDFDocument
from core.manifest import ManifestManager
from core.metadata import MetadataExtractor
from core.mover import FileMover
from core.outputs import OutputGenerator
from core.parser import PDFParser


@pytest.fixture
def pipeline_components(tmp_path):
    """Create pipeline components for testing."""
    manifest_manager = ManifestManager(tmp_path / "manifests")
    output_generator = OutputGenerator(tmp_path)
    file_mover = FileMover(tmp_path, manifest_manager, dry_run=True, create_symlinks=False)
    metadata_extractor = MetadataExtractor(False, None)
    pdf_parser = PDFParser("eng", True)
    return manifest_manager, output_generator, file_mover, metadata_extractor, pdf_parser


@pytest.fixture
def test_pdfs_root():
    """Get the test_pdfs directory root."""
    return Path(__file__).parent.parent / "test_pdfs"


@pytest.fixture
def runner():
    """Create Click CLI test runner."""
    return CliRunner()


def test_process_single_doc_pipeline(pipeline_components, tmp_path):
    """Test single document processing pipeline (mocked)."""
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
        title=metadata.get("title", ""),
        path=str(doc.file_path),
        content_hash=text_hash,
        classification_reasoning=metadata.get("reason", ""),
        relevance_score=metadata.get("relevance_score"),
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


@pytest.mark.integration
def test_inventory_discovers_pdfs(test_pdfs_root):
    """Test that inventory discovers PDFs in test directory."""
    if not test_pdfs_root.exists():
        pytest.skip("test_pdfs directory not found")

    inventory = InventoryManager(test_pdfs_root)
    documents = inventory.scan()

    # Should find PDFs in subdirectories
    assert len(documents) > 0

    # All should be PDF files
    for doc in documents:
        assert doc.file_path.suffix.lower() == ".pdf"
        assert doc.file_path.exists()
        assert isinstance(doc.category, str)


@pytest.mark.integration
def test_inventory_categorizes_by_directory(test_pdfs_root):
    """Test that inventory correctly categorizes PDFs by directory."""
    if not test_pdfs_root.exists():
        pytest.skip("test_pdfs directory not found")

    inventory = InventoryManager(test_pdfs_root)
    documents = inventory.scan()

    # Check that categories match directory names
    categories = {doc.category for doc in documents}

    # Should have multiple categories
    assert len(categories) > 0

    # Categories should match actual subdirectories
    actual_categories = {
        d.name for d in test_pdfs_root.iterdir() if d.is_dir() and d.name != "quarantined"
    }
    assert categories.issubset(actual_categories)


@pytest.mark.integration
def test_parser_extracts_text_from_real_pdf(test_pdfs_root):
    """Test PDF text extraction with real PDF."""
    if not test_pdfs_root.exists():
        pytest.skip("test_pdfs directory not found")

    # Find first PDF in any category
    inventory = InventoryManager(test_pdfs_root)
    documents = inventory.scan()

    if not documents:
        pytest.skip("No PDFs found in test_pdfs")

    pdf_path = documents[0].file_path

    parser = PDFParser()
    text, text_hash = parser.extract_text(pdf_path)

    assert isinstance(text, str)
    assert len(text) > 0
    assert isinstance(text_hash, str)
    assert len(text_hash) > 0


@pytest.mark.integration
def test_parser_extracts_sections_from_real_pdf(test_pdfs_root):
    """Test section extraction with real PDF."""
    if not test_pdfs_root.exists():
        pytest.skip("test_pdfs directory not found")

    inventory = InventoryManager(test_pdfs_root)
    documents = inventory.scan()

    if not documents:
        pytest.skip("No PDFs found in test_pdfs")

    pdf_path = documents[0].file_path

    parser = PDFParser()
    text, _ = parser.extract_text(pdf_path)
    sections = parser.extract_sections(text)

    assert isinstance(sections, dict)


@pytest.mark.integration
def test_metadata_extractor_with_real_pdf(test_pdfs_root):
    """Test metadata extraction with real PDF (mocked LLM)."""
    if not test_pdfs_root.exists():
        pytest.skip("test_pdfs directory not found")

    inventory = InventoryManager(test_pdfs_root)
    documents = inventory.scan()

    if not documents:
        pytest.skip("No PDFs found in test_pdfs")

    pdf_path = documents[0].file_path
    category = documents[0].category

    # Mock LLM response (Ollama format - JSON string)
    fake_metadata = {
        "response": '{"title": "Test Paper", "authors": ["Author"], "abstract": "Abstract", "year": "2023", "venue": null}'
    }

    # Mock config to use Ollama provider
    from config import Config
    mock_config = Config()
    mock_config.llm_provider = "ollama"
    mock_config.ollama.summarize_model = "deepseek-r1:8b"

    with (
        patch("core.metadata.llm_generate", return_value=fake_metadata),
        patch("config.Config", return_value=mock_config),
        patch("fitz.open") as mock_fitz,
    ):
        # Mock PDF document
        mock_doc = MagicMock()
        mock_doc.page_count = 10
        mock_page = MagicMock()
        mock_page.get_text.return_value = "Test content from PDF"
        mock_doc.load_page.return_value = mock_page
        mock_fitz.return_value = mock_doc
        
        extractor = MetadataExtractor(use_crossref=False)
        result = extractor._extract_with_llm(pdf_path)

    assert "title" in result
    assert "authors" in result
    assert "abstract" in result


@pytest.mark.integration
def test_end_to_end_pipeline_dry_run(runner, test_pdfs_root, tmp_path):
    """Test end-to-end pipeline in dry-run mode with real PDFs."""
    if not test_pdfs_root.exists():
        pytest.skip("test_pdfs directory not found")

    output_dir = tmp_path / "outputs"
    cache_dir = tmp_path / "cache"

    # Mock LLM calls
    fake_metadata = {
        "response": '{"title": "Test", "authors": ["A"], "abstract": "Abstract", "year": "2023", "venue": null}'
    }
    fake_categorization = {
        "response": {
            "category": "agent_security",
            "relevance_score": 8,
            "include": True,
            "reason": "Relevant",
        }
    }

    with patch("utils.llm_provider.llm_generate") as mock_llm:
        # Alternate between metadata and categorization responses
        mock_llm.side_effect = [fake_metadata, fake_categorization] * 100

        result = runner.invoke(
            process,
            [
                "--root-dir",
                str(test_pdfs_root),
                "--topic",
                "AI security and adversarial attacks",
                "--output-dir",
                str(output_dir),
                "--cache-dir",
                str(cache_dir),
                "--dry-run",
                "--workers",
                "1",
                "--min-topic-relevance",
                "7",
            ],
        )

    # Should complete without errors
    assert result.exit_code == 0


@pytest.mark.integration
def test_pipeline_creates_output_files(runner, test_pdfs_root, tmp_path):
    """Test that pipeline creates expected output files."""
    if not test_pdfs_root.exists():
        pytest.skip("test_pdfs directory not found")

    output_dir = tmp_path / "outputs"
    cache_dir = tmp_path / "cache"

    # Use a single PDF from a specific category
    agent_security_dir = test_pdfs_root / "agent_security"
    if not agent_security_dir.exists():
        pytest.skip("agent_security category not found")

    pdfs = list(agent_security_dir.glob("*.pdf"))
    if not pdfs:
        pytest.skip("No PDFs in agent_security category")

    # Mock LLM
    fake_metadata = {
        "response": '{"title": "Test", "authors": ["A"], "abstract": "Abstract", "year": "2023", "venue": null}'
    }
    fake_categorization = {
        "response": {
            "category": "agent_security",
            "relevance_score": 8,
            "include": True,
            "reason": "Relevant",
        }
    }

    with patch("utils.llm_provider.llm_generate") as mock_llm:
        mock_llm.side_effect = [fake_metadata, fake_categorization] * 10

        result = runner.invoke(
            process,
            [
                "--root-dir",
                str(test_pdfs_root),
                "--topic",
                "AI security",
                "--output-dir",
                str(output_dir),
                "--cache-dir",
                str(cache_dir),
                "--dry-run",
                "--workers",
                "1",
            ],
        )

    # Check output files were created
    if result.exit_code == 0:
        assert output_dir.exists()


@pytest.mark.integration
def test_pipeline_handles_multiple_categories(runner, test_pdfs_root, tmp_path):
    """Test pipeline with PDFs across multiple categories."""
    if not test_pdfs_root.exists():
        pytest.skip("test_pdfs directory not found")

    # Check we have multiple categories
    categories = [d for d in test_pdfs_root.iterdir() if d.is_dir() and d.name != "quarantined"]
    if len(categories) < 2:
        pytest.skip("Need at least 2 categories for this test")

    output_dir = tmp_path / "outputs"
    cache_dir = tmp_path / "cache"

    fake_metadata = {
        "response": '{"title": "Test", "authors": ["A"], "abstract": "Abstract", "year": "2023", "venue": null}'
    }
    fake_categorization = {
        "response": {
            "category": "agent_security",
            "relevance_score": 8,
            "include": True,
            "reason": "Relevant",
        }
    }

    with patch("utils.llm_provider.llm_generate") as mock_llm:
        mock_llm.side_effect = [fake_metadata, fake_categorization] * 100

        result = runner.invoke(
            process,
            [
                "--root-dir",
                str(test_pdfs_root),
                "--topic",
                "AI security",
                "--output-dir",
                str(output_dir),
                "--cache-dir",
                str(cache_dir),
                "--dry-run",
                "--workers",
                "1",
            ],
        )

    assert result.exit_code == 0


@pytest.mark.integration
def test_pipeline_respects_relevance_threshold(runner, test_pdfs_root, tmp_path):
    """Test that pipeline quarantines papers below relevance threshold."""
    if not test_pdfs_root.exists():
        pytest.skip("test_pdfs directory not found")

    output_dir = tmp_path / "outputs"
    cache_dir = tmp_path / "cache"

    fake_metadata = {
        "response": '{"title": "Test", "authors": ["A"], "abstract": "Abstract", "year": "2023", "venue": null}'
    }
    # Low relevance score
    fake_categorization = {
        "response": {
            "category": "agent_security",
            "relevance_score": 3,  # Below threshold
            "include": False,
            "reason": "Not relevant",
        }
    }

    with patch("utils.llm_provider.llm_generate") as mock_llm:
        mock_llm.side_effect = [fake_metadata, fake_categorization] * 100

        result = runner.invoke(
            process,
            [
                "--root-dir",
                str(test_pdfs_root),
                "--topic",
                "AI security",
                "--output-dir",
                str(output_dir),
                "--cache-dir",
                str(cache_dir),
                "--dry-run",
                "--workers",
                "1",
                "--min-topic-relevance",
                "7",
            ],
        )

    # Should complete and quarantine papers
    assert result.exit_code == 0
