"""PDF text extraction with OCR fallback."""

import logging
import subprocess
from pathlib import Path
from typing import Dict, Optional, Tuple

import fitz  # PyMuPDF

from utils.hash import text_hash
from utils.text import extract_abstract, extract_introduction, normalize_text

logger = logging.getLogger(__name__)


class PDFParser:
    """Extract text from PDFs with OCR fallback."""

    def __init__(self, ocr_language: str = "eng", skip_ocr_if_text: bool = True):
        """
        Initialize PDF parser.

        Args:
            ocr_language: Tesseract language code
            skip_ocr_if_text: Skip OCR if text extraction succeeds
        """
        self.ocr_language = ocr_language
        self.skip_ocr_if_text = skip_ocr_if_text

    def extract_text(self, pdf_path: Path, cache_dir: Optional[Path] = None) -> Tuple[str, str]:
        """
        Extract text from PDF.

        Args:
            pdf_path: Path to PDF file
            cache_dir: Directory to cache OCR results

        Returns:
            Tuple of (full_text, text_hash)
        """
        # Try PyMuPDF first
        text = self._extract_with_pymupdf(pdf_path)

        # Check if we got meaningful text
        if len(text.strip()) < 100:
            logger.info(f"Low text content in {pdf_path.name}, attempting OCR")
            text = self._extract_with_ocr(pdf_path, cache_dir)

        # Fallback to pdfminer if still poor
        if len(text.strip()) < 100:
            logger.info(f"OCR failed for {pdf_path.name}, trying pdfminer")
            text = self._extract_with_pdfminer(pdf_path)

        # Normalize and hash
        normalized = normalize_text(text)
        content_hash = text_hash(normalized)

        return normalized, content_hash

    def extract_sections(self, text: str) -> Dict[str, Optional[str]]:
        """
        Extract key sections from text.

        Args:
            text: Full paper text

        Returns:
            Dictionary with abstract, introduction, etc.
        """
        return {
            "abstract": extract_abstract(text),
            "introduction": extract_introduction(text),
            "full_text": text,
        }

    def _extract_with_pymupdf(self, pdf_path: Path) -> str:
        """Extract text using PyMuPDF."""
        try:
            doc = fitz.open(pdf_path)
            text_blocks = []

            for page in doc:
                # Extract text with layout preservation
                text_blocks.append(page.get_text("text"))

            doc.close()
            return "\n\n".join(text_blocks)

        except Exception as e:
            logger.error(f"PyMuPDF extraction failed for {pdf_path.name}: {e}")
            return ""

    def _extract_with_pdfminer(self, pdf_path: Path) -> str:
        """Extract text using pdfminer.six."""
        try:
            from pdfminer.high_level import extract_text as pdfminer_extract

            return pdfminer_extract(pdf_path)
        except Exception as e:
            logger.error(f"pdfminer extraction failed for {pdf_path.name}: {e}")
            return ""

    def _extract_with_ocr(self, pdf_path: Path, cache_dir: Optional[Path] = None) -> str:
        """Extract text using OCR (ocrmypdf + PyMuPDF)."""
        if not cache_dir:
            logger.warning("No cache_dir provided for OCR, skipping")
            return ""

        try:
            # Create OCR output path
            ocr_cache = cache_dir / "ocr"
            ocr_cache.mkdir(parents=True, exist_ok=True)
            ocr_output = ocr_cache / f"{pdf_path.stem}_ocr.pdf"

            # Check if OCR version already exists
            if ocr_output.exists():
                logger.debug(f"Using cached OCR for {pdf_path.name}")
                return self._extract_with_pymupdf(ocr_output)

            # Run ocrmypdf
            logger.info(f"Running OCR on {pdf_path.name}")
            result = subprocess.run(
                [
                    "ocrmypdf",
                    "--language",
                    self.ocr_language,
                    "--skip-text" if self.skip_ocr_if_text else "--force-ocr",
                    "--optimize",
                    "0",
                    str(pdf_path),
                    str(ocr_output),
                ],
                capture_output=True,
                text=True,
                timeout=300,
            )

            if result.returncode == 0:
                return self._extract_with_pymupdf(ocr_output)
            else:
                logger.warning(f"OCR failed for {pdf_path.name}: {result.stderr}")
                return ""

        except Exception as e:
            logger.error(f"OCR extraction failed for {pdf_path.name}: {e}")
            return ""

    def get_page_count(self, pdf_path: Path) -> int:
        """Get number of pages in PDF."""
        try:
            doc = fitz.open(pdf_path)
            count: int = doc.page_count
            doc.close()
            return count
        except Exception as e:
            logger.error(f"Failed to get page count for {pdf_path.name}: {e}")
            return 0
