"""Metadata extraction using GROBID and Crossref."""

import logging
from pathlib import Path
from typing import Any, Dict, Optional

from utils.grobid_client import GrobidClient
from utils.text import clean_title, create_bibtex_key

logger = logging.getLogger(__name__)


class MetadataExtractor:
    """Extract structured metadata from PDFs."""

    def __init__(
        self,
        grobid_url: str = "http://localhost:8070",
        use_crossref: bool = True,
        crossref_email: Optional[str] = None,
    ):
        """
        Initialize metadata extractor.

        Args:
            grobid_url: GROBID service URL
            use_crossref: Enable Crossref enrichment
            crossref_email: Email for Crossref polite pool
        """
        self.grobid_client = GrobidClient(grobid_url)
        self.use_crossref = use_crossref
        self.crossref_email = crossref_email

    def extract(self, pdf_path: Path) -> Dict[str, Any]:
        """
        Extract metadata from PDF.

        Args:
            pdf_path: Path to PDF file

        Returns:
            Dictionary with metadata fields
        """
        # Try GROBID first
        metadata = self.grobid_client.process_pdf(pdf_path)

        if not metadata or not metadata.get("title"):
            # Fallback to PDF internal metadata
            metadata = self._extract_from_pdf_metadata(pdf_path)

        # Enrich with Crossref if DOI available
        if self.use_crossref and metadata.get("doi"):
            crossref_data = self._enrich_with_crossref(metadata["doi"])
            if crossref_data:
                metadata.update(crossref_data)

        # Generate BibTeX
        bibtex = self._generate_bibtex(metadata, pdf_path)
        metadata["bibtex"] = bibtex

        return metadata

    def _extract_from_pdf_metadata(self, pdf_path: Path) -> Dict[str, Any]:
        """Extract basic metadata from PDF properties."""
        import fitz

        try:
            doc = fitz.open(pdf_path)
            meta = doc.metadata
            doc.close()

            return {
                "title": clean_title(meta.get("title", pdf_path.stem)),
                "authors": [meta.get("author", "Unknown")] if meta.get("author") else [],
                "year": meta.get("creationDate", "")[:4] if meta.get("creationDate") else None,
                "venue": None,
                "doi": None,
                "abstract": None,
            }
        except Exception as e:
            logger.error(f"PDF metadata extraction failed for {pdf_path.name}: {e}")
            return {
                "title": pdf_path.stem,
                "authors": [],
                "year": None,
                "venue": None,
                "doi": None,
                "abstract": None,
            }

    def _enrich_with_crossref(self, doi: str) -> Optional[Dict[str, Any]]:
        """Enrich metadata using Crossref API."""
        try:
            from habanero import Crossref

            cr = Crossref(mailto=self.crossref_email)
            result = cr.works(ids=doi)

            if result and "message" in result:
                msg = result["message"]

                return {
                    "title": msg.get("title", [None])[0],
                    "year": str(
                        msg.get("published-print", {}).get("date-parts", [[None]])[0][0]
                        or msg.get("published-online", {}).get("date-parts", [[None]])[0][0]
                    ),
                    "venue": msg.get("container-title", [None])[0],
                    "authors": [
                        f"{a.get('given', '')} {a.get('family', '')}".strip()
                        for a in msg.get("author", [])
                    ],
                }

        except Exception as e:
            logger.warning(f"Crossref enrichment failed for DOI {doi}: {e}")

        return None

    def _generate_bibtex(self, metadata: Dict[str, Any], pdf_path: Path) -> str:
        """Generate BibTeX citation."""
        title = metadata.get("title", pdf_path.stem)
        authors = metadata.get("authors", [])
        year = metadata.get("year", "n.d.")
        venue = metadata.get("venue", "")
        doi = metadata.get("doi", "")

        # Create citation key
        key = create_bibtex_key(authors, str(year), title)

        # Format authors
        author_str = " and ".join(authors) if authors else "Unknown"

        # Build BibTeX entry
        bibtex_lines = [f"@article{{{key},", f"  title={{{title}}},"]

        if authors:
            bibtex_lines.append(f"  author={{{author_str}}},")

        if year and year != "n.d.":
            bibtex_lines.append(f"  year={{{year}}},")

        if venue:
            bibtex_lines.append(f"  journal={{{venue}}},")

        if doi:
            bibtex_lines.append(f"  doi={{{doi}}},")

        bibtex_lines.append("}")

        return "\n".join(bibtex_lines)
