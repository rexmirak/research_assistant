"""Metadata extraction using GROBID and Crossref."""

import logging
from pathlib import Path
from typing import Any, Dict, Optional

from utils.grobid_client import GrobidClient
from utils.text import clean_title, create_bibtex_key

# Optional imports with graceful degradation; add type ignores to silence static analysis
try:  # pragma: no cover
    import fitz  # type: ignore
except Exception:  # pragma: no cover
    fitz = None  # type: ignore

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
        self.use_grobid = True

    def extract(self, pdf_path: Path) -> Dict[str, Any]:
        """
        Extract metadata from PDF.

        Args:
            pdf_path: Path to PDF file

        Returns:
            Dictionary with metadata fields
        """
        # Try GROBID first unless disabled
        metadata = None
        if self.use_grobid:
            metadata = self.grobid_client.process_pdf(pdf_path)

        grobid_status = None
        if metadata and "grobid_status" in metadata:
            grobid_status = metadata.get("grobid_status")

        if not metadata or not metadata.get("title"):
            # Fallback to PDF internal metadata BUT preserve grobid_status if we had one
            fallback = self._extract_from_pdf_metadata(pdf_path)
            # If still missing authors, try a light-weight first-page heuristic
            if not fallback.get("authors"):
                inferred = self._infer_title_authors_from_first_page(pdf_path)
                if inferred:
                    if inferred.get("title") and not fallback.get("title"):
                        fallback["title"] = inferred["title"]
                    if inferred.get("authors"):
                        fallback["authors"] = inferred["authors"]
            if grobid_status and "grobid_status" not in fallback:
                fallback["grobid_status"] = grobid_status
            metadata = fallback

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
        try:
            if fitz is None:
                raise ImportError("PyMuPDF (fitz) not available in environment")
            doc = fitz.open(pdf_path)  # type: ignore[attr-defined]
            meta = doc.metadata
            doc.close()

            raw_title = meta.get("title") or ""
            cleaned_title = clean_title(raw_title) if raw_title else ""
            if not cleaned_title:
                cleaned_title = pdf_path.stem

            authors: list[str] = []
            raw_author = meta.get("author") or ""
            if raw_author:
                # Split common delimiters
                parts = [a.strip() for a in raw_author.replace(";", ",").split(",") if a.strip()]
                if parts:
                    authors = parts

            return {
                "title": cleaned_title,
                "authors": authors,
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
            from habanero import Crossref  # type: ignore

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

    def _infer_title_authors_from_first_page(self, pdf_path: Path) -> Optional[Dict[str, Any]]:
        """Heuristic extraction of title and authors from first page text layout.

        Returns a dict with optional keys 'title' and 'authors'.
        """
        try:
            if fitz is None:
                return None
            doc = fitz.open(pdf_path)  # type: ignore[attr-defined]
            if doc.page_count == 0:
                doc.close()
                return None
            page = doc.load_page(0)
            blocks = page.get_text("blocks") or []  # list of (x0, y0, x1, y1, text, block_no, ...)
            doc.close()

            # Sort by vertical position
            blocks_sorted = sorted(blocks, key=lambda b: (b[1], b[0]))
            top_blocks = []
            if blocks_sorted:
                page_height = max(b[3] for b in blocks_sorted)
                cutoff = page_height * 0.35  # top 35% of the page
                for b in blocks_sorted:
                    y0, y1, text = b[1], b[3], (b[4] or "").strip()
                    if not text:
                        continue
                    if y1 <= cutoff:
                        top_blocks.append(text)

            top_text = "\n".join(top_blocks).strip()
            if not top_text:
                return None

            lines = [ln.strip() for ln in top_text.splitlines() if ln.strip()]
            if not lines:
                return None

            # Title heuristic: first non-all-caps/short line with enough length
            title_candidate = None
            for ln in lines[:8]:  # inspect first few lines
                if len(ln) >= 8 and len(ln) <= 180:
                    # avoid lines that are mostly uppercase (section headers)
                    letters = [c for c in ln if c.isalpha()]
                    if not letters or (sum(1 for c in letters if c.isupper()) / max(1, len(letters))) > 0.9:
                        continue
                    title_candidate = ln
                    break

            authors = []
            # Authors heuristic: next line(s) containing comma/and-separated capitalized tokens
            for ln in lines[1:6]:
                if any(tok in ln.lower() for tok in ["abstract", "introduction", "keywords", "university", "institute", "department"]):
                    break
                # Split by common separators
                parts = [p.strip() for p in ln.replace(";", ",").replace(" and ", ",").split(",") if p.strip()]
                candidate_names = []
                for p in parts:
                    # naive name check: 2-4 words with capitalization
                    tokens = p.split()
                    if 1 < len(tokens) <= 4 and sum(t[0].isupper() for t in tokens if t) >= 2:
                        candidate_names.append(p)
                if candidate_names:
                    authors = candidate_names
                    break

            result: Dict[str, Any] = {}
            if title_candidate:
                result["title"] = clean_title(title_candidate)
            if authors:
                result["authors"] = authors
            return result or None
        except Exception as e:
            logger.debug(f"First-page inference failed for {pdf_path.name}: {e}")
            return None
