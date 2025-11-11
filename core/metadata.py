"""Metadata extraction using LLM and PDF heuristics."""

import json as _json
import logging
import re
from pathlib import Path
from typing import Any, Dict, Optional

import fitz  # PyMuPDF (for first page text)
import ollama

from utils.text import clean_title, create_bibtex_key

logger = logging.getLogger(__name__)


class MetadataExtractor:
    """Extract structured metadata from PDFs."""

    def __init__(
        self,
        use_crossref: bool = True,
        crossref_email: Optional[str] = None,
    ):
        """
        Initialize metadata extractor.

        Args:
            use_crossref: Enable Crossref enrichment
            crossref_email: Email for Crossref polite pool
        """
        self.use_crossref = use_crossref
        self.crossref_email = crossref_email

    def extract(
        self,
        pdf_path: Path,
        topic: Optional[str] = None,
        available_categories: Optional[list[str]] = None,
    ) -> Dict[str, Any]:
        """
        Extract metadata from PDF using LLM only, then categorize and score relevance to topic using LLM.
        """
        # logger.info(f"[META][LLM] {pdf_path.name}: extracting metadata using LLM...")
        metadata = self._extract_with_llm(pdf_path)
        # logger.info(f"[META][LLM][RESULT] {pdf_path.name}: {metadata}")

        # LLM-based categorization and relevancy scoring
        if topic and metadata.get("title"):
            cat_score = self._llm_categorize_and_score(
                title=metadata.get("title") or "",
                abstract=metadata.get("abstract") or "",
                topic=topic,
                available_categories=available_categories or [],
            )
            metadata.update(cat_score)
            # logger.info(f"[META][LLM][CAT_SCORE] {pdf_path.name}: {cat_score}")

        return metadata

    def _llm_categorize_and_score(
        self,
        title: str,
        abstract: str,
        topic: str,
        available_categories: Optional[list[str]] = None,
    ) -> Dict[str, Any]:
        """
        Use LLM to categorize the paper, score its relevance to the topic, and decide inclusion, with reasoning.
        """
        # (imports removed, use top-level imports)

        categories_str = ", ".join(available_categories) if available_categories else ""
        prompt = (
            f"You are an expert research assistant. Given the research topic and a paper, do the following:\n"
            "1. Carefully read the topic, title, and abstract.\n"
            "2. Compare and contrast ALL available categories, explaining for each why it may or may not fit.\n"
            "3. Justify in detail why the chosen category is the best fit for this paper.\n"
            "4. Explicitly state why other categories are less suitable.\n"
            "5. You are allowed to create a new category if none of the provided categories are a good fit, and explain why.\n"
            "6. Be as accurate and specific as possible.\n"
            f"\nResearch Topic:\n{topic}\n\n"
            f"Paper Title: {title}\n\n"
            f"Paper Abstract: {abstract or 'Not available'}\n\n"
            + (f"Available Categories: {categories_str}\n\n" if categories_str else "")
            + "After your reasoning, respond in JSON format ONLY:\n"
            "{\n"
            '  "category": "category_name",\n'
            '  "relevance_score": 0-10,\n'
            '  "include": true/false,\n'
            '  "reason": "brief explanation"\n'
            "}"
        )
        # logger.info(f"[LLM][CAT_SCORE][PROMPT] Input prompt for categorization/scoring:\n{prompt}")
        response = ollama.generate(
            model="deepseek-r1:8b",
            prompt=prompt,
            options={"temperature": 0.1},
        )
        text = response["response"].strip()
        # logger.info(f"[LLM][CAT_SCORE][RAW_RESPONSE] LLM response for categorization/scoring:\n{text}")
        json_match = re.search(r"{[\s\S]*}", text)
        if json_match:
            json_str = json_match.group(0)
            try:
                result: Dict[str, Any] = _json.loads(json_str)
                # Defensive: ensure correct types
                if not isinstance(result.get("relevance_score"), (int, float)):
                    logger.warning("LLM result: relevance_score had wrong type, coercing to None.")
                    result["relevance_score"] = None
                if not isinstance(result.get("category"), str):
                    logger.warning("LLM result: category had wrong type, coercing to None.")
                    result["category"] = None
                if not isinstance(result.get("reason"), str):
                    logger.warning("LLM result: reason had wrong type, coercing to empty string.")
                    result["reason"] = ""
                if not isinstance(result.get("include"), bool):
                    # Accept "true"/"false" strings as well
                    if isinstance(result.get("include"), str):
                        logger.warning("LLM result: include was string, coercing to bool.")
                        result["include"] = result["include"].strip().lower() == "true"
                    else:
                        logger.warning("LLM result: include had wrong type, coercing to None.")
                        result["include"] = None
                return result
            except Exception as e:
                logger.warning(
                    f"LLM JSON parse failed for categorization/scoring: {e}\nResponse: {text}"
                )
        else:
            logger.warning(
                f"No JSON object found in LLM response for categorization/scoring: {text}"
            )
        return {
            "category": None,
            "relevance_score": None,
            "include": None,
            "reason": "LLM failed to respond correctly.",
        }

    def _extract_with_llm(self, pdf_path: Path) -> Dict[str, Any]:
        """Extract metadata using Ollama LLM (same model as summaries)."""
        try:
            # (imports removed, use top-level imports)

            # Extract first page text
            doc = fitz.open(pdf_path)
            page = doc.load_page(0)
            first_page_text = page.get_text("text")
            doc.close()
            # Schema-based prompt
            prompt = (
                "You are an expert at extracting structured metadata from research papers. "
                "Given the first page of a paper, extract the following fields and output ONLY a valid JSON object matching this schema. "
                "\n\nSCHEMA (output must be valid JSON, no extra text):\n"
                "{\n"
                '  "title": string or null,\n'
                '  "authors": list of strings or null,\n'
                '  "year": string or null,\n'
                '  "venue": string or null,\n'
                '  "abstract": string or null\n'
                "}\n\n"
                "If a field is missing, use null.\n"
                "First page text:\n"
                + (
                    first_page_text[:2000]
                    + ("... [truncated]" if len(first_page_text) > 2000 else "")
                )
            )
            # logger.info(f"[LLM][EXTRACT][PROMPT] Input prompt for metadata extraction from {pdf_path.name}:\n{prompt}")
            # Use the same model as summaries (deepseek-r1:8b or from config)
            model = None
            try:
                from config import Config

                model = Config().ollama.summarize_model
            except Exception:
                model = "deepseek-r1:8b"
            response = ollama.generate(
                model=model,
                prompt=prompt,
                options={"temperature": 0.1},
            )
            text = response["response"].strip()
            # logger.info(f"[LLM][EXTRACT][RAW_RESPONSE] LLM response for metadata extraction from {pdf_path.name}:\n{text}")
            # Robustly extract the first valid JSON object from the response
            json_match = re.search(r"{[\s\S]*}", text)
            if json_match:
                json_str = json_match.group(0)
                try:
                    meta = _json.loads(json_str)
                    # Ensure authors is a list
                    if meta.get("authors") is None:
                        meta["authors"] = []
                    elif not isinstance(meta["authors"], list):
                        meta["authors"] = [str(meta["authors"])]
                except Exception as e:
                    logger.warning(
                        f"LLM JSON parse failed for {pdf_path.name}: {e}\nResponse: {text}"
                    )
                    return {}
            else:
                logger.warning(f"No JSON object found in LLM response for {pdf_path.name}: {text}")
                return {}

            metadata: Dict[str, Any] = meta

            # Enrich with Crossref if DOI available, but only update non-None fields
            if self.use_crossref and metadata.get("doi"):
                crossref_data = self._enrich_with_crossref(metadata["doi"])
                if crossref_data:
                    logger.info(f"[META][CROSSREF] {pdf_path.name}: {crossref_data}")
                    for k, v in crossref_data.items():
                        if v is not None:
                            metadata[k] = v

            # Generate BibTeX
            bibtex = self._generate_bibtex(metadata, pdf_path)
            metadata["bibtex"] = bibtex

            # logger.info(f"[META][FINAL] {pdf_path.name}: {metadata}")
            return metadata
        except Exception as e:
            logger.warning(f"LLM extraction failed for {pdf_path.name}: {e}")
            return {}

    def _extract_from_pdf_metadata(self, pdf_path: Path) -> Dict[str, Any]:
        """Extract basic metadata from PDF properties."""
        try:
            if fitz is None:
                raise ImportError("PyMuPDF (fitz) not available in environment")
            doc = fitz.open(pdf_path)
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
            return None
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
                return {}
            doc = fitz.open(pdf_path)
            if doc.page_count == 0:
                doc.close()
                return {}
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
                return {}

            lines = [ln.strip() for ln in top_text.splitlines() if ln.strip()]
            if not lines:
                return {}

            # Title heuristic: first non-all-caps/short line with enough length
            title_candidate = None
            for ln in lines[:8]:  # inspect first few lines
                if len(ln) >= 8 and len(ln) <= 180:
                    # avoid lines that are mostly uppercase (section headers)
                    letters = [c for c in ln if c.isalpha()]
                    if (
                        not letters
                        or (sum(1 for c in letters if c.isupper()) / max(1, len(letters))) > 0.9
                    ):
                        continue
                    title_candidate = ln
                    break

            authors = []
            # Authors heuristic: next line(s) containing comma/and-separated capitalized tokens
            for ln in lines[1:6]:
                if any(
                    tok in ln.lower()
                    for tok in [
                        "abstract",
                        "introduction",
                        "keywords",
                        "university",
                        "institute",
                        "department",
                    ]
                ):
                    break
                # Split by common separators
                parts = [
                    p.strip()
                    for p in ln.replace(";", ",").replace(" and ", ",").split(",")
                    if p.strip()
                ]
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
            return {}
