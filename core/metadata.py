from typing import Dict
from pydantic import BaseModel


class MetadataSchema(BaseModel):
    title: str | None
    authors: list[str] | None
    year: str | None
    venue: str | None
    abstract: str | None


class CategorizationSchema(BaseModel):
    category: str
    relevance_score: float
    include: bool
    reason: str


class MultiCategoryScoreSchema(BaseModel):
    """Schema for multi-category scoring with topic relevance."""

    topic_relevance: int  # 1-10 relevance to overall research topic
    category_scores: Dict[str, int]  # category_name -> score (1-10)
    reasoning: str  # Explanation of scoring and categorization


"""Metadata extraction using LLM and PDF heuristics."""

import json as _json
import logging
import re
from pathlib import Path

from core.category_validator import CategoryValidator
from typing import Any, Dict, Optional

import fitz  # PyMuPDF (for first page text)

from utils.llm_provider import llm_generate
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
        self.validator = CategoryValidator()

    def extract(
        self,
        pdf_path: Path,
        topic: Optional[str] = None,
        available_categories: Optional[list[str]] = None,
    ) -> Dict[str, Any]:
        """
        Extract metadata from PDF using LLM only, then categorize and score relevance to topic using LLM.
        """
        metadata = self._extract_with_llm(pdf_path)

        # LLM-based categorization and relevancy scoring
        if topic and metadata.get("title"):
            cat_score = self._llm_categorize_and_score(
                title=metadata.get("title") or "",
                abstract=metadata.get("abstract") or "",
                topic=topic,
                available_categories=available_categories or [],
            )
            metadata.update(cat_score)

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
            f"You are an expert research assistant specializing in paper categorization. "
            f"Use step-by-step reasoning to categorize this paper accurately.\n\n"
            f"RESEARCH TOPIC:\n{topic}\n\n"
            f"PAPER TITLE:\n{title}\n\n"
            f"PAPER CONTENT (Abstract + Introduction):\n{abstract or 'Not available'}\n\n"
            + (f"AVAILABLE CATEGORIES:\n{categories_str}\n\n" if categories_str else "")
            + "STEP-BY-STEP REASONING PROCESS:\n\n"  # nosec B608 - This is an LLM prompt, not SQL
            "Step 1 - UNDERSTAND THE PAPER:\n"
            "- What is the main focus/contribution of this paper?\n"
            "- What specific problem does it address?\n"
            "- What methods or techniques does it discuss?\n"
            "- What are the key concepts, keywords, and technical terms?\n\n"
            "Step 2 - ANALYZE ALIGNMENT WITH RESEARCH TOPIC:\n"
            "- How does this paper relate to the research topic?\n"
            "- What specific aspects of the topic does it address?\n"
            "- Does it provide theoretical foundations, empirical results, practical techniques, or survey/overview?\n"
            "- What is the relevance score (0-10) and why?\n\n"
            "Step 3 - EVALUATE EACH CATEGORY:\n"
            "For EACH available category, explicitly consider:\n"
            "- Does the paper's focus align with this category?\n"
            "- What keywords or concepts match or don't match?\n"
            "- Rate the fit: strong/moderate/weak/none and explain why\n\n"
            "Step 4 - SELECT BEST CATEGORY:\n"
            "- Which category has the STRONGEST alignment?\n"
            "- If no category fits well, what new category name would be more appropriate?\n"
            "- Why is this category better than the others?\n\n"
            "Step 5 - MAKE INCLUSION DECISION:\n"
            "- Should this paper be included (true) or excluded (false) from the research collection?\n"
            "- Consider: relevance score, topic alignment, quality of content\n"
            "- Explain your decision\n\n"
            "After completing your reasoning, output ONLY valid JSON:\n"
            "{\n"
            '  "category": "exact_category_name_or_new_category",\n'
            '  "relevance_score": 0-10,\n'
            '  "include": true or false,\n'
            '  "reason": "concise explanation of categorization and inclusion decision"\n'
            "}\n\n"
            "IMPORTANT:\n"
            "- Be precise and analytical in your reasoning\n"
            "- Consider ALL available categories before deciding\n"
            "- Choose the MOST SPECIFIC category that fits\n"
            "- Only create a new category if truly necessary\n"
            "- Base your decision on the paper's actual content and contributions"
        )

        from config import Config

        cfg = Config()
        provider = getattr(cfg, "llm_provider", "ollama")
        options: dict = {"temperature": 0.1}
        if provider == "gemini":
            options["schema"] = CategorizationSchema
            model = None  # Use default Gemini model from config
        else:
            model = "deepseek-r1:8b"  # Ollama model
        response = llm_generate(
            prompt=prompt,
            model=model,
            options=options,
        )
        if provider == "gemini":
            # Gemini returns parsed dict
            return dict(response["response"])
        else:
            text = response["response"].strip()
            json_match = re.search(r"{[\s\S]*}", text)
            if json_match:
                json_str = json_match.group(0)
                try:
                    result: Dict[str, Any] = _json.loads(json_str)
                    # Defensive: ensure correct types
                    if not isinstance(result.get("relevance_score"), (int, float)):
                        logger.warning(
                            "LLM result: relevance_score had wrong type, coercing to None."
                        )
                        result["relevance_score"] = None
                    if not isinstance(result.get("category"), str):
                        logger.warning("LLM result: category had wrong type, coercing to None.")
                        result["category"] = None
                    if not isinstance(result.get("reason"), str):
                        logger.warning(
                            "LLM result: reason had wrong type, coercing to empty string."
                        )
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

    def classify_paper_with_scores(
        self,
        title: str,
        abstract: str,
        topic: str,
        categories: Dict[str, str],
    ) -> Dict[str, Any]:
        """
        Classify paper with relevance scores for all categories.

        Args:
            title: Paper title
            abstract: Paper abstract + introduction
            topic: Research topic
            categories: Dictionary of category_name -> definition

        Returns:
            Dictionary with:
                - topic_relevance: int (1-10)
                - category_scores: dict {category_name: score (1-10)}
                - best_category: str (category with highest score)
                - reasoning: str (explanation)
        """
        # Build categories description
        categories_desc = "\n".join(
            [f"- {name}: {definition}" for name, definition in categories.items()]
        )

        prompt = f"""You are an expert research assistant evaluating paper relevance and categorization.

RESEARCH TOPIC: {topic}

PAPER TITLE: {title}

PAPER CONTENT (Abstract + Introduction):
{abstract or 'Not available'}

AVAILABLE CATEGORIES:
{categories_desc}

YOUR TASK:
1. Evaluate how relevant this paper is to the overall research topic "{topic}" (1-10 scale)
   - 1-3: Not relevant or only tangentially related
   - 4-6: Somewhat relevant, addresses related concepts
   - 7-8: Highly relevant, directly addresses the topic
   - 9-10: Extremely relevant, core contribution to the topic

2. For EACH category above, rate how well this paper fits (1-10 scale):
   - Consider the paper's focus, methodology, and contributions
   - Compare against the category definition
   - Be precise - different categories will have different scores

3. Identify the BEST FITTING category (highest score)

4. Provide clear reasoning for your scores

Return ONLY valid JSON matching this structure:
{{
  "topic_relevance": <integer 1-10>,
  "category_scores": {{
    "category_name_1": <integer 1-10>,
    "category_name_2": <integer 1-10>,
    ...
  }},
  "reasoning": "Brief explanation of relevance and categorization"
}}

IMPORTANT:
- All scores must be integers from 1 to 10
- Score ALL categories (include all category names in category_scores)
- Be discriminating - not everything scores 8-10
- Base scores on actual paper content and category definitions
"""

        from config import Config

        cfg = Config()
        provider = getattr(cfg, "llm_provider", "ollama")
        options: dict = {"temperature": 0.1}

        # Don't use schema for this call - dynamic category names don't work with Gemini's schema validation
        model = None if provider == "gemini" else "deepseek-r1:8b"

        try:
            response = llm_generate(
                prompt=prompt,
                model=model,
                options=options,
            )

            # Parse JSON from response (both providers)
            text = response["response"].strip()
            json_match = re.search(r"{[\s\S]*}", text)
            if json_match:
                json_str = json_match.group(0)
                parsed_result = _json.loads(json_str)
                if not isinstance(parsed_result, dict):
                    logger.warning("Parsed JSON is not a dictionary")
                    return self._fallback_classification(categories)
                result: Dict[str, Any] = parsed_result
            else:
                logger.warning(f"No JSON in classification response: {text[:200]}")
                return self._fallback_classification(categories)

            # Validate and add best_category
            if not isinstance(result.get("category_scores"), dict):
                logger.warning("Invalid category_scores format")
                return self._fallback_classification(categories)

            # Validate and match LLM-returned category names to existing categories
            category_scores = result["category_scores"]
            validated_scores = {}
            existing_category_names = list(categories.keys())

            for llm_category, score in category_scores.items():
                # Try to match to existing categories
                matched = self.validator.match_to_existing(
                    llm_category, existing_category_names
                )
                if matched:
                    validated_scores[matched] = score
                    if matched != llm_category:
                        logger.info(
                            f"Matched LLM category '{llm_category}' to '{matched}'"
                        )
                else:
                    # LLM returned unknown category - sanitize and log warning
                    sanitized = self.validator.sanitize_name(llm_category)
                    logger.warning(
                        f"LLM returned unknown category '{llm_category}' (sanitized: '{sanitized}') - not in taxonomy"
                    )
                    # Still include it with low score so we don't lose data
                    validated_scores[sanitized] = score

            # Find best category (from validated scores)
            if validated_scores:
                best_category = max(validated_scores.items(), key=lambda x: x[1])[0]
                result["best_category"] = best_category
                result["category_scores"] = validated_scores
            else:
                result["best_category"] = None
                result["category_scores"] = {}

            return result

        except Exception as e:
            logger.error(f"Classification failed: {e}")
            return self._fallback_classification(categories)

    def _fallback_classification(self, categories: Dict[str, str]) -> Dict[str, Any]:
        """Fallback classification when LLM fails."""
        # Default to first category with low scores
        first_category = list(categories.keys())[0] if categories else "uncategorized"
        return {
            "topic_relevance": 1,
            "category_scores": {name: 1 for name in categories.keys()},
            "best_category": first_category,
            "reasoning": "Classification failed - using fallback values",
        }

    def _extract_with_llm(self, pdf_path: Path) -> Dict[str, Any]:
        """Extract metadata using LLM (Ollama or Gemini)."""
        try:
            from config import Config

            cfg = Config()
            provider = getattr(cfg, "llm_provider", "ollama")

            # Extract first 5 pages for better abstract + intro extraction
            doc = fitz.open(pdf_path)
            pages_text = []
            max_pages = min(5, doc.page_count)
            for page_num in range(max_pages):
                page = doc.load_page(page_num)
                pages_text.append(page.get_text("text"))
            doc.close()

            # Combine first pages (up to 5000 chars for better abstract+intro coverage)
            combined_text = "\n".join(pages_text)[:5000]

            # Schema-based prompt - ask for abstract+intro combined
            prompt = (
                "You are an expert at extracting structured metadata from research papers. "
                "Given the first pages of a paper, extract the following fields and output ONLY a valid JSON object matching this schema. "
                "\n\nSCHEMA (output must be valid JSON, no extra text):\n"
                "{\n"
                '  "title": string or null,\n'
                '  "authors": list of strings or null,\n'
                '  "year": string or null,\n'
                '  "venue": string or null,\n'
                '  "abstract": string or null (combine abstract and introduction sections for comprehensive content)\n'
                "}\n\n"
                "IMPORTANT: For the 'abstract' field, extract both the abstract AND introduction sections together. "
                "This provides richer context for categorization. Combine them into a single coherent text.\n\n"
                "If a field is missing, use null.\n"
                "First pages text:\n"
                + combined_text
                + ("... [truncated]" if len("\n".join(pages_text)) > 5000 else "")
            )
            options: dict = {"temperature": 0.1}
            if provider == "gemini":
                options["schema"] = MetadataSchema
                # For Gemini, don't specify model (use default from config)
                model = None
            else:
                # Use the same model as summaries for Ollama
                try:
                    model = Config().ollama.summarize_model
                except Exception:
                    model = "deepseek-r1:8b"
            response = llm_generate(
                prompt=prompt,
                model=model,
                options=options,
            )
            if provider == "gemini":
                meta = dict(response["response"])
                # Ensure authors is a list
                if meta.get("authors") is None:
                    meta["authors"] = []
                elif not isinstance(meta["authors"], list):
                    meta["authors"] = [str(meta["authors"])]
            else:
                text = response["response"].strip()
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
                    logger.warning(
                        f"No JSON object found in LLM response for {pdf_path.name}: {text}"
                    )
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
