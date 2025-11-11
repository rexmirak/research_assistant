"""GROBID client for PDF metadata extraction.

Enhancements:
 - Robust retry logic for transient failures (connection, 5xx, malformed output)
 - Graceful handling of invalid/non-XML responses (HTML error page, empty string)
 - Reduced log noise: single warning per failure instead of noisy stack traces
 - Returns None on hard failures so upstream fallback (PDF metadata) can engage
"""

import logging
import time
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

import requests

logger = logging.getLogger(__name__)


@dataclass
class _RetryConfig:
    attempts: int = 3
    base_delay: float = 1.0  # seconds
    max_delay: float = 5.0
    backoff: float = 2.0


class GrobidClient:
    """Client for GROBID service."""

    def __init__(
        self,
        base_url: str = "http://localhost:8070",
        timeout: int = 60,
        retry: _RetryConfig | None = None,
    ):
        """
        Initialize GROBID client.

        Args:
            base_url: GROBID service URL
            timeout: Request timeout in seconds
        """
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout
        self.retry = retry or _RetryConfig()

    def is_alive(self) -> bool:
        """Check if GROBID service is running."""
        try:
            response = requests.get(f"{self.base_url}/api/isalive", timeout=5)
            return response.status_code == 200
        except Exception as e:
            logger.error(f"GROBID health check failed: {e}")
            return False

    def process_pdf(self, pdf_path: Path) -> Optional[Dict[str, Any]]:
        """
        Process PDF and extract metadata.

        Args:
            pdf_path: Path to PDF file

        Returns:
            Dictionary with extracted metadata or None
        """
        if not pdf_path.exists():
            logger.error(f"PDF not found: {pdf_path}")
            return {"grobid_status": "file_missing"}

        url = f"{self.base_url}/api/processHeaderDocument"

        def _do_request() -> Optional[str]:
            try:
                with open(pdf_path, "rb") as f:
                    files = {"input": f}
                    response = requests.post(url, files=files, timeout=self.timeout)
                if response.status_code != 200:
                    logger.warning(
                        f"GROBID response status {response.status_code} for {pdf_path.name}"
                    )
                    return None
                text = response.text.strip()
                if not text:
                    logger.warning(f"Empty GROBID response for {pdf_path.name}")
                    return None
                return text
            except Exception as e:  # network / connection errors
                logger.warning(f"GROBID request error for {pdf_path.name}: {e}")
                return None

        tei_xml = self._with_retries(_do_request)
        if not tei_xml:
            # Give up – upstream will fallback; return status object
            return {"grobid_status": "request_failed"}

        parsed = self._parse_tei(tei_xml)
        if not parsed.get("title") and not parsed.get("authors"):
            # Bad parse – mark status for fallback; keep partial fields
            if "grobid_status" not in parsed:
                parsed["grobid_status"] = "parse_error"
            return parsed
        # Successful parse
        parsed["grobid_status"] = parsed.get("grobid_status", "ok")
        return parsed

    def get_bibtex(self, pdf_path: Path) -> Optional[str]:
        """
        Get BibTeX citation for PDF.

        Args:
            pdf_path: Path to PDF file

        Returns:
            BibTeX string or None
        """
        if not pdf_path.exists():
            return None

        url = f"{self.base_url}/api/processHeaderDocument"

        def _do_bibtex() -> Optional[str]:
            try:
                with open(pdf_path, "rb") as f:
                    files = {"input": f}
                    data = {"format": "bibtex"}
                    response = requests.post(url, files=files, data=data, timeout=self.timeout)
                if response.status_code != 200:
                    return None
                text = response.text.strip()
                return text or None
            except Exception as e:
                logger.debug(f"BibTeX request error for {pdf_path.name}: {e}")
                return None

        return self._with_retries(_do_bibtex)

    def _parse_tei(self, tei_xml: str) -> Dict[str, Any]:
        """
        Parse GROBID TEI XML output.

        Args:
            tei_xml: TEI XML string

        Returns:
            Dictionary with extracted fields
        """
        result: Dict[str, Any] = {
            "title": None,
            "authors": [],
            "year": None,
            "venue": None,
            "doi": None,
            "abstract": None,
        }

        # Quick sanity check to avoid parsing obvious non-XML (e.g., HTML error page)
        sample = tei_xml[:100].lower()
        if not tei_xml.strip().startswith("<") or "<html" in sample:
            logger.warning("Non-XML or HTML response received instead of TEI; skipping parse")
            result["grobid_status"] = "non_xml"
            return result

        try:
            ns = {"tei": "http://www.tei-c.org/ns/1.0"}
            root = ET.fromstring(tei_xml)

            # Extract title
            title_elem = root.find('.//tei:titleStmt/tei:title[@type="main"]', ns)
            if title_elem is not None and title_elem.text:
                result["title"] = title_elem.text.strip()

            # Extract authors
            authors_list: List[str] = []
            for author in root.findall(".//tei:sourceDesc//tei:author", ns):
                persname = author.find(".//tei:persName", ns)
                if persname is not None:
                    forename = persname.find("tei:forename", ns)
                    surname = persname.find("tei:surname", ns)
                    if forename is not None and surname is not None:
                        forename_text = forename.text or ""
                        surname_text = surname.text or ""
                        name = f"{forename_text} {surname_text}".strip()
                        authors_list.append(name)
            result["authors"] = authors_list

            # Extract year
            date_elem = root.find('.//tei:publicationStmt//tei:date[@type="published"]', ns)
            if date_elem is not None:
                when_attr = date_elem.get("when")
                if when_attr and len(when_attr) >= 4:
                    result["year"] = when_attr[:4]

            # Extract venue
            venue_elem = root.find(".//tei:monogr//tei:title", ns)
            if venue_elem is not None and venue_elem.text:
                result["venue"] = venue_elem.text.strip()

            # Extract DOI
            doi_elem = root.find('.//tei:idno[@type="DOI"]', ns)
            if doi_elem is not None and doi_elem.text:
                result["doi"] = doi_elem.text.strip()

            # Extract abstract
            abstract_elem = root.find(".//tei:abstract/tei:div/tei:p", ns)
            if abstract_elem is not None and abstract_elem.text:
                result["abstract"] = abstract_elem.text.strip()

        except ET.ParseError as e:
            logger.warning(f"TEI parsing error (malformed XML): {e}")
            result["grobid_status"] = "parse_error"
        except Exception as e:
            logger.warning(f"Unexpected TEI parsing error: {e}")
            result["grobid_status"] = "unexpected_error"

        return result

    # ------------------------------------------------------------------
    # Retry helper
    # ------------------------------------------------------------------
    def _with_retries(self, fn: Callable[[], Optional[str]]) -> Optional[str]:
        attempt = 0
        delay = self.retry.base_delay
        while attempt < self.retry.attempts:
            result = fn()
            if result is not None:
                return result
            attempt += 1
            if attempt < self.retry.attempts:
                time.sleep(min(delay, self.retry.max_delay))
                delay *= self.retry.backoff
        return None
