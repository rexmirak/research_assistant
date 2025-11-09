"""GROBID client for PDF metadata extraction."""

import logging
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Any, Dict, Optional

import requests

logger = logging.getLogger(__name__)


class GrobidClient:
    """Client for GROBID service."""

    def __init__(self, base_url: str = "http://localhost:8070", timeout: int = 60):
        """
        Initialize GROBID client.

        Args:
            base_url: GROBID service URL
            timeout: Request timeout in seconds
        """
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout

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
            return None

        try:
            # Call GROBID processHeaderDocument endpoint
            url = f"{self.base_url}/api/processHeaderDocument"
            with open(pdf_path, "rb") as f:
                files = {"input": f}
                response = requests.post(url, files=files, timeout=self.timeout)

            if response.status_code != 200:
                logger.warning(f"GROBID failed for {pdf_path.name}: {response.status_code}")
                return None

            # Parse TEI XML response
            return self._parse_tei(response.text)

        except Exception as e:
            logger.error(f"GROBID processing error for {pdf_path.name}: {e}")
            return None

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

        try:
            url = f"{self.base_url}/api/processHeaderDocument"
            with open(pdf_path, "rb") as f:
                files = {"input": f}
                data = {"format": "bibtex"}
                response = requests.post(url, files=files, data=data, timeout=self.timeout)

            if response.status_code == 200:
                return response.text
            return None

        except Exception as e:
            logger.error(f"BibTeX extraction error for {pdf_path.name}: {e}")
            return None

    def _parse_tei(self, tei_xml: str) -> Dict[str, Any]:
        """
        Parse GROBID TEI XML output.

        Args:
            tei_xml: TEI XML string

        Returns:
            Dictionary with extracted fields
        """
        result = {
            "title": None,
            "authors": [],
            "year": None,
            "venue": None,
            "doi": None,
            "abstract": None,
        }

        try:
            # Register namespace
            ns = {"tei": "http://www.tei-c.org/ns/1.0"}
            root = ET.fromstring(tei_xml)

            # Extract title
            title_elem = root.find('.//tei:titleStmt/tei:title[@type="main"]', ns)
            if title_elem is not None and title_elem.text:
                result["title"] = title_elem.text.strip()

            # Extract authors
            for author in root.findall(".//tei:sourceDesc//tei:author", ns):
                persname = author.find(".//tei:persName", ns)
                if persname is not None:
                    forename = persname.find("tei:forename", ns)
                    surname = persname.find("tei:surname", ns)
                    if forename is not None and surname is not None:
                        name = f"{forename.text} {surname.text}".strip()
                        result["authors"].append(name)

            # Extract year
            date_elem = root.find('.//tei:publicationStmt//tei:date[@type="published"]', ns)
            if date_elem is not None and date_elem.get("when"):
                result["year"] = date_elem.get("when")[:4]

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

        except Exception as e:
            logger.error(f"TEI parsing error: {e}")

        return result
