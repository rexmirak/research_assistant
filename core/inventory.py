"""Directory traversal and PDF inventory."""

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional

from utils.hash import file_hash

logger = logging.getLogger(__name__)


@dataclass
class PDFDocument:
    """Represents a PDF document in the inventory."""

    file_path: Path
    file_name: str
    category: str
    file_size: int
    file_hash: str
    modified_time: float

    def to_dict(self) -> Dict:
        """Convert to dictionary."""
        return {
            "file_path": str(self.file_path),
            "file_name": self.file_name,
            "category": self.category,
            "file_size": self.file_size,
            "file_hash": self.file_hash,
            "modified_time": self.modified_time,
        }


class InventoryManager:
    """Manages PDF inventory and directory traversal."""

    def __init__(self, root_dir: Path, ignore_patterns: Optional[List[str]] = None):
        """
        Initialize inventory manager.

        Args:
            root_dir: Root directory containing category folders
            ignore_patterns: Patterns to ignore (e.g., ['.*', '_*'])
        """
        self.root_dir = root_dir
        self.ignore_patterns = ignore_patterns or [".*", "__*", "repeated", "quarantined"]
        self.documents: List[PDFDocument] = []

    def scan(self) -> List[PDFDocument]:
        """
        Scan root directory for PDFs.

        Returns:
            List of PDF documents
        """
        logger.info(f"Scanning directory: {self.root_dir}")
        self.documents = []

        # Traverse directory structure
        for pdf_path in self.root_dir.rglob("*.pdf"):
            # Skip ignored patterns
            if self._should_ignore(pdf_path):
                continue

            # Determine category from directory structure
            category = self._get_category(pdf_path)

            try:
                # Get file metadata
                stat = pdf_path.stat()
                doc_hash = file_hash(pdf_path)

                doc = PDFDocument(
                    file_path=pdf_path,
                    file_name=pdf_path.name,
                    category=category,
                    file_size=stat.st_size,
                    file_hash=doc_hash,
                    modified_time=stat.st_mtime,
                )

                self.documents.append(doc)
                logger.debug(f"Found: {doc.file_name} in {category}")

            except Exception as e:
                logger.error(f"Error processing {pdf_path}: {e}")

        logger.info(
            f"Found {len(self.documents)} PDFs across {len(self.get_categories())} categories"
        )
        return self.documents

    def _should_ignore(self, path: Path) -> bool:
        """Check if path should be ignored."""
        for pattern in self.ignore_patterns:
            # Check if any parent directory matches pattern
            for part in path.parts:
                if pattern.startswith("*") and part.endswith(pattern[1:]):
                    return True
                elif pattern.endswith("*") and part.startswith(pattern[:-1]):
                    return True
                elif pattern.strip("*") == part:
                    return True
        return False

    def _get_category(self, pdf_path: Path) -> str:
        """
        Extract category from file path.

        Args:
            pdf_path: Path to PDF file

        Returns:
            Category name
        """
        # Get relative path from root
        try:
            rel_path = pdf_path.relative_to(self.root_dir)
            # First directory is the category
            if len(rel_path.parts) > 1:
                return rel_path.parts[0]
            return "uncategorized"
        except ValueError:
            return "unknown"

    def get_categories(self) -> List[str]:
        """Get list of unique categories."""
        return sorted(set(doc.category for doc in self.documents))

    def get_documents_by_category(self, category: str) -> List[PDFDocument]:
        """Get all documents in a category."""
        return [doc for doc in self.documents if doc.category == category]

    def get_document_by_path(self, path: Path) -> Optional[PDFDocument]:
        """Get document by file path."""
        for doc in self.documents:
            if doc.file_path == path:
                return doc
        return None

    def summary(self) -> Dict:
        """Get inventory summary statistics."""
        categories = {}
        for doc in self.documents:
            if doc.category not in categories:
                categories[doc.category] = {"count": 0, "total_size": 0}
            categories[doc.category]["count"] += 1
            categories[doc.category]["total_size"] += doc.file_size

        return {
            "total_documents": len(self.documents),
            "total_categories": len(categories),
            "categories": categories,
        }
