"""Deduplication using exact and near-duplicate detection."""

import logging
from typing import Dict, List, Optional, Set, Tuple

from datasketch import MinHash, MinHashLSH

from core.inventory import PDFDocument

logger = logging.getLogger(__name__)


class DedupManager:
    """Manage exact and near-duplicate detection."""

    def __init__(self, similarity_threshold: float = 0.95, num_perm: int = 128):
        """
        Initialize dedup manager.

        Args:
            similarity_threshold: Jaccard similarity threshold for near-duplicates
            num_perm: Number of permutations for MinHash
        """
        self.similarity_threshold = similarity_threshold
        self.num_perm = num_perm
        self.lsh = MinHashLSH(threshold=similarity_threshold, num_perm=num_perm)

        # Track canonical papers
        self.exact_duplicates: Dict[str, str] = {}  # hash -> canonical_paper_id
        self.near_duplicates: Dict[str, str] = {}  # paper_id -> canonical_paper_id

    def find_exact_duplicates(self, documents: List[PDFDocument]) -> Dict[str, List[PDFDocument]]:
        """
        Find exact duplicates based on file hash.

        Args:
            documents: List of PDF documents

        Returns:
            Dictionary mapping canonical doc to its duplicates
        """
        hash_groups: Dict[str, List[PDFDocument]] = {}

        for doc in documents:
            if doc.file_hash not in hash_groups:
                hash_groups[doc.file_hash] = []
            hash_groups[doc.file_hash].append(doc)

        # Filter to groups with duplicates
        duplicates = {docs[0].file_name: docs[1:] for docs in hash_groups.values() if len(docs) > 1}

        logger.info(f"Found {len(duplicates)} exact duplicate groups")
        return duplicates

    def find_near_duplicates(
        self, paper_texts: Dict[str, str], paper_ids: Dict[str, str]
    ) -> Dict[str, List[str]]:
        """
        Find near-duplicates using MinHash LSH.

        Args:
            paper_texts: Dictionary mapping paper_id to text
            paper_ids: Dictionary mapping paper_id to display name

        Returns:
            Dictionary mapping canonical paper_id to near-duplicate paper_ids
        """
        logger.info(f"Computing MinHash for {len(paper_texts)} papers")

        # Compute MinHash for each paper
        minhashes: Dict[str, MinHash] = {}
        for paper_id, text in paper_texts.items():
            mh = self._compute_minhash(text)
            minhashes[paper_id] = mh
            self.lsh.insert(paper_id, mh)

        # Find near-duplicates
        near_dup_groups: Dict[str, Set[str]] = {}
        processed: Set[str] = set()

        for paper_id, mh in minhashes.items():
            if paper_id in processed:
                continue

            # Query LSH for similar papers
            similar = self.lsh.query(mh)

            if len(similar) > 1:
                # This paper and its duplicates
                canonical = paper_id
                duplicates = [p for p in similar if p != canonical]

                if duplicates:
                    near_dup_groups[canonical] = set(duplicates)
                    processed.update(duplicates)
                    processed.add(canonical)

        logger.info(f"Found {len(near_dup_groups)} near-duplicate groups")
        return {k: list(v) for k, v in near_dup_groups.items()}

    def _compute_minhash(self, text: str) -> MinHash:
        """Compute MinHash for text."""
        mh = MinHash(num_perm=self.num_perm)

        # Tokenize text into words
        words = text.lower().split()

        # Create shingles (n-grams)
        shingle_size = 3
        for i in range(len(words) - shingle_size + 1):
            shingle = " ".join(words[i : i + shingle_size])
            mh.update(shingle.encode("utf-8"))

        return mh

    def mark_duplicate(self, paper_id: str, canonical_id: str):
        """Mark paper as duplicate of canonical paper."""
        self.near_duplicates[paper_id] = canonical_id

    def is_duplicate(self, paper_id: str) -> Tuple[bool, Optional[str]]:
        """
        Check if paper is a duplicate.

        Returns:
            Tuple of (is_duplicate, canonical_paper_id)
        """
        if paper_id in self.near_duplicates:
            return True, self.near_duplicates[paper_id]
        return False, None
