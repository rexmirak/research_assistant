"""Output generation: JSONL, CSV, and Markdown summaries."""

import csv
import json
import logging
from pathlib import Path
from typing import Any, Dict, List

logger = logging.getLogger(__name__)


class OutputGenerator:
    """Generate all output files."""

    def __init__(self, output_dir: Path):
        """
        Initialize output generator.

        Args:
            output_dir: Directory for output files
        """
        self.output_dir = output_dir
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self.summaries_dir = output_dir / "summaries"
        self.summaries_dir.mkdir(exist_ok=True)

    def write_jsonl(self, records: List[Dict[str, Any]], filename: str = "index.jsonl"):
        """
        Write records to JSONL file.

        Args:
            records: List of record dictionaries
            filename: Output filename
        """
        output_path = self.output_dir / filename

        try:
            with open(output_path, "w") as f:
                for record in records:
                    f.write(json.dumps(record) + "\n")

            logger.info(f"Wrote {len(records)} records to {output_path}")

        except Exception as e:
            logger.error(f"Failed to write JSONL: {e}")

    def write_csv(self, records: List[Dict[str, Any]], filename: str = "index.csv"):
        """
        Write records to CSV file.

        Args:
            records: List of record dictionaries
            filename: Output filename
        """
        if not records:
            logger.warning("No records to write to CSV")
            return

        output_path = self.output_dir / filename

        try:
            # Define column order
            columns = [
                "paper_id",
                "title",
                "authors",
                "year",
                "venue",
                "doi",
                "category",
                "topic_relevance",
                "category_scores",
                "reasoning",
                "duplicate_of",
                "is_duplicate",
                "path",
                "bibtex",
                "summary_file",
                "analyzed",
            ]

            with open(output_path, "w", newline="", encoding="utf-8") as f:
                writer = csv.DictWriter(f, fieldnames=columns, extrasaction="ignore")
                writer.writeheader()

                for record in records:
                    # Format authors as string
                    if isinstance(record.get("authors"), list):
                        record["authors"] = "; ".join(record["authors"])
                    
                    # Format category_scores as JSON string
                    if isinstance(record.get("category_scores"), dict):
                        record["category_scores"] = json.dumps(record["category_scores"])

                    writer.writerow(record)

            logger.info(f"Wrote {len(records)} records to {output_path}")

        except Exception as e:
            logger.error(f"Failed to write CSV: {e}")

    def write_category_summary(self, category: str, summaries: List[Dict[str, Any]]):
        """
        Write markdown summary file for category.

        Args:
            category: Category name
            summaries: List of paper summaries with metadata
        """
        output_path = self.summaries_dir / f"{category}.md"

        try:
            with open(output_path, "w", encoding="utf-8") as f:
                # Write header
                f.write(f"# {category}\n\n")
                f.write(f"*{len(summaries)} papers*\n\n")

                # Write table of contents
                f.write("## Table of Contents\n\n")
                for i, summary in enumerate(summaries, 1):
                    title = summary.get("title", "Untitled")
                    anchor = self._create_anchor(title)
                    f.write(f"{i}. [{title}](#{anchor})\n")

                f.write("\n---\n\n")

                # Write summaries
                for i, summary in enumerate(summaries, 1):
                    title = summary.get("title", "Untitled")
                    anchor = self._create_anchor(title)

                    f.write(f'<a name="{anchor}"></a>\n\n')
                    f.write(f"## {i}. {title}\n\n")

                    # Metadata
                    authors = summary.get("authors", [])
                    if authors:
                        f.write(f"**Authors:** {', '.join(authors)}\n\n")

                    year = summary.get("year")
                    if year:
                        f.write(f"**Year:** {year}\n\n")

                    venue = summary.get("venue")
                    if venue:
                        f.write(f"**Venue:** {venue}\n\n")

                    score = summary.get("relevance_score")
                    if score is not None:
                        f.write(f"**Relevance Score:** {score}/10\n\n")

                    # Summary content
                    content = summary.get("summary", "")
                    if content:
                        f.write("### Summary\n\n")
                        f.write(f"{content}\n\n")

                    # BibTeX
                    bibtex = summary.get("bibtex", "")
                    if bibtex:
                        f.write("### Citation\n\n")
                        f.write("```bibtex\n")
                        f.write(f"{bibtex}\n")
                        f.write("```\n\n")

                    f.write("---\n\n")

            logger.info(f"Wrote category summary to {output_path}")

        except Exception as e:
            logger.error(f"Failed to write category summary for {category}: {e}")

    def _create_anchor(self, title: str) -> str:
        """Create markdown anchor from title."""
        # Convert to lowercase and replace spaces with hyphens
        anchor = title.lower().strip()
        anchor = "".join(c if c.isalnum() or c.isspace() else "" for c in anchor)
        anchor = "-".join(anchor.split())
        return anchor

    def write_statistics(self, stats: Dict[str, Any], filename: str = "statistics.json"):
        """
        Write statistics to JSON file.

        Args:
            stats: Statistics dictionary
            filename: Output filename
        """
        output_path = self.output_dir / filename

        try:
            with open(output_path, "w") as f:
                json.dump(stats, f, indent=2)

            logger.info(f"Wrote statistics to {output_path}")

        except Exception as e:
            logger.error(f"Failed to write statistics: {e}")
