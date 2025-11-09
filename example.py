#!/usr/bin/env python3
"""
Example: Running the research assistant pipeline

This script demonstrates how to use the research assistant
on a sample dataset.
"""

from pathlib import Path
from config import Config
from cli import process
import click


def main():
    """Run example pipeline."""

    # Example configuration
    root_dir = Path("./sample_papers")  # Replace with your directory
    topic = """
    I am researching the application of machine learning techniques in medical diagnosis,
    particularly focusing on deep learning models for image analysis, clinical decision support,
    and patient outcome prediction. I'm interested in both methodological advances and
    practical deployment considerations.
    """

    # Check if sample directory exists
    if not root_dir.exists():
        print(f"Error: Sample directory not found: {root_dir}")
        print("\nPlease either:")
        print(f"  1. Create {root_dir} and add PDF subdirectories")
        print("  2. Update this script with your actual PDF directory")
        return

    print("=" * 80)
    print("Research Assistant - Example Run")
    print("=" * 80)
    print(f"\nRoot Directory: {root_dir}")
    print(f"Topic: {topic.strip()[:100]}...")
    print(f"\nOutput: ./outputs")
    print(f"Cache: ./cache")
    print("\nThis will:")
    print("  1. Scan PDFs in category subdirectories")
    print("  2. Extract text and metadata")
    print("  3. Detect and move duplicates")
    print("  4. Score relevance to your topic (0-10)")
    print("  5. Validate and recategorize papers")
    print("  6. Quarantine low-relevance papers")
    print("  7. Generate summaries")
    print("  8. Output CSV, JSONL, and Markdown")
    print("\n" + "=" * 80)

    proceed = input("\nProceed? (yes/no): ")
    if proceed.lower() not in ["yes", "y"]:
        print("Cancelled.")
        return

    # Run pipeline using Click context
    ctx = click.Context(process)
    ctx.invoke(
        process,
        root_dir=root_dir,
        topic=topic,
        output_dir=Path("./outputs"),
        cache_dir=Path("./cache"),
        config_file=None,
        dry_run=False,
        resume=False,
        relevance_threshold=6.5,
        workers=4,
    )


if __name__ == "__main__":
    main()
