"""Main CLI for research assistant pipeline."""

import json
import logging
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
from pathlib import Path

import click
from tqdm import tqdm

from config import Config
from core.dedup import DedupManager
from core.inventory import InventoryManager
from core.manifest import ManifestManager
from core.metadata import MetadataExtractor
from core.mover import FileMover
from core.outputs import OutputGenerator
from core.parser import PDFParser
from core.summarizer import Summarizer
from utils.cache_manager import CacheManager

# Setup logging - initially to stderr only (will add file handler later)
# This prevents logger output from interfering with tqdm progress bar
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[],  # Start with no handlers - we'll add file handler only
)
logger = logging.getLogger(__name__)


@click.group()
def cli():
    """Research Assistant - Intelligent PDF analysis pipeline."""
    pass


@cli.command()
@click.option(
    "--root-dir",
    required=True,
    type=click.Path(exists=True, path_type=Path),
    help="Root directory containing category folders with PDFs",
)
@click.option("--topic", required=True, type=str, help="Research topic description")
@click.option(
    "--output-dir", default="./outputs", type=click.Path(path_type=Path), help="Output directory"
)
@click.option(
    "--cache-dir", default="./cache", type=click.Path(path_type=Path), help="Cache directory"
)
@click.option(
    "--purge-cache",
    is_flag=True,
    help="Purge the cache directory before processing (deletes cache.db and OCR cache)",
)
@click.option(
    "--config-file",
    type=click.Path(exists=True, path_type=Path),
    help="Configuration YAML file (optional)",
)
@click.option(
    "--llm-provider",
    type=click.Choice(["ollama", "gemini"]),
    default=None,
    help="LLM provider to use (ollama or gemini)",
)
@click.option("--dry-run", is_flag=True, help="Run without moving files")
@click.option("--resume", is_flag=True, help="Resume from cache")
@click.option(
    "--relevance-threshold",
    type=float,
    default=7.0,
    help="Minimum relevance score for inclusion (papers below this are quarantined)",
)
@click.option("--workers", type=int, default=4, help="Number of worker processes")
def process(
    root_dir,
    topic,
    output_dir,
    cache_dir,
    purge_cache,
    config_file,
    llm_provider,
    dry_run,
    resume,
    relevance_threshold,
    workers,
):
    """Process PDFs through the full analysis pipeline."""

    # Load configuration
    if config_file:
        config = Config.from_yaml(config_file)
    else:
        config = Config()

    # Override with CLI parameters
    config.root_dir = root_dir
    config.topic = topic
    config.output_dir = Path(output_dir)
    config.cache_dir = Path(cache_dir)
    config.dry_run = dry_run
    config.resume = resume
    config.scoring.relevance_threshold = relevance_threshold
    config.processing.workers = workers
    if llm_provider:
        config.llm_provider = llm_provider
        # Also set as environment variable so it's picked up by Config() instances
        import os

        os.environ["LLM_PROVIDER"] = llm_provider

    # Setup directories
    config.setup_directories()

    # Optionally purge cache
    if purge_cache:
        import shutil

        try:
            logger.info(f"Purging cache at {config.cache_dir} ...")
            shutil.rmtree(config.cache_dir, ignore_errors=True)
            config.cache_dir.mkdir(parents=True, exist_ok=True)
            logger.info("Cache purged.")
        except Exception as e:
            logger.error(f"Failed to purge cache: {e}")
            sys.exit(1)

    # Setup file logging - ONLY to file, not to terminal
    log_file = (
        config.output_dir / "logs" / f"pipeline_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
    )
    file_handler = logging.FileHandler(log_file)
    file_handler.setFormatter(
        logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
    )
    # Set handler only on root logger, no StreamHandler to terminal
    logging.getLogger().handlers = []  # Clear any existing handlers
    logging.getLogger().addHandler(file_handler)

    # Print minimal info to terminal
    print(f"Pipeline started. Logging to: {log_file}")
    print("=" * 80)

    logger.info("=" * 100)
    logger.info("=" * 100)
    logger.info("PIPELINE START - Research Assistant LLM-Driven Pipeline")
    logger.info("=" * 100)
    logger.info(f"Root Directory: {root_dir}")
    logger.info(f"Topic: {topic}")
    logger.info(f"Output Directory: {output_dir}")
    logger.info(f"Cache Directory: {cache_dir}")
    logger.info(f"LLM Provider: {config.llm_provider}")
    logger.info(f"Workers: {workers}")
    logger.info(f"Relevance Threshold: {relevance_threshold}")
    logger.info(f"Dry Run: {dry_run}")
    logger.info(f"Resume: {resume}")
    logger.info("=" * 100)

    try:
        # Initialize components
        logger.info("")
        logger.info("=" * 100)
        logger.info("[INITIALIZATION] Initializing pipeline components...")
        logger.info("=" * 100)
        cache_manager = CacheManager(config.cache_dir, config.cache.ttl_days)
        manifest_manager = ManifestManager(config.output_dir / "manifests")
        inventory_manager = InventoryManager(config.root_dir)
        pdf_parser = PDFParser(
            config.processing.ocr_language, config.processing.skip_ocr_if_text_exists
        )
        metadata_extractor = MetadataExtractor(config.crossref.enabled, config.crossref.email)
        dedup_manager = DedupManager(config.dedup.similarity_threshold, config.dedup.num_perm)
        summarizer = Summarizer(config.ollama.summarize_model, config.ollama.temperature)
        file_mover = FileMover(
            config.root_dir, manifest_manager, config.dry_run, config.move.create_symlinks
        )
        output_generator = OutputGenerator(config.output_dir)
        logger.info("All components initialized successfully")

        # Stage 1: Inventory
        logger.info("")
        logger.info("=" * 100)
        logger.info("[STAGE 1] INVENTORY - Scanning for PDFs and categories")
        logger.info("=" * 100)
        documents = inventory_manager.scan()
        summary = inventory_manager.summary()
        logger.info(f"Total PDFs found: {summary['total_documents']}")
        logger.info(f"Total categories: {summary['total_categories']}")
        logger.info(f"Categories list: {', '.join(inventory_manager.get_categories())}")

        # Stage 2: Parse and extract metadata
        logger.info("")
        logger.info("=" * 100)
        logger.info("[STAGE 2] METADATA EXTRACTION - LLM-based extraction and categorization")
        logger.info("=" * 100)

        paper_data = {}
        diversion_counters = {
            "short_text": 0,
            "missing_core_metadata": 0,
        }

        # --- Ensure manifests exist at process start ---
        manifest_manager.save_all()
        # Note: Index files will be built incrementally and written at the end
    except Exception as e:
        logger.error(f"Pipeline initialization failed: {e}")
        sys.exit(1)

    start_time = time.time()
    processed_count = 0
    total_papers = len(documents)

    def process_single_doc(doc):
        paper_id = doc.file_hash[:12]

        logger.info("-" * 100)
        logger.info(f"Processing: {doc.file_name}")
        logger.info(f"Paper ID: {paper_id}")
        logger.info(f"Current category: {doc.category}")

        # Check manifest to avoid reprocessing moved papers
        manifest = manifest_manager.get_manifest(doc.category)
        if manifest.should_skip(paper_id):
            logger.info(f"SKIPPED - Already processed (found in manifest)")
            logger.info("-" * 100)
            return None

        # Check cache
        cached_text = cache_manager.get_text(paper_id) if resume else None
        cached_metadata = cache_manager.get_metadata(paper_id) if resume else None

        if cached_text and cached_metadata:
            text, text_hash = cached_text
            metadata = cached_metadata
            logger.info(f"[CACHE HIT] Using cached text and metadata")
        else:
            logger.info(f"[EXTRACTION] Extracting text and metadata via LLM...")
            text, text_hash = pdf_parser.extract_text(doc.file_path, config.cache_dir)
            sections = pdf_parser.extract_sections(text)
            metadata = metadata_extractor._extract_with_llm(doc.file_path)
            if not metadata.get("abstract"):
                metadata["abstract"] = sections.get("abstract")
            cache_manager.set_text(paper_id, text, text_hash)
            cache_manager.set_metadata(paper_id, metadata)
            logger.info(f"[EXTRACTION COMPLETE] Title: {metadata.get('title', 'N/A')}")

        # Diversion logic: only divert if both title and authors are missing (unreadable metadata)
        title_missing = not bool(metadata.get("title"))
        authors_missing = len(metadata.get("authors", [])) == 0
        missing_core = title_missing and authors_missing
        if missing_core:
            logger.info(f"[DIVERSION] Missing core metadata (title AND authors)")
            logger.info(f"[DIVERSION] Moving to 'need_human_element' for manual review")
            manifest = manifest_manager.get_manifest(doc.category)
            manifest.add_paper(
                paper_id=paper_id,
                path=str(doc.file_path),
                content_hash=text_hash,
                original_category=doc.category,
            )
            diversion_counters["missing_core_metadata"] += 1
            file_mover.move_to_category(
                paper_id=paper_id,
                current_path=doc.file_path,
                from_category=doc.category,
                to_category="need_human_element",
                reason="Manual review required: missing_core_metadata",
            )
            logger.info(f"[DIVERSION COMPLETE] Moved to need_human_element")
            logger.info("-" * 100)
            # Update manifests after diversion
            manifest_manager.save_all()
            return None

        # Always run LLM scoring/categorization as a separate step
        # LLM categorization
        logger.info(f"[CATEGORIZATION] Running LLM scoring and categorization...")
        cat_score = metadata_extractor._llm_categorize_and_score(
            title=metadata.get("title", ""),
            abstract=metadata.get("abstract", ""),
            topic=config.topic,
            available_categories=inventory_manager.get_categories(),
        )
        metadata.update(cat_score)
        logger.info(f"[CATEGORIZATION] Category: {metadata.get('category', 'N/A')}")
        logger.info(
            f"[CATEGORIZATION] Relevance Score: {metadata.get('relevance_score', 'N/A')}/10"
        )
        logger.info(f"[CATEGORIZATION] Include: {metadata.get('include', False)}")

        # After categorization, update manifest and outputs
        # (status logic copied from main output block)
        if not metadata.get("include", False):
            status = "quarantined"
        elif doc.category == "need_human_element":
            status = "diverted"
        else:
            status = "active"
        manifest = manifest_manager.get_manifest(doc.category)
        manifest.add_paper(
            paper_id=paper_id,
            path=str(doc.file_path),
            content_hash=text_hash,
            status=status,
            original_category=doc.category,
        )
        manifest_manager.save_all()

        # Update index and CSV after each categorization
        # Build a minimal record for this paper
        record = {
            "paper_id": paper_id,
            "title": metadata.get("title", ""),
            "authors": metadata.get("authors", []),
            "year": metadata.get("year"),
            "venue": metadata.get("venue"),
            "doi": metadata.get("doi"),
            "category": doc.category,
            "original_category": doc.category,
            "relevance_score": metadata.get("relevance_score"),
            "include": metadata.get("include", False),
            "status": status,
            "duplicate_of": None,
            "is_duplicate": False,
            "original_path": str(doc.file_path),
            "current_path": str(doc.file_path),
            "bibtex": metadata.get("bibtex", ""),
            "summary_file": f"summaries/{doc.category}.md",
            "notes": "",
        }
        # Read current index, append or update this record, and write back
        index_path = config.output_dir / "index.jsonl"
        try:
            existing = {}
            if index_path.exists():
                with open(index_path, "r") as f:
                    for line in f:
                        try:
                            rec = json.loads(line)
                            existing[rec["paper_id"]] = rec
                        except Exception:
                            pass
            existing[paper_id] = record
            with open(index_path, "w") as f:
                for rec in existing.values():
                    f.write(json.dumps(rec) + "\n")
            # Also update CSV
            output_generator.write_csv(list(existing.values()), filename="index.csv")
        except Exception as e:
            logger.error(f"Failed to update index after categorization: {e}")

        # Time estimation (logged to file only, not to terminal)
        nonlocal processed_count
        processed_count += 1
        elapsed = time.time() - start_time
        avg_time = elapsed / processed_count if processed_count else 0
        remaining = total_papers - processed_count
        eta = avg_time * remaining
        logger.info(
            f"[PROGRESS] {processed_count}/{total_papers} papers processed | "
            f"Elapsed: {elapsed:.1f}s ({elapsed/60:.1f} min) | "
            f"Avg: {avg_time:.1f}s/paper | "
            f"Remaining: {remaining} papers | "
            f"ETA: {eta/60:.1f} min ({eta/3600:.1f} hrs)"
        )

        # Move paper to the correct category folder if categorization is available
        new_category = metadata.get("category")
        if new_category and new_category != doc.category:
            logger.info(f"[MOVE] Category change detected: {doc.category} -> {new_category}")
            new_path = file_mover.move_to_category(
                paper_id=paper_id,
                current_path=doc.file_path,
                from_category=doc.category,
                to_category=new_category,
                reason="LLM categorization",
            )
            logger.info(f"[MOVE COMPLETE] File moved to: {new_path}")
            # Update doc object in memory to reflect new location
            if new_path:
                doc.category = new_category
                doc.file_path = new_path
            # Update outputs after move
            manifest_manager.save_all()
            # Note: Don't clear index files here - they will be rebuilt at the end

        logger.info("-" * 100)

        # Store paper data
        return (
            paper_id,
            {
                "doc": doc,
                "text": text,
                "text_hash": text_hash,
                "metadata": metadata,
                "sections": pdf_parser.extract_sections(text),
                "include": metadata.get("include", False),
                "relevance_score": metadata.get("relevance_score"),
            },
        )

    # Parallelize LLM processing for extraction, categorization, and scoring
    logger.info(f"Starting parallel processing with {workers} workers...")
    logger.info("")

    with ThreadPoolExecutor(max_workers=workers) as executor:
        results = list(
            tqdm(
                executor.map(process_single_doc, documents),
                total=len(documents),
                desc="Processing PDFs",
                unit="paper",
                leave=True,
                ncols=80,
                # Estimate 22.5 seconds per paper on average (midpoint of 20-25s)
                # tqdm will auto-adjust based on actual timing
                smoothing=0.1,  # More responsive to recent timing changes
                bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}]",
            )
        )
    # Filter out None results (skipped/diverted)
    for result in results:
        if result is not None:
            paper_id, data = result
            paper_data[paper_id] = data

    # --- Manifest tracking for all processed papers ---
    for paper_id, data in paper_data.items():
        doc = data["doc"]
        # Determine status
        if not data.get("include", False):
            status = "quarantined"
        elif doc.category == "need_human_element":
            status = "diverted"
        else:
            status = "active"
        # Add to manifest of final category
        manifest = manifest_manager.get_manifest(doc.category)
        manifest.add_paper(
            paper_id=paper_id,
            path=str(doc.file_path),
            content_hash=data["text_hash"],
            status=status,
            original_category=doc.category,
        )
    manifest_manager.save_all()
    logger.info("All manifests saved after processing")

    total_diverted_unique = (
        sum(1 for _ in (manifest_manager.get_manifest("need_human_element").entries.values()))
        if "need_human_element" in manifest_manager.manifests
        else 0
    )
    if total_diverted_unique:
        logger.info("")
        logger.info("DIVERSION SUMMARY (need_human_element):")
        logger.info(f"  Total diverted (unique papers): {total_diverted_unique}")
        logger.info(
            "  Reasons: "
            + ", ".join(f"{reason}={count}" for reason, count in diversion_counters.items())
        )

    # Stage 3: Deduplication
    logger.info("")
    logger.info("=" * 100)
    logger.info("[STAGE 3] DEDUPLICATION - Detecting and moving duplicates")
    logger.info("=" * 100)

    # Exact duplicates
    exact_dups = dedup_manager.find_exact_duplicates(documents)

    # Near duplicates
    paper_texts = {pid: data["text"] for pid, data in paper_data.items()}
    paper_names = {pid: data["doc"].file_name for pid, data in paper_data.items()}
    near_dups = dedup_manager.find_near_duplicates(paper_texts, paper_names)

    # Move duplicates
    for canonical, duplicates in near_dups.items():
        for dup_id in duplicates:
            if dup_id in paper_data:
                dup_data = paper_data[dup_id]
                logger.info(
                    f"[STAGE 3.4][{dup_data['doc'].file_name}] Deduplication: moving to repeated (duplicate of {canonical})"
                )
                new_path = file_mover.move_to_repeated(
                    paper_id=dup_id,
                    current_path=dup_data["doc"].file_path,
                    from_category=dup_data["doc"].category,
                    canonical_id=canonical,
                )
                # Mark in manifest
                manifest = manifest_manager.get_manifest(dup_data["doc"].category)
                manifest.mark_duplicate(dup_id, canonical)
                # Update doc object in memory
                if new_path:
                    dup_data["doc"].category = "repeated"
                    dup_data["doc"].file_path = new_path
                # Update manifests after dedup move
                manifest_manager.save_all()

    manifest_manager.save_all()

    # Stage 4: Embeddings and Scoring (now handled in parallel processing)
    logger.info("")
    logger.info("=" * 100)
    logger.info("[STAGE 4] SCORING - LLM Relevance Scoring and Categorization (completed)")
    logger.info("=" * 100)

    # Stage 5: Quarantine low-relevance papers
    logger.info("")
    logger.info("=" * 100)
    logger.info("[STAGE 5] QUARANTINE - Moving papers with include=False")
    logger.info("=" * 100)

    # Quarantine papers not included by LLM
    quarantine_count = 0
    for paper_id, data in list(paper_data.items()):
        if not data.get("include", False):
            score = data.get("relevance_score")
            score_str = f"{score:.1f}" if isinstance(score, (int, float)) else "N/A"
            logger.info(
                f"[QUARANTINE] {data['doc'].file_name} | Score: {score_str} | Reason: Include=False"
            )
            new_path = file_mover.move_to_quarantined(
                paper_id=paper_id,
                current_path=data["doc"].file_path,
                from_category=data["doc"].category,
                reason=f"LLM include: False, score: {score_str}",
            )
            quarantine_count += 1
            # Update doc object in memory
            if new_path:
                data["doc"].category = "quarantined"
                data["doc"].file_path = new_path
            # Update manifests after quarantine
            manifest_manager.save_all()

    logger.info(f"Total papers quarantined: {quarantine_count}")

    # Stage 6: Summarization
    logger.info("")
    logger.info("=" * 100)
    logger.info("[STAGE 6] SUMMARIZATION - Generating LLM summaries for included papers")
    logger.info("=" * 100)

    category_summaries = {}

    def summarize_paper_task(paper_id, data):
        """Summarize a single paper and return result."""
        # Only summarize included papers
        if not data.get("include", False):
            return None

        metadata = data["metadata"]
        sections = data["sections"]

        summary = summarizer.summarize_paper(
            title=metadata.get("title", ""),
            abstract=sections.get("abstract"),
            intro=sections.get("introduction"),
            topic=config.topic,
            metadata=metadata,
            full_text=data.get("text", None),
        )

        return paper_id, data, summary

    # Parallelize summarization with 2 workers
    included_papers = [
        (pid, data) for pid, data in paper_data.items() if data.get("include", False)
    ]
    logger.info(f"Generating summaries for {len(included_papers)} included papers...")

    with ThreadPoolExecutor(max_workers=2) as executor:
        results = list(
            tqdm(
                executor.map(lambda item: summarize_paper_task(item[0], item[1]), included_papers),
                total=len(included_papers),
                desc="Generating summaries",
                unit="paper",
                ncols=80,
                bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}]",
            )
        )

    # Process results
    for result in results:
        if result is None:
            continue

        paper_id, data, summary = result
        data["summary"] = summary

        # Group by category
        category = data["doc"].category
        if category not in category_summaries:
            category_summaries[category] = []

        metadata = data["metadata"]
        category_summaries[category].append(
            {
                "title": metadata.get("title", ""),
                "authors": metadata.get("authors", []),
                "year": metadata.get("year"),
                "venue": metadata.get("venue"),
                "relevance_score": data.get("relevance_score"),
                "summary": summary,
                "bibtex": metadata.get("bibtex", ""),
            }
        )

    # Write category summaries
    logger.info("")
    logger.info("Writing category summary markdown files...")
    for category, summaries in category_summaries.items():
        output_generator.write_category_summary(category, summaries)
        logger.info(f"  - {category}.md ({len(summaries)} papers)")

    # Stage 7: Generate Outputs
    logger.info("")
    logger.info("=" * 100)
    logger.info("[STAGE 7] OUTPUT GENERATION - Writing final index files")
    logger.info("=" * 100)

    # Build output records
    records = []
    for paper_id, data in paper_data.items():
        metadata = data["metadata"]
        doc = data["doc"]
        # Determine status
        if not data.get("include", False):
            status = "quarantined"
        elif doc.category == "need_human_element":
            status = "diverted"
        else:
            status = "active"
        record = {
            "paper_id": paper_id,
            "title": metadata.get("title", ""),
            "authors": metadata.get("authors", []),
            "year": metadata.get("year"),
            "venue": metadata.get("venue"),
            "doi": metadata.get("doi"),
            "category": doc.category,
            "original_category": doc.category,
            "relevance_score": data.get("relevance_score"),
            "include": data.get("include", False),
            "status": status,
            "duplicate_of": None,
            "is_duplicate": False,
            "original_path": str(doc.file_path),
            "current_path": str(doc.file_path),
            "bibtex": metadata.get("bibtex", ""),
            "summary_file": f"summaries/{doc.category}.md",
            "notes": "",
        }
        records.append(record)

    # Write outputs
    output_generator.write_jsonl(records)
    output_generator.write_csv(records)
    logger.info(f"Wrote {len(records)} records to index.jsonl and index.csv")
    # output_generator.write_statistics(stats, "statistics.json")

    # Final summary
    logger.info("")
    logger.info("=" * 100)
    logger.info("PIPELINE COMPLETE")
    logger.info("=" * 100)
    logger.info(f"Total papers processed: {len(paper_data)}")
    logger.info(
        f"Papers included: {sum(1 for d in paper_data.values() if d.get('include', False))}"
    )
    logger.info(
        f"Papers quarantined: {sum(1 for d in paper_data.values() if not d.get('include', False))}"
    )
    logger.info(f"Output directory: {config.output_dir}")
    logger.info("=" * 100)

    # Print minimal summary to terminal
    print("\n" + "=" * 80)
    print("✓ Pipeline Complete!")
    print(f"  Processed: {len(paper_data)} papers")
    print(f"  Outputs: {config.output_dir}")
    print(f"  Log file: {log_file}")
    print("=" * 80)


if __name__ == "__main__":
    cli()
