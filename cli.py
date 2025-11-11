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

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
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

    # Setup file logging
    log_file = (
        config.output_dir / "logs" / f"pipeline_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
    )
    file_handler = logging.FileHandler(log_file)
    file_handler.setFormatter(
        logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
    )
    logging.getLogger().addHandler(file_handler)

    logger.info("=" * 80)
    logger.info("Research Assistant LLM-Driven Pipeline Starting")
    logger.info(f"Root Directory: {root_dir}")
    logger.info(f"Topic: {topic}")
    logger.info(f"Output Directory: {output_dir}")
    logger.info(f"Dry Run: {dry_run}")
    logger.info("=" * 80)

    try:
        # Initialize components
        logger.info("[INIT] Initializing pipeline components...")
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

        # Stage 1: Inventory
        logger.info("\n" + "=" * 80)
        logger.info("[STAGE 1] Inventory: Scanning for PDFs and categories...")
        logger.info("=" * 80)
        documents = inventory_manager.scan()
        summary = inventory_manager.summary()
        logger.info(f"Total PDFs: {summary['total_documents']}")
        logger.info(f"Categories: {summary['total_categories']}")

        # Stage 2: Parse and extract metadata
        logger.info("\n" + "=" * 80)
        logger.info("[STAGE 2] LLM Metadata Extraction...")
        logger.info("=" * 80)

        paper_data = {}
        diversion_counters = {
            "short_text": 0,
            "missing_core_metadata": 0,
        }

        # --- Ensure manifests, index, and CSV exist at process start ---
        manifest_manager.save_all()
        output_generator.write_jsonl([], filename="index.jsonl")
        output_generator.write_csv([], filename="index.csv")
    except Exception as e:
        logger.error(f"Pipeline initialization failed: {e}")
        sys.exit(1)

    start_time = time.time()
    processed_count = 0
    total_papers = len(documents)

    def log_to_file(msg):
        logger.info(msg)

    def process_single_doc(doc):
        paper_id = doc.file_hash[:12]

        # Check manifest to avoid reprocessing moved papers
        manifest = manifest_manager.get_manifest(doc.category)
        if manifest.should_skip(paper_id):
            return None

        # Check cache
        cached_text = cache_manager.get_text(paper_id) if resume else None
        cached_metadata = cache_manager.get_metadata(paper_id) if resume else None

        if cached_text and cached_metadata:
            text, text_hash = cached_text
            metadata = cached_metadata
            logger.info(f"[CACHE] Used cached text and metadata for {doc.file_name}")
        else:
            logger.info(f"[LLM][EXTRACT] Extracting metadata for {doc.file_name} using LLM...")
            text, text_hash = pdf_parser.extract_text(doc.file_path, config.cache_dir)
            sections = pdf_parser.extract_sections(text)
            metadata = metadata_extractor._extract_with_llm(doc.file_path)
            if not metadata.get("abstract"):
                metadata["abstract"] = sections.get("abstract")
            cache_manager.set_text(paper_id, text, text_hash)
            cache_manager.set_metadata(paper_id, metadata)

        # Diversion logic: only divert if both title and authors are missing (unreadable metadata)
        title_missing = not bool(metadata.get("title"))
        authors_missing = len(metadata.get("authors", [])) == 0
        missing_core = title_missing and authors_missing
        if missing_core:
            logger.info(
                f"[STAGE 3.1][{doc.file_name}] Diversion: missing core metadata, diverting to need_human_element."
            )
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
            logger.info(
                f"[DIVERT] {doc.file_name} diverted to need_human_element (unreadable metadata)"
            )
            # Update outputs after diversion
            manifest_manager.save_all()
            output_generator.write_jsonl([], filename="index.jsonl")
            output_generator.write_csv([], filename="index.csv")
            return None

        # Always run LLM scoring/categorization as a separate step
        # LLM categorization
        log_to_file(f"[STAGE 3.2][{doc.file_name}] LLM scoring/categorization...")
        cat_score = metadata_extractor._llm_categorize_and_score(
            title=metadata.get("title", ""),
            abstract=metadata.get("abstract", ""),
            topic=config.topic,
            available_categories=summary["categories"] if "categories" in summary else None,
        )
        metadata.update(cat_score)

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
            log_to_file(f"[ERROR] Failed to update index after categorization: {e}")

        # Time estimation
        nonlocal processed_count
        processed_count += 1
        elapsed = time.time() - start_time
        avg_time = elapsed / processed_count if processed_count else 0
        remaining = total_papers - processed_count
        eta = avg_time * remaining
        log_to_file(
            f"[TIME] Processed {processed_count}/{total_papers} papers. Elapsed: {elapsed:.1f}s, ETA: {eta/60:.1f} min"
        )

        # Sleep after every 10 PDFs
        if processed_count % 10 == 0:
            log_to_file(f"[SLEEP] Sleeping for 10 minutes after {processed_count} papers...")
            time.sleep(600)

        # Move paper to the correct category folder if categorization is available
        new_category = metadata.get("category")
        if new_category and new_category != doc.category:
            logger.info(
                f"[STAGE 3.3][{doc.file_name}] Moving to category '{new_category}' by LLM categorization."
            )
            new_path = file_mover.move_to_category(
                paper_id=paper_id,
                current_path=doc.file_path,
                from_category=doc.category,
                to_category=new_category,
                reason="LLM categorization",
            )
            logger.info(
                f"[MOVE] {doc.file_name} moved to category '{new_category}' by LLM categorization."
            )
            # Update doc object in memory to reflect new location
            if new_path:
                doc.category = new_category
                doc.file_path = new_path
            # Update outputs after move
            manifest_manager.save_all()
            output_generator.write_jsonl([], filename="index.jsonl")
            output_generator.write_csv([], filename="index.csv")

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
    with ThreadPoolExecutor(max_workers=3) as executor:
        results = list(
            tqdm(
                executor.map(process_single_doc, documents),
                total=len(documents),
                desc="Processing PDFs",
                leave=True,
                ncols=100,
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
    # Only log big stage labels to console
    print("All manifests saved after deduplication and moves.")
    total_diverted_unique = (
        sum(1 for _ in (manifest_manager.get_manifest("need_human_element").entries.values()))
        if "need_human_element" in manifest_manager.manifests
        else 0
    )
    if total_diverted_unique:
        logger.info("Diversion summary (need_human_element):")
        logger.info(f"  Total diverted (unique papers): {total_diverted_unique}")
        logger.info(
            "  Reasons (counts; a paper may appear in multiple reason tallies): "
            + ", ".join(f"{reason}={count}" for reason, count in diversion_counters.items())
        )

    # Stage 3: Deduplication
    print("\n" + "=" * 80)
    print("[STAGE 3] Deduplication: Detecting and moving duplicates...")
    print("=" * 80)

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
                # Update outputs after dedup move
                manifest_manager.save_all()
                output_generator.write_jsonl([], filename="index.jsonl")
                output_generator.write_csv([], filename="index.csv")

    manifest_manager.save_all()

    # Stage 4: Embeddings and Scoring
    print("\n" + "=" * 80)
    print("[STAGE 4] LLM Relevance Scoring and Categorization...")
    print("=" * 80)

    # Stage 5: Quarantine low-relevance papers
    print("\n" + "=" * 80)
    print("[STAGE 5] Quarantine Unrelated Papers (LLM include: False)...")
    print("=" * 80)

    # Quarantine papers not included by LLM
    for paper_id, data in list(paper_data.items()):
        if not data.get("include", False):
            score = data.get("relevance_score")
            score_str = f"{score:.1f}" if isinstance(score, (int, float)) else "N/A"
            logger.info(
                f"[STAGE 3.5][{data['doc'].file_name}] Quarantine: LLM include: False, score: {score_str}"
            )
            new_path = file_mover.move_to_quarantined(
                paper_id=paper_id,
                current_path=data["doc"].file_path,
                from_category=data["doc"].category,
                reason=f"LLM include: False, score: {score_str}",
            )
            # Update doc object in memory
            if new_path:
                data["doc"].category = "quarantined"
                data["doc"].file_path = new_path
            # Update outputs after quarantine
            manifest_manager.save_all()
            output_generator.write_jsonl([], filename="index.jsonl")
            output_generator.write_csv([], filename="index.csv")

    # Stage 6: Summarization
    print("\n" + "=" * 80)
    print("[STAGE 6] Generating Summaries with LLM...")
    print("=" * 80)

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
    with ThreadPoolExecutor(max_workers=2) as executor:
        paper_items = list(paper_data.items())
        results = list(
            tqdm(
                executor.map(lambda item: summarize_paper_task(item[0], item[1]), paper_items),
                total=len(paper_items),
                desc="Generating summaries",
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
    for category, summaries in category_summaries.items():
        output_generator.write_category_summary(category, summaries)
        logger.info(f"Wrote summary markdown for category: {category}")

    # Stage 7: Generate Outputs
    print("\n" + "=" * 80)
    print("[STAGE 7] Writing Outputs...")
    print("=" * 80)

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
    logger.info("Wrote index.jsonl and index.csv outputs.")
    # output_generator.write_statistics(stats, "statistics.json")

    print("\n" + "=" * 80)
    print("Pipeline Complete!")
    print(f"Processed: {len(paper_data)} papers")
    print(f"Outputs written to: {config.output_dir}")
    print("=" * 80)


if __name__ == "__main__":
    cli()
