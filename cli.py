"""Main CLI for research assistant pipeline - Refactored for dynamic LLM-driven taxonomy."""

import json
import logging
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional

import click
from tqdm import tqdm

from config import Config
from core.dedup import DedupManager
from core.inventory import InventoryManager, PDFDocument
from core.manifest import ManifestManager
from core.metadata import MetadataExtractor
from core.mover import FileMover
from core.outputs import OutputGenerator
from core.parser import PDFParser
from core.summarizer import Summarizer
from core.taxonomy import TaxonomyGenerator
from utils.cache_manager import CacheManager
from utils.hash import file_hash

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[],
)
logger = logging.getLogger(__name__)


@click.group()
def cli():
    """Research Assistant - Intelligent PDF analysis pipeline with dynamic LLM categorization."""
    pass


@cli.command(short_help="Process PDFs through the full analysis pipeline")
@click.option(
    "--root-dir",
    required=True,
    type=click.Path(exists=True, path_type=Path),
    help="Root directory containing PDFs to analyze",
)
@click.option("--topic", required=True, type=str, help="Research topic for category generation")
@click.option(
    "--output-dir",
    default=None,
    type=click.Path(path_type=Path),
    help="Output directory (default: ~/Desktop/output_DD_MMM_HH_MM)",
)
@click.option(
    "--cache-dir", default="./cache", type=click.Path(path_type=Path), help="Cache directory"
)
@click.option(
    "--purge-cache",
    is_flag=True,
    help="Purge the cache directory before processing",
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
    help="LLM provider to use",
)
@click.option("--dry-run", is_flag=True, help="Run without moving files")
@click.option("--resume", is_flag=True, help="Resume from previous run (skip analyzed papers)")
@click.option(
    "--min-topic-relevance",
    type=int,
    default=5,
    help="Minimum topic relevance (1-10) to avoid quarantine",
)
@click.option("--workers", type=int, default=2, help="Number of parallel workers")
@click.option(
    "--force-regenerate-categories",
    is_flag=True,
    help="Force regeneration of category taxonomy (ignore cache)",
)
def process(
    root_dir: Path,
    topic: str,
    output_dir: Path,
    cache_dir: Path,
    purge_cache: bool,
    config_file: Optional[Path],
    llm_provider: Optional[str],
    dry_run: bool,
    resume: bool,
    min_topic_relevance: int,
    workers: int,
    force_regenerate_categories: bool,
):
    """Process PDFs through the full analysis pipeline with dynamic LLM-driven categorization."""

    # Load configuration
    if config_file:
        config = Config.from_yaml(config_file)
    else:
        config = Config()

    # Set default output directory to Desktop with timestamp
    if output_dir is None:
        desktop = Path.home() / "Desktop"
        timestamp = datetime.now().strftime("%d_%b_%H_%M")
        output_dir = desktop / f"output_{timestamp}"
        logger.info(f"Using default output directory: {output_dir}")

    # Override with CLI parameters
    config.root_dir = root_dir
    config.topic = topic
    config.output_dir = output_dir
    config.cache_dir = cache_dir
    config.dry_run = dry_run
    config.resume = resume
    config.processing.workers = workers
    config.scoring.min_topic_relevance = min_topic_relevance

    if llm_provider:
        import os
        os.environ["LLM_PROVIDER"] = llm_provider
        config.llm_provider = llm_provider

    # Setup directories
    config.setup_directories()

    # Setup file logging
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = output_dir / "logs" / f"pipeline_{timestamp}.log"
    file_handler = logging.FileHandler(log_file)
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(
        logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
    )
    logging.getLogger().addHandler(file_handler)

    logger.info("=" * 100)
    logger.info("RESEARCH ASSISTANT PIPELINE - DYNAMIC LLM CATEGORIZATION")
    logger.info("=" * 100)
    logger.info(f"Root directory: {root_dir}")
    logger.info(f"Research topic: {topic}")
    logger.info(f"Output directory: {output_dir}")
    logger.info(f"Cache directory: {cache_dir}")
    logger.info(f"LLM Provider: {config.llm_provider}")
    logger.info(f"Workers: {workers}")
    logger.info(f"Min topic relevance: {min_topic_relevance}")
    logger.info(f"Resume mode: {resume}")
    logger.info(f"Dry run: {dry_run}")
    logger.info("=" * 100)

    # Handle cache purging
    if purge_cache:
        logger.info("")
        logger.info("=" * 100)
        logger.info("[PURGE] Clearing cache directory")
        logger.info("=" * 100)
        cache_db = cache_dir / "cache.db"
        if cache_db.exists():
            cache_db.unlink()
            logger.info(f"Deleted cache database: {cache_db}")
        ocr_cache = cache_dir / "ocr"
        if ocr_cache.exists():
            import shutil
            shutil.rmtree(ocr_cache)
            logger.info(f"Deleted OCR cache: {ocr_cache}")
        logger.info("Cache purged successfully")

    # Initialize components
    cache_manager = CacheManager(cache_dir)
    metadata_extractor = MetadataExtractor(
        use_crossref=config.crossref.enabled,
        crossref_email=config.crossref.email,
    )
    pdf_parser = PDFParser()
    dedup_manager = DedupManager(
        similarity_threshold=config.dedup.similarity_threshold,
        num_perm=config.dedup.num_perm,
    )
    manifest_manager = ManifestManager(output_dir / "manifests")
    file_mover = FileMover(
        root_dir=root_dir,
        manifest_manager=manifest_manager,
        dry_run=dry_run,
        create_symlinks=config.move.create_symlinks,
    )
    output_generator = OutputGenerator(output_dir)
    summarizer = Summarizer()
    taxonomy_generator = TaxonomyGenerator(cache_dir=cache_dir, output_dir=output_dir)

    # ================================================================================
    # PASS 1: GENERATE CATEGORY TAXONOMY FROM TOPIC
    # ================================================================================
    logger.info("")
    logger.info("=" * 100)
    logger.info("[PASS 1] TAXONOMY GENERATION - LLM generates categories from topic")
    logger.info("=" * 100)

    try:
        categories = taxonomy_generator.generate_categories(
            topic=topic, force_regenerate=force_regenerate_categories
        )
        logger.info(f"Generated {len(categories)} categories:")
        for name, definition in categories.items():
            logger.info(f"  • {name}: {definition[:100]}...")
    except Exception as e:
        logger.error(f"Failed to generate taxonomy: {e}")
        sys.exit(1)

    # ================================================================================
    # PASS 2: INVENTORY ALL PDFS
    # ================================================================================
    logger.info("")
    logger.info("=" * 100)
    logger.info("[PASS 2] INVENTORY - Discovering PDFs in root directory")
    logger.info("=" * 100)

    inventory_manager = InventoryManager(root_dir)
    documents = inventory_manager.scan()

    if not documents:
        logger.error("No PDF files found in root directory")
        sys.exit(1)

    logger.info(f"Discovered {len(documents)} PDF files")

    # Detect and move file-level duplicates (same file_hash)
    hash_to_docs: dict[str, PDFDocument] = {}
    duplicates: list[tuple[PDFDocument, PDFDocument]] = []
    unique_documents: list[PDFDocument] = []

    for doc in documents:
        if doc.file_hash in hash_to_docs:
            # Duplicate found
            canonical = hash_to_docs[doc.file_hash]
            duplicates.append((doc, canonical))
            logger.info(f"[DUPLICATE] {doc.file_name} is identical to {canonical.file_name}")
        else:
            # First occurrence - this is the canonical version
            hash_to_docs[doc.file_hash] = doc
            unique_documents.append(doc)
    
    # Move duplicates to duplicates/ folder
    if duplicates:
        duplicates_dir = root_dir / "duplicates"
        duplicates_dir.mkdir(exist_ok=True)
        logger.info(f"Moving {len(duplicates)} duplicate files to duplicates/")
        
        for dup_doc, canonical_doc in duplicates:
            try:
                dest_path = duplicates_dir / dup_doc.file_name
                dup_doc.file_path.rename(dest_path)
                logger.info(f"  → {dup_doc.file_name} (duplicate of {canonical_doc.file_name})")
            except Exception as e:
                logger.error(f"Failed to move duplicate {dup_doc.file_name}: {e}")
    
    # Use only unique documents for processing
    documents = unique_documents
    logger.info(f"Processing {len(documents)} unique PDF files")

    # Load existing index for resume mode
    existing_index = {}
    if resume:
        index_path = output_dir / "index.jsonl"
        if index_path.exists():
            logger.info("Resume mode: Loading existing index...")
            try:
                with open(index_path, "r") as f:
                    for line in f:
                        record = json.loads(line.strip())
                        if record.get("analyzed"):
                            existing_index[record["paper_id"]] = record
                logger.info(f"Found {len(existing_index)} previously analyzed papers")
            except Exception as e:
                logger.warning(f"Failed to load existing index: {e}")

    # ================================================================================
    # PASS 3: METADATA EXTRACTION & CLASSIFICATION
    # ================================================================================
    logger.info("")
    logger.info("=" * 100)
    logger.info("[PASS 3] PROCESSING - Metadata extraction & classification")
    logger.info("=" * 100)

    paper_data = {}
    unreadable_papers = []
    start_time = time.time()
    processed_count = 0
    total_papers = len(documents)

    def process_single_doc(doc):
        """Process a single document: metadata extraction + classification."""
        nonlocal processed_count

        paper_id = doc.file_hash
        
        # Resume: Skip if already analyzed
        if resume and paper_id in existing_index:
            logger.info(f"[RESUME] Skipping {doc.file_name} (already analyzed)")
            return None

        logger.info("=" * 100)
        logger.info(f"[PROCESSING] {doc.file_name}")
        logger.info("=" * 100)

        # Step 1: Extract metadata
        logger.info(f"[STEP 1] Extracting metadata...")
        
        # Check cache if resume mode
        metadata = None
        text = None
        text_hash = None
        
        if resume:
            # Try to load from cache
            metadata = cache_manager.get_metadata(paper_id)
            cached_text = cache_manager.get_text(paper_id)
            if metadata and cached_text:
                text, text_hash = cached_text
                logger.info(f"[CACHE] Loaded cached metadata for {doc.file_name}")

        if not metadata:
            try:
                metadata = metadata_extractor.extract(doc.file_path, topic=topic)
                text, content_hash = pdf_parser.extract_text(doc.file_path)
                text_hash = file_hash(doc.file_path)
                
                # Cache metadata and text
                cache_manager.set_metadata(paper_id, metadata)
                cache_manager.set_text(paper_id, text, text_hash)
            except Exception as e:
                logger.error(f"[ERROR] Metadata extraction failed: {e}")
                metadata = {}
                text = ""
                text_hash = ""

        # Check if readable
        if not metadata or not metadata.get("title"):
            logger.warning(f"[UNREADABLE] {doc.file_name} - No metadata extracted")
            unreadable_papers.append((paper_id, doc, text_hash))
            return None

        # Step 2: Classify with multi-category scoring
        logger.info(f"[STEP 2] Classifying paper across {len(categories)} categories...")
        
        try:
            classification = metadata_extractor.classify_paper_with_scores(
                title=metadata.get("title", ""),
                abstract=metadata.get("abstract", ""),
                topic=topic,
                categories=categories,
            )
            
            topic_rel = classification.get("topic_relevance", 1)
            best_cat = classification.get("best_category")
            cat_scores = classification.get("category_scores", {})
            reasoning = classification.get("reasoning", "")
            
            logger.info(f"[CLASSIFICATION] Topic relevance: {topic_rel}/10")
            logger.info(f"[CLASSIFICATION] Best category: {best_cat} (score: {cat_scores.get(best_cat, 'N/A')}/10)")
            logger.info(f"[CLASSIFICATION] Reasoning: {reasoning[:150]}...")
            
        except Exception as e:
            logger.error(f"[ERROR] Classification failed: {e}")
            topic_rel = 1
            best_cat = list(categories.keys())[0] if categories else "uncategorized"
            cat_scores = {name: 1 for name in categories.keys()}
            reasoning = f"Classification failed: {e}"

        # Update progress
        processed_count += 1
        elapsed = time.time() - start_time
        avg_time = elapsed / processed_count if processed_count else 0
        remaining = total_papers - processed_count
        eta = avg_time * remaining
        logger.info(
            f"[PROGRESS] {processed_count}/{total_papers} papers | "
            f"Elapsed: {elapsed/60:.1f} min | "
            f"Avg: {avg_time:.1f}s/paper | "
            f"ETA: {eta/60:.1f} min"
        )
        logger.info("-" * 100)

        return (
            paper_id,
            {
                "doc": doc,
                "metadata": metadata,
                "text": text,
                "text_hash": text_hash,
                "topic_relevance": topic_rel,
                "best_category": best_cat,
                "category_scores": cat_scores,
                "reasoning": reasoning,
            },
        )

    # Parallelize processing
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
                smoothing=0.1,
                bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}]",
                position=0,
                dynamic_ncols=False,
            )
        )

    # Filter out None results (skipped/unreadable)
    for result in results:
        if result is not None:
            paper_id, data = result
            paper_data[paper_id] = data

    logger.info(f"Successfully processed {len(paper_data)} papers")
    logger.info(f"Unreadable papers: {len(unreadable_papers)}")

    # ================================================================================
    # PASS 4: MOVE PAPERS TO CATEGORY FOLDERS
    # ================================================================================
    logger.info("")
    logger.info("=" * 100)
    logger.info("[PASS 4] ORGANIZATION - Moving papers to category folders")
    logger.info("=" * 100)

    for paper_id, data in paper_data.items():
        doc = data["doc"]
        best_category = data["best_category"]
        topic_rel = data["topic_relevance"]
        cat_score = data["category_scores"].get(best_category, 0)
        
        # Determine destination based on topic relevance
        if topic_rel < min_topic_relevance:
            destination = "quarantined"
            logger.info(
                f"[QUARANTINE] {doc.file_name} | Topic relevance: {topic_rel}/10 < {min_topic_relevance}"
            )
        else:
            destination = best_category
            logger.info(
                f"[MOVE] {doc.file_name} → {destination} | "
                f"Category score: {cat_score}/10 | Topic relevance: {topic_rel}/10"
            )

        # Move paper
        dest_dir = root_dir / destination
        dest_dir.mkdir(parents=True, exist_ok=True)
        new_path = dest_dir / doc.file_path.name

        if not dry_run and doc.file_path != new_path:
            try:
                import shutil
                shutil.move(str(doc.file_path), str(new_path))
                data["doc"].file_path = new_path
                logger.info(f"[MOVED] {doc.file_name} to {destination}/")
            except Exception as e:
                logger.error(f"[ERROR] Failed to move {doc.file_name}: {e}")

    # Handle unreadable papers
    if unreadable_papers:
        logger.info("")
        logger.info(f"Moving {len(unreadable_papers)} unreadable papers to need_human_element/")
        need_human_dir = root_dir / "need_human_element"
        need_human_dir.mkdir(parents=True, exist_ok=True)
        
        for paper_id, doc, text_hash in unreadable_papers:
            new_path = need_human_dir / doc.file_path.name
            if not dry_run and doc.file_path != new_path:
                try:
                    import shutil
                    shutil.move(str(doc.file_path), str(new_path))
                    logger.info(f"[MOVED] {doc.file_name} to need_human_element/")
                except Exception as e:
                    logger.error(f"[ERROR] Failed to move {doc.file_name}: {e}")

    # ================================================================================
    # PASS 5: DEDUPLICATION
    # ================================================================================
    logger.info("")
    logger.info("=" * 100)
    logger.info("[PASS 5] DEDUPLICATION - Detecting and moving duplicates")
    logger.info("=" * 100)

    # Find duplicates
    paper_texts = {pid: data["text"] for pid, data in paper_data.items()}
    paper_names = {pid: data["doc"].file_name for pid, data in paper_data.items()}
    near_dups = dedup_manager.find_near_duplicates(paper_texts, paper_names)

    # Move duplicates
    duplicate_count = 0
    for canonical_id, dup_ids in near_dups.items():
        for dup_id in dup_ids:
            if dup_id in paper_data:
                dup_data = paper_data[dup_id]
                logger.info(
                    f"[DUPLICATE] {dup_data['doc'].file_name} is duplicate of {canonical_id}"
                )

                # Move to repeated folder
                repeated_dir = root_dir / "repeated"
                repeated_dir.mkdir(parents=True, exist_ok=True)
                new_path = repeated_dir / dup_data["doc"].file_path.name
                
                if not dry_run and dup_data["doc"].file_path != new_path:
                    try:
                        import shutil
                        shutil.move(str(dup_data["doc"].file_path), str(new_path))
                        dup_data["doc"].file_path = new_path
                        dup_data["canonical_id"] = canonical_id
                        duplicate_count += 1
                        logger.info(f"[MOVED] {dup_data['doc'].file_name} to repeated/")
                    except Exception as e:
                        logger.error(f"[ERROR] Failed to move duplicate: {e}")

    logger.info(f"Total duplicates moved: {duplicate_count}")

    # ================================================================================
    # PASS 6: UPDATE MANIFESTS
    # ================================================================================
    logger.info("")
    logger.info("=" * 100)
    logger.info("[PASS 6] MANIFESTS - Updating category manifests")
    logger.info("=" * 100)

    for paper_id, data in paper_data.items():
        doc = data["doc"]
        metadata = data["metadata"]
        topic_rel = data["topic_relevance"]
        best_category = data["best_category"]
        cat_scores = data["category_scores"]
        reasoning = data["reasoning"]
        
        # Determine final category (where the file ended up)
        if topic_rel < min_topic_relevance:
            final_category = "quarantined"
        elif data.get("canonical_id"):
            final_category = "repeated"
        else:
            final_category = best_category

        # Add to manifest
        manifest = manifest_manager.get_manifest(final_category)
        manifest.add_paper(
            paper_id=paper_id,
            title=metadata.get("title", ""),
            path=str(doc.file_path),
            content_hash=data["text_hash"],
            classification_reasoning=reasoning,
            relevance_score=cat_scores.get(best_category, 0),
            topic_relevance=topic_rel,
            analyzed=True,
        )
        
        # Mark as duplicate if applicable
        if data.get("canonical_id"):
            manifest.mark_duplicate(paper_id, data["canonical_id"])

    # Handle unreadable papers manifest
    if unreadable_papers:
        need_human_manifest = manifest_manager.get_manifest("need_human_element")
        for paper_id, doc, text_hash in unreadable_papers:
            need_human_manifest.add_paper(
                paper_id=paper_id,
                title=doc.file_name,
                path=str(need_human_dir / doc.file_path.name),
                content_hash=text_hash or "",
                classification_reasoning="Unreadable - no metadata extracted",
                relevance_score=None,
                topic_relevance=None,
                analyzed=True,
            )

    manifest_manager.save_all()
    logger.info("All manifests saved")

    # ================================================================================
    # PASS 7: SUMMARIZATION
    # ================================================================================
    logger.info("")
    logger.info("=" * 100)
    logger.info("[PASS 7] SUMMARIZATION - Generating LLM summaries for papers")
    logger.info("=" * 100)

    category_summaries: dict[str, list[dict[str, Any]]] = {}

    def summarize_paper_task(paper_id, data):
        """Summarize a single paper."""
        # Only summarize papers that weren't quarantined
        if data["topic_relevance"] < min_topic_relevance:
            return None

        metadata = data["metadata"]
        sections = pdf_parser.extract_sections(data["text"])

        summary = summarizer.summarize_paper(
            title=metadata.get("title", ""),
            abstract=sections.get("abstract"),
            intro=sections.get("introduction"),
            topic=topic,
            metadata=metadata,
            full_text=data.get("text"),
        )

        return paper_id, data, summary

    # Parallelize summarization
    papers_to_summarize = [
        (pid, data) for pid, data in paper_data.items()
        if data["topic_relevance"] >= min_topic_relevance and not data.get("canonical_id")
    ]
    
    logger.info(f"Generating summaries for {len(papers_to_summarize)} papers...")

    with ThreadPoolExecutor(max_workers=2) as executor:
        results = list(
            tqdm(
                executor.map(lambda item: summarize_paper_task(item[0], item[1]), papers_to_summarize),
                total=len(papers_to_summarize),
                desc="Generating summaries",
                unit="paper",
                ncols=80,
                bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}]",
                position=0,
                dynamic_ncols=False,
            )
        )

    # Group by category
    for result in results:
        if result is None:
            continue

        paper_id, data, summary = result
        data["summary"] = summary

        category = data["best_category"]
        if category not in category_summaries:
            category_summaries[category] = []

        metadata = data["metadata"]
        category_summaries[category].append(
            {
                "title": metadata.get("title", ""),
                "authors": metadata.get("authors", []),
                "year": metadata.get("year"),
                "venue": metadata.get("venue"),
                "relevance_score": data["category_scores"].get(category, 0),
                "summary": summary,
                "bibtex": metadata.get("bibtex", ""),
            }
        )

    # Write category summaries
    logger.info("")
    logger.info("Writing category summary markdown files...")
    for category, summaries in category_summaries.items():
        output_generator.write_category_summary(category, summaries)
        logger.info(f"  • {category}.md ({len(summaries)} papers)")

    # ================================================================================
    # PASS 8: GENERATE INDEX
    # ================================================================================
    logger.info("")
    logger.info("=" * 100)
    logger.info("[PASS 8] INDEX GENERATION - Writing final index files")
    logger.info("=" * 100)

    # Build output records
    records = []
    for paper_id, data in paper_data.items():
        metadata = data["metadata"]
        doc = data["doc"]
        topic_rel = data["topic_relevance"]
        best_category = data["best_category"]
        cat_scores = data["category_scores"]
        reasoning = data["reasoning"]
        
        # Determine final category
        if topic_rel < min_topic_relevance:
            final_category = "quarantined"
        elif data.get("canonical_id"):
            final_category = "repeated"
        else:
            final_category = best_category

        record = {
            "paper_id": paper_id,
            "title": metadata.get("title", ""),
            "authors": metadata.get("authors", []),
            "year": metadata.get("year"),
            "venue": metadata.get("venue"),
            "doi": metadata.get("doi"),
            "category": final_category,
            "topic_relevance": topic_rel,
            "category_scores": cat_scores,
            "reasoning": reasoning,
            "duplicate_of": data.get("canonical_id"),
            "is_duplicate": data.get("canonical_id") is not None,
            "path": str(doc.file_path),
            "bibtex": metadata.get("bibtex", ""),
            "summary_file": f"summaries/{final_category}.md" if topic_rel >= min_topic_relevance else None,
            "analyzed": True,
        }
        records.append(record)

    # Write outputs
    output_generator.write_jsonl(records)
    output_generator.write_csv(records)
    logger.info(f"Wrote {len(records)} records to index.jsonl and index.csv")

    # ================================================================================
    # FINAL SUMMARY
    # ================================================================================
    logger.info("")
    logger.info("=" * 100)
    logger.info("PIPELINE COMPLETE")
    logger.info("=" * 100)
    logger.info(f"Total papers processed: {len(paper_data)}")
    logger.info(f"Papers by category:")

    category_counts: dict[str, int] = {}
    for data in paper_data.values():
        topic_rel = data["topic_relevance"]
        best_cat = data["best_category"]
        
        if topic_rel < min_topic_relevance:
            cat = "quarantined"
        elif data.get("canonical_id"):
            cat = "repeated"
        else:
            cat = best_cat
            
        category_counts[cat] = category_counts.get(cat, 0) + 1

    for cat, count in sorted(category_counts.items()):
        logger.info(f"  • {cat}: {count} papers")
    
    logger.info(f"Unreadable papers: {len(unreadable_papers)}")
    logger.info(f"Output directory: {output_dir}")
    logger.info(f"Categories file: {output_dir / 'categories.json'}")
    logger.info("=" * 100)

    # Print minimal summary to terminal
    print("\n" + "=" * 80)
    print("✓ Pipeline Complete!")
    print(f"  Topic: {topic}")
    print(f"  Categories: {len(categories)}")
    print(f"  Processed: {len(paper_data)} papers")
    print(f"  Outputs: {output_dir}")
    print(f"  Log file: {log_file}")
    print("=" * 80)


if __name__ == "__main__":
    cli()
