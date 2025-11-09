"""Main CLI for research assistant pipeline."""

import click
import logging
import sys
from pathlib import Path
from datetime import datetime
from config import Config
from core.inventory import InventoryManager
from core.parser import PDFParser
from core.metadata import MetadataExtractor
from core.dedup import DedupManager
from core.embeddings import EmbeddingGenerator
from core.scoring import ScoringEngine
from core.classifier import CategoryClassifier
from core.summarizer import Summarizer
from core.mover import FileMover
from core.outputs import OutputGenerator
from core.manifest import ManifestManager
from cache.cache_manager import CacheManager
from tqdm import tqdm

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
    "--config-file",
    type=click.Path(exists=True, path_type=Path),
    help="Configuration YAML file (optional)",
)
@click.option("--dry-run", is_flag=True, help="Run without moving files")
@click.option("--resume", is_flag=True, help="Resume from cache")
@click.option(
    "--relevance-threshold", type=float, default=6.5, help="Relevance score threshold for inclusion"
)
@click.option("--workers", type=int, default=4, help="Number of worker processes")
def process(
    root_dir,
    topic,
    output_dir,
    cache_dir,
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
    logger.info("Research Assistant Pipeline Starting")
    logger.info(f"Root Directory: {root_dir}")
    logger.info(f"Topic: {topic}")
    logger.info(f"Output Directory: {output_dir}")
    logger.info(f"Dry Run: {dry_run}")
    logger.info("=" * 80)

    try:
        # Initialize components
        logger.info("Initializing components...")
        cache_manager = CacheManager(config.cache_dir, config.cache.ttl_days)
        manifest_manager = ManifestManager(config.output_dir / "manifests")
        inventory_manager = InventoryManager(config.root_dir)
        pdf_parser = PDFParser(
            config.processing.ocr_language, config.processing.skip_ocr_if_text_exists
        )
        metadata_extractor = MetadataExtractor(
            config.grobid.url, config.crossref.enabled, config.crossref.email
        )
        dedup_manager = DedupManager(config.dedup.similarity_threshold, config.dedup.num_perm)
        embedding_generator = EmbeddingGenerator(
            config.ollama.embed_model, config.ollama.base_url, config.processing.batch_size
        )

        # Test connections
        logger.info("Testing service connections...")
        if not embedding_generator.test_connection():
            logger.error("Ollama connection failed. Please ensure Ollama is running.")
            sys.exit(1)

        # Initialize scoring engine
        scoring_engine = ScoringEngine(
            config.topic,
            embedding_generator,
            config.scoring.min_score,
            config.scoring.max_score,
            config.scoring.relevance_threshold,
        )

        classifier = CategoryClassifier(config.ollama.classify_model, config.ollama.temperature)

        summarizer = Summarizer(config.ollama.summarize_model, config.ollama.temperature)

        file_mover = FileMover(
            config.root_dir, manifest_manager, config.dry_run, config.move.create_symlinks
        )

        output_generator = OutputGenerator(config.output_dir)

        # Stage 1: Inventory
        logger.info("\n" + "=" * 80)
        logger.info("STAGE 1: Inventory")
        logger.info("=" * 80)
        documents = inventory_manager.scan()
        summary = inventory_manager.summary()
        logger.info(f"Total PDFs: {summary['total_documents']}")
        logger.info(f"Categories: {summary['total_categories']}")

        # Stage 2: Parse and extract metadata
        logger.info("\n" + "=" * 80)
        logger.info("STAGE 2: Parsing and Metadata Extraction")
        logger.info("=" * 80)

        paper_data = {}

        for doc in tqdm(documents, desc="Processing PDFs"):
            paper_id = doc.file_hash[:12]

            # Check manifest to avoid reprocessing moved papers
            manifest = manifest_manager.get_manifest(doc.category)
            if manifest.should_skip(paper_id):
                logger.debug(f"Skipping {doc.file_name} (moved or duplicate)")
                continue

            # Check cache
            cached_text = cache_manager.get_text(paper_id) if resume else None
            cached_metadata = cache_manager.get_metadata(paper_id) if resume else None

            if cached_text and cached_metadata:
                text, text_hash = cached_text
                metadata = cached_metadata
                logger.debug(f"Using cached data for {doc.file_name}")
            else:
                # Parse PDF
                text, text_hash = pdf_parser.extract_text(doc.file_path, config.cache_dir)
                sections = pdf_parser.extract_sections(text)

                # Extract metadata
                metadata = metadata_extractor.extract(doc.file_path)

                # Use GROBID abstract if available, otherwise extract from text
                if not metadata.get("abstract"):
                    metadata["abstract"] = sections.get("abstract")

                # Cache results
                cache_manager.set_text(paper_id, text, text_hash)
                cache_manager.set_metadata(paper_id, metadata)

            # Store paper data
            paper_data[paper_id] = {
                "doc": doc,
                "text": text,
                "text_hash": text_hash,
                "metadata": metadata,
                "sections": pdf_parser.extract_sections(text),
            }

            # Add to manifest
            manifest.add_paper(
                paper_id=paper_id,
                path=str(doc.file_path),
                content_hash=text_hash,
                original_category=doc.category,
            )

        manifest_manager.save_all()

        # Stage 3: Deduplication
        logger.info("\n" + "=" * 80)
        logger.info("STAGE 3: Deduplication")
        logger.info("=" * 80)

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
                    file_mover.move_to_repeated(
                        paper_id=dup_id,
                        current_path=dup_data["doc"].file_path,
                        from_category=dup_data["doc"].category,
                        canonical_id=canonical,
                    )
                    # Mark in manifest
                    manifest = manifest_manager.get_manifest(dup_data["doc"].category)
                    manifest.mark_duplicate(dup_id, canonical)

        manifest_manager.save_all()

        # Stage 4: Embeddings and Scoring
        logger.info("\n" + "=" * 80)
        logger.info("STAGE 4: Relevance Scoring")
        logger.info("=" * 80)

        paper_embeddings = {}
        for paper_id, data in tqdm(list(paper_data.items()), desc="Generating embeddings"):
            # Check cache
            cache_key = f"{paper_id}_embed"
            cached_embedding = cache_manager.get_embedding(cache_key) if resume else None

            if cached_embedding:
                paper_embeddings[paper_id] = cached_embedding
            else:
                # Generate embedding
                metadata = data["metadata"]
                sections = data["sections"]

                embedding = embedding_generator.embed_paper(
                    metadata.get("title", ""),
                    sections.get("abstract"),
                    sections.get("introduction"),
                )

                if embedding:
                    paper_embeddings[paper_id] = embedding
                    cache_manager.set_embedding(cache_key, embedding)

        # Score papers
        scores = scoring_engine.score_papers_batch(paper_embeddings)

        # Store scores
        for paper_id, (score, include) in scores.items():
            paper_data[paper_id]["relevance_score"] = score
            paper_data[paper_id]["include"] = include

        # Log statistics
        stats = scoring_engine.get_statistics(scores)
        logger.info(f"Scoring Statistics:")
        logger.info(f"  Total: {stats['total_papers']}")
        logger.info(f"  Included: {stats['included']}")
        logger.info(f"  Excluded: {stats['excluded']}")
        logger.info(f"  Mean Score: {stats['mean_score']:.2f}")

        # Stage 5: Category Validation
        logger.info("\n" + "=" * 80)
        logger.info("STAGE 5: Category Validation")
        logger.info("=" * 80)

        categories = inventory_manager.get_categories()

        for paper_id, data in tqdm(list(paper_data.items()), desc="Validating categories"):
            metadata = data["metadata"]
            current_category = data["doc"].category

            recommended_cat, confidence, reason = classifier.classify_paper(
                title=metadata.get("title", ""),
                abstract=data["sections"].get("abstract"),
                current_category=current_category,
                available_categories=categories,
                topic=config.topic,
            )

            # Move if needed
            if classifier.should_recategorize(current_category, recommended_cat, confidence):
                logger.info(
                    f"Recategorizing {data['doc'].file_name}: {current_category} -> {recommended_cat} ({reason})"
                )
                new_path = file_mover.move_to_category(
                    paper_id=paper_id,
                    current_path=data["doc"].file_path,
                    from_category=current_category,
                    to_category=recommended_cat,
                    reason=reason,
                )
                if new_path:
                    data["doc"].category = recommended_cat

        # Stage 6: Quarantine low-relevance papers
        logger.info("\n" + "=" * 80)
        logger.info("STAGE 6: Quarantine Unrelated Papers")
        logger.info("=" * 80)

        quarantine_threshold = 3.0  # Papers below this are quarantined
        for paper_id, data in list(paper_data.items()):
            if data.get("relevance_score", 0) < quarantine_threshold:
                logger.info(
                    f"Quarantining {data['doc'].file_name} (score: {data['relevance_score']:.1f})"
                )
                file_mover.move_to_quarantined(
                    paper_id=paper_id,
                    current_path=data["doc"].file_path,
                    from_category=data["doc"].category,
                    reason=f"Low relevance score: {data['relevance_score']:.1f}",
                )

        # Stage 7: Summarization
        logger.info("\n" + "=" * 80)
        logger.info("STAGE 7: Generating Summaries")
        logger.info("=" * 80)

        category_summaries = {}

        for paper_id, data in tqdm(list(paper_data.items()), desc="Generating summaries"):
            # Only summarize included papers
            if not data.get("include", False):
                continue

            metadata = data["metadata"]
            sections = data["sections"]

            summary = summarizer.summarize_paper(
                title=metadata.get("title", ""),
                abstract=sections.get("abstract"),
                intro=sections.get("introduction"),
                topic=config.topic,
                metadata=metadata,
            )

            data["summary"] = summary

            # Group by category
            category = data["doc"].category
            if category not in category_summaries:
                category_summaries[category] = []

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

        # Stage 8: Generate Outputs
        logger.info("\n" + "=" * 80)
        logger.info("STAGE 8: Generating Outputs")
        logger.info("=" * 80)

        # Build output records
        records = []
        for paper_id, data in paper_data.items():
            metadata = data["metadata"]

            record = {
                "paper_id": paper_id,
                "title": metadata.get("title", ""),
                "authors": metadata.get("authors", []),
                "year": metadata.get("year"),
                "venue": metadata.get("venue"),
                "doi": metadata.get("doi"),
                "category": data["doc"].category,
                "original_category": data["doc"].category,
                "relevance_score": data.get("relevance_score"),
                "include": data.get("include", False),
                "status": "active",
                "duplicate_of": None,
                "is_duplicate": False,
                "original_path": str(data["doc"].file_path),
                "current_path": str(data["doc"].file_path),
                "bibtex": metadata.get("bibtex", ""),
                "summary_file": f"summaries/{data['doc'].category}.md",
                "notes": "",
            }

            records.append(record)

        # Write outputs
        output_generator.write_jsonl(records)
        output_generator.write_csv(records)
        output_generator.write_statistics(stats, "statistics.json")

        logger.info("\n" + "=" * 80)
        logger.info("Pipeline Complete!")
        logger.info(f"Processed: {len(paper_data)} papers")
        logger.info(f"Outputs written to: {config.output_dir}")
        logger.info("=" * 80)

    except Exception as e:
        logger.error(f"Pipeline failed: {e}", exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    cli()
