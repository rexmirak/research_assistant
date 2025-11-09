# Research Assistant

An intelligent, offline-first pipeline for processing hundreds of research papers using local LLMs (Ollama), accurate PDF parsing, metadata extraction, relevance scoring, deduplication, and automated summarization.

## Features

- **Generic & Configurable**: Runtime topic and directory configuration (no hardcoding)
- **Accurate PDF Parsing**: PyMuPDF + OCR fallback (ocrmypdf + Tesseract) + pdfminer.six
- **Structured Metadata**: GROBID integration for titles, authors, venues, years, DOIs, and BibTeX
- **Smart Deduplication**: Exact (hash-based) and near-duplicate (similarity-based) detection
- **Relevance Scoring**: Ollama embeddings + cosine similarity → 0-10 score + inclusion boolean
- **Category Validation**: LLM-based recategorization with confidence tracking
- **Topic-Focused Summaries**: Per-paper summaries with "how this helps your research"
- **Move Tracking**: Manifest per category prevents duplicate analysis after recategorization
- **Resumable**: SQLite cache for embeddings, GROBID results, OCR outputs
- **Multiple Outputs**: JSONL master index + CSV spreadsheet + Markdown summaries per category

## Architecture

```
research_assistant/
├── cli.py                  # Main CLI entry point
├── config.py               # Configuration and settings
├── core/
│   ├── inventory.py        # Directory traversal and PDF discovery
│   ├── parser.py           # PDF text extraction (PyMuPDF + OCR)
│   ├── metadata.py         # GROBID integration + Crossref enrichment
│   ├── dedup.py            # Exact and near-duplicate detection
│   ├── embeddings.py       # Ollama embedding generation
│   ├── scoring.py          # Relevance scoring and ranking
│   ├── classifier.py       # Category validation and recategorization
│   ├── summarizer.py       # Topic-focused summary generation
│   ├── mover.py            # File moving with manifest tracking
│   └── outputs.py          # JSONL, CSV, and Markdown generation
├── cache/
│   └── cache_manager.py    # SQLite-based caching
├── utils/
│   ├── hash.py             # Content hashing utilities
│   ├── text.py             # Text normalization and processing
│   └── grobid_client.py    # GROBID Docker client
└── tests/                  # Unit and integration tests
```

## Prerequisites

- **Python 3.10+**
- **Docker** (for GROBID server)
- **Ollama** with models:
  - `deepseek-r1:8b` (summarization & classification - superior reasoning for academic content)
  - `nomic-embed-text` (embeddings)
- **Tesseract** (for OCR): `brew install tesseract`

## Installation

```bash
# Clone or navigate to project
cd research_assistant

# Create virtual environment
python3 -m venv venv
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt

# Pull Ollama models
ollama pull deepseek-r1:8b
ollama pull nomic-embed-text

# Start GROBID (Docker)
docker run -d -p 8070:8070 lfoppiano/grobid:0.8.0
```

## Quick Start

```bash
# Run the full pipeline
python cli.py process \
  --root-dir /path/to/papers \
  --topic "Your research topic description here" \
  --output-dir ./outputs \
  --cache-dir ./cache

# Dry-run (no file moves)
python cli.py process \
  --root-dir /path/to/papers \
  --topic "Your research topic" \
  --dry-run

# Resume a previous run
python cli.py process \
  --root-dir /path/to/papers \
  --topic "Your research topic" \
  --resume
```

## Configuration

Runtime configuration via CLI flags or `config.yaml`:

```yaml
# config.yaml (optional)
relevance_threshold: 6.5  # Include papers with score >= 6.5
dedup_similarity: 0.95    # Near-duplicate threshold
ollama:
  summarize_model: "deepseek-r1:8b"
  classify_model: "deepseek-r1:8b"
  embed_model: "nomic-embed-text"
  temperature: 0.2
grobid:
  url: "http://localhost:8070"
  timeout: 60
crossref:
  enabled: true
  email: "your.email@domain.com"  # Polite pool
move:
  enabled: true
  track_manifest: true
```

## Move Tracking & Manifest System

**Problem**: If a paper is moved from `CategoryA/` to `CategoryB/` during analysis, we must ensure it's not analyzed twice.

**Solution**: Each category maintains a `.manifest.json`:
- Tracks all papers ever analyzed in this category (by content hash)
- Records move history (from/to, timestamp, reason)
- On processing, skip papers already in manifest with status "moved_out"
- When moving a paper in, add to destination manifest with status "moved_in" and link to original

**Manifest Entry Example**:
```json
{
  "paper_id": "abc123def456...",
  "original_path": "CategoryA/smith2023.pdf",
  "current_path": "CategoryB/smith2023.pdf",
  "status": "moved_in",
  "moved_from": "CategoryA",
  "moved_at": "2025-11-09T14:23:00Z",
  "reason": "Better fit for CategoryB based on content",
  "analyzed": true
}
```

## Output Structure

```
outputs/
├── index.jsonl              # Full machine-readable index
├── index.csv                # Spreadsheet with all metadata
├── summaries/
│   ├── CategoryA.md         # Summaries for all papers in CategoryA
│   ├── CategoryB.md
│   └── ...
├── logs/
│   ├── pipeline.log         # Detailed execution log
│   └── moves.log            # All file moves with reasons
└── manifests/
    ├── CategoryA.manifest.json
    ├── CategoryB.manifest.json
    └── ...
```

## CSV Columns

- `paper_id`: Unique identifier (content hash)
- `title`, `authors`, `year`, `venue`, `doi`
- `bibtex`: Complete BibTeX citation
- `category`: Current category
- `original_category`: Initial category from folder
- `relevance_score`: 0-10 relevance to topic
- `include`: Boolean for inclusion in research
- `duplicate_of`: Paper ID if this is a duplicate
- `is_duplicate`: Boolean flag
- `status`: `active`, `duplicate`, `quarantined`, `moved`
- `original_path`, `current_path`
- `summary_file`: Link to markdown summary
- `notes`: Additional information

## Advanced Usage

### Resume from specific stage
```bash
python cli.py process --root-dir ./papers --topic "..." --start-from embeddings
```

### Custom thresholds
```bash
python cli.py process \
  --root-dir ./papers \
  --topic "..." \
  --relevance-threshold 7.0 \
  --dedup-threshold 0.98
```

### Export only (skip analysis)
```bash
python cli.py export --cache-dir ./cache --output-dir ./outputs
```

## Troubleshooting

### GROBID not responding
```bash
# Check GROBID is running
curl http://localhost:8070/api/isalive

# Restart if needed
docker restart $(docker ps -q --filter ancestor=lfoppiano/grobid:0.8.0)
```

### OCR failing
```bash
# Verify Tesseract installation
tesseract --version

# Install additional language packs if needed
brew install tesseract-lang
```

### Ollama connection issues
```bash
# Check Ollama is running
ollama list

# Restart Ollama service
brew services restart ollama
```

## Performance Tips

- **Parallel processing**: Set `--workers 4` for multiprocessing
- **Cache warming**: Run inventory + parsing first, then scoring/summarization
- **Selective OCR**: Skip OCR for born-digital PDFs (auto-detected)
- **Batch embeddings**: Automatically batched in groups of 64

## License

MIT
