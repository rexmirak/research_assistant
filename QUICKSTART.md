# 🎯 Research Assistant - Complete Implementation

## ✅ What's Been Built

A **production-ready, fully-functional research paper analysis pipeline** that processes hundreds of PDFs with:

1. **Accurate PDF parsing** (PyMuPDF + OCR + fallbacks)
2. **Structured metadata extraction** (GROBID + Crossref + BibTeX)
3. **Smart deduplication** (exact + near-duplicate detection)
4. **AI-powered relevance scoring** (Ollama embeddings, 0-10 scale)
5. **Category validation & recategorization** (LLM-based)
6. **Topic-focused summaries** (per paper + aggregated by category)
7. **Move tracking system** ⭐ (prevents duplicate analysis)
8. **Multiple output formats** (JSONL + CSV + Markdown)

## 🔑 Critical Features

### Move Tracking System (Your Key Requirement)
**Problem**: Papers moved between categories during analysis could be processed twice.

**Solution**: Manifest system per category:
- Tracks every analyzed paper by content hash
- Records move history (from → to, reason, timestamp)
- Skips papers with status "moved_out" in source category
- Links moved-in papers to original location
- **Prevents duplicate entries in final output** ✅

### Runtime Configuration (Generic Design)
- **No hardcoded topics**: Topic provided via CLI `--topic "..."`
- **No hardcoded paths**: Root directory via CLI `--root-dir /path`
- **Everything configurable**: Models, thresholds, workers via CLI or YAML

### Smart File Organization
- Duplicates → `repeated/`
- Unrelated (low score) → `quarantined/`
- Recategorized papers physically moved with tracking
- Original locations preserved in manifests

## 📁 Project Structure

```
research_assistant/
├── README.md                 ← Main documentation
├── USAGE.md                  ← Detailed usage guide
├── TROUBLESHOOTING.md        ← Common issues & fixes
├── PROJECT_SUMMARY.md        ← Technical overview
├── requirements.txt          ← Python dependencies
├── config.example.yaml       ← Example configuration
├── setup.sh                  ← Automated setup
├── Makefile                  ← Convenience commands
├── cli.py                    ← Main entry point ⭐
├── config.py                 ← Configuration system
├── example.py                ← Example usage script
├── check_install.py          ← Installation checker
│
├── core/                     ← Core processing modules
│   ├── inventory.py          - PDF discovery & scanning
│   ├── parser.py             - Text extraction (OCR)
│   ├── metadata.py           - GROBID + Crossref + BibTeX
│   ├── dedup.py              - Duplicate detection
│   ├── embeddings.py         - Ollama embeddings
│   ├── scoring.py            - Relevance scoring
│   ├── classifier.py         - Category validation
│   ├── summarizer.py         - LLM summaries
│   ├── mover.py              - File moving with tracking
│   ├── outputs.py            - JSONL/CSV/Markdown generation
│   └── manifest.py           - Move tracking system ⭐⭐⭐
│
├── cache/
│   └── cache_manager.py      - SQLite caching for resume
│
├── utils/
│   ├── hash.py               - Content hashing
│   ├── text.py               - Text processing
│   └── grobid_client.py      - GROBID API client
│
└── tests/                    ← Test suite
    ├── conftest.py
    ├── test_scoring.py
    ├── test_dedup.py
    └── test_manifest.py      - Manifest system tests
```

## 🚀 Quick Start

### 1. Setup (One Time)
```bash
cd /Users/karim/Desktop/projects/research_assistant

# Run automated setup
./setup.sh

# Or manually:
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
ollama pull llama3.1:8b
ollama pull nomic-embed-text
docker run -d -p 8070:8070 lfoppiano/grobid:0.8.0
```

### 2. Verify Installation
```bash
./check_install.py

# Or check services manually:
make check-services
```

### 3. Prepare Your Papers
```
your_papers/
├── Machine_Learning/
│   ├── paper1.pdf
│   └── paper2.pdf
├── Computer_Vision/
│   └── paper3.pdf
└── NLP/
    └── paper4.pdf
```

### 4. Run Pipeline
```bash
# Activate environment
source venv/bin/activate

# Run with your topic
python cli.py process \
  --root-dir /path/to/your_papers \
  --topic "Your detailed research topic description here"

# Or with Makefile:
make run \
  ROOT_DIR=/path/to/your_papers \
  TOPIC="Your research topic"
```

## 📊 What You Get

### Outputs Directory Structure
```
outputs/
├── index.jsonl              ← Machine-readable full index
├── index.csv                ← Spreadsheet (open in Excel/Numbers)
├── statistics.json          ← Score distributions
├── summaries/
│   ├── Machine_Learning.md  ← Per-category summaries
│   ├── Computer_Vision.md
│   └── NLP.md
├── logs/
│   ├── pipeline_*.log       ← Execution logs
│   └── moves.log            ← File move history
└── manifests/
    ├── Machine_Learning.manifest.json
    ├── Computer_Vision.manifest.json
    └── repeated.manifest.json
```

### CSV Columns (index.csv)
- **paper_id**: Unique identifier
- **title**, **authors**, **year**, **venue**, **doi**
- **bibtex**: Full citation ready for LaTeX
- **category**: Current category
- **relevance_score**: 0-10 relevance to your topic
- **include**: Boolean for inclusion recommendation
- **status**: active | moved | duplicate | quarantined
- **duplicate_of**: Link to canonical paper if duplicate
- **current_path**, **original_path**: File locations
- **summary_file**: Link to markdown summary

### Markdown Summaries
Each category gets a summary file with:
- Table of contents
- Per-paper summaries including:
  - Title, authors, year, venue
  - Relevance score
  - **Key contributions**
  - **Methods used**
  - **How this paper helps your research** ⭐
  - **Specific points relevant to your topic** ⭐
  - BibTeX citation

## 🎨 Example Usage Patterns

### Pattern 1: First-Time Analysis
```bash
# Dry run to preview
python cli.py process \
  --root-dir ~/papers \
  --topic "Machine learning in healthcare" \
  --dry-run

# Review what would happen, then run for real
python cli.py process \
  --root-dir ~/papers \
  --topic "Machine learning in healthcare"
```

### Pattern 2: Adjusting Threshold
```bash
# More selective (higher threshold = fewer papers)
python cli.py process \
  --root-dir ~/papers \
  --topic "Your topic" \
  --relevance-threshold 7.5

# More inclusive (lower threshold = more papers)
python cli.py process \
  --root-dir ~/papers \
  --topic "Your topic" \
  --relevance-threshold 5.0
```

### Pattern 3: Adding New Papers
```bash
# Add new PDFs to category folders
# Run with --resume to skip already-processed
python cli.py process \
  --root-dir ~/papers \
  --topic "Same topic as before" \
  --resume
```

## ⚙️ Configuration

### Quick Config (CLI)
```bash
python cli.py process \
  --root-dir /path \
  --topic "..." \
  --relevance-threshold 6.5 \
  --workers 4 \
  --dry-run \
  --resume
```

### Advanced Config (YAML)
```bash
# Copy example
cp config.example.yaml my_config.yaml

# Edit with your preferences
# Then run:
python cli.py process \
  --root-dir /path \
  --topic "..." \
  --config-file my_config.yaml
```

### Key Settings to Adjust
- **relevance_threshold**: 6.5 (default) - papers >= this are included
- **dedup.similarity_threshold**: 0.95 (default) - lower = more sensitive
- **ollama.temperature**: 0.2 (default) - lower = more focused
- **processing.workers**: 4 (default) - adjust based on RAM

## 🧪 Testing

```bash
# Run all tests
make test

# With coverage
make test-coverage

# Test specific module
pytest tests/test_manifest.py -v
```

## 🔧 Troubleshooting

### GROBID Not Running
```bash
# Check
curl http://localhost:8070/api/isalive

# Start
make grobid-start

# Restart
make grobid-restart
```

### Ollama Issues
```bash
# Check models
ollama list

# Pull missing models
ollama pull llama3.1:8b
ollama pull nomic-embed-text
```

### Pipeline Slow
- Reduce workers: `--workers 2`
- Skip OCR if not needed (config)
- Process categories separately

### See Full Guide
`TROUBLESHOOTING.md` has solutions for common issues

## 📚 Documentation

- **README.md**: Overview and quick start
- **USAGE.md**: Detailed usage guide with examples
- **TROUBLESHOOTING.md**: Common issues and solutions
- **PROJECT_SUMMARY.md**: Technical architecture and decisions

## 🎯 What Makes This Special

### 1. Move Tracking (Your Key Requirement) ⭐⭐⭐
The manifest system ensures papers moved during analysis are:
- Never analyzed twice
- Tracked with full history
- Linked to original locations
- Excluded from duplicate entries in outputs

### 2. Fully Generic (No Hardcoding)
- Topic: runtime CLI argument
- Root directory: runtime CLI argument
- Everything else: configurable via CLI or YAML

### 3. Offline-First
- Local LLMs (Ollama)
- Local GROBID (Docker)
- Optional internet for Crossref enrichment
- All data stays on your machine

### 4. Resumable & Cached
- SQLite cache for expensive operations
- Resume from any point
- Avoid reprocessing

### 5. Explainable
- Every move has a reason
- Logs track all decisions
- Manifests provide audit trail

## 🚦 Status: READY TO USE

✅ All core functionality implemented
✅ Move tracking system working
✅ Tests written and passing
✅ Documentation complete
✅ Setup automation ready
✅ Example scripts provided

## 📞 Next Steps

1. **Run setup**: `./setup.sh`
2. **Verify install**: `./check_install.py`
3. **Prepare papers**: Organize in category folders
4. **Run pipeline**: `python cli.py process --root-dir ... --topic "..."`
5. **Review outputs**: Check `outputs/index.csv` and summaries
6. **Iterate**: Adjust threshold or topic, re-run with `--resume`

## 💡 Tips

- Start with `--dry-run` to preview
- Use specific, detailed topic descriptions (100-500 words)
- Review `outputs/statistics.json` to calibrate threshold
- Check `repeated/` and `quarantined/` folders periodically
- Use `--resume` when adding new papers
- Keep manifests for audit trail

---

**You now have a complete, production-ready research assistant that:**
- Processes hundreds of papers automatically
- Never duplicates analysis after moves ✅
- Provides topic-focused summaries
- Outputs structured data (CSV, JSONL, Markdown)
- Is fully configurable at runtime
- Works offline with local LLMs

**Happy researching! 🎓📚**
