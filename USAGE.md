# Usage Guide

## Quick Start

### 1. Setup
```bash
# Run setup script
./setup.sh

# Or manually:
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
ollama pull llama3.1:8b
ollama pull nomic-embed-text
docker run -d -p 8070:8070 lfoppiano/grobid:0.8.0
```

### 2. Prepare Your Papers
Organize PDFs in category folders:
```
my_papers/
├── Machine_Learning/
│   ├── paper1.pdf
│   └── paper2.pdf
├── Computer_Vision/
│   ├── paper3.pdf
│   └── paper4.pdf
└── NLP/
    └── paper5.pdf
```

### 3. Run Pipeline
```bash
python cli.py process \
  --root-dir /path/to/my_papers \
  --topic "Your research topic description here"
```

## Detailed Usage

### Command-Line Options

```bash
python cli.py process [OPTIONS]
```

**Required Options:**
- `--root-dir PATH`: Root directory containing category folders with PDFs
- `--topic TEXT`: Research topic description (be specific!)

**Optional Options:**
- `--output-dir PATH`: Output directory (default: ./outputs)
- `--cache-dir PATH`: Cache directory (default: ./cache)
- `--config-file PATH`: YAML configuration file
- `--dry-run`: Preview moves without executing
- `--resume`: Resume from cached data
- `--relevance-threshold FLOAT`: Score threshold for inclusion (default: 6.5)
- `--workers INT`: Number of parallel workers (default: 4)

### Examples

#### Basic Usage
```bash
python cli.py process \
  --root-dir ~/Documents/research_papers \
  --topic "Machine learning applications in medical imaging"
```

#### With Custom Configuration
```bash
# Copy example config
cp config.example.yaml config.yaml

# Edit config.yaml with your preferences
# Then run:
python cli.py process \
  --root-dir ~/Documents/research_papers \
  --topic "Deep learning for drug discovery" \
  --config-file config.yaml
```

#### Dry Run (Preview)
```bash
# See what would happen without moving files
python cli.py process \
  --root-dir ~/Documents/research_papers \
  --topic "Reinforcement learning in robotics" \
  --dry-run
```

#### Resume from Cache
```bash
# If pipeline was interrupted, resume:
python cli.py process \
  --root-dir ~/Documents/research_papers \
  --topic "Same topic as before" \
  --resume
```

#### Adjust Relevance Threshold
```bash
# More selective (higher threshold)
python cli.py process \
  --root-dir ~/Documents/research_papers \
  --topic "Quantum computing" \
  --relevance-threshold 7.5

# More inclusive (lower threshold)
python cli.py process \
  --root-dir ~/Documents/research_papers \
  --topic "Blockchain technology" \
  --relevance-threshold 5.0
```

### Using Makefile

#### Check Services
```bash
make check-services
```

#### Run with Makefile
```bash
make run \
  ROOT_DIR=/path/to/papers \
  TOPIC="Your research topic"
```

#### Dry Run
```bash
make dry-run \
  ROOT_DIR=/path/to/papers \
  TOPIC="Your research topic"
```

## Writing Effective Topic Descriptions

### Good Topic Description
```
I am researching the application of transformer architectures in natural 
language processing, with a focus on:
1. Pre-training methods and efficiency improvements
2. Fine-tuning strategies for domain-specific tasks
3. Multilingual and cross-lingual transfer learning
4. Interpretability and attention mechanism analysis

I'm particularly interested in recent advances (2020-2025) and practical 
deployment considerations for production systems.
```

### Less Effective Topic Description
```
Machine learning
```

**Why?** Too broad. The system can't distinguish relevant from irrelevant papers.

### Tips for Topic Descriptions
- **Be specific**: Mention key techniques, domains, or problems
- **Include scope**: Time period, specific aspects, exclusions
- **Add context**: Your research goals, what you're looking for
- **Length**: 100-500 words is ideal
- **Keywords**: Include important terms that would appear in relevant papers

## Understanding Outputs

### Directory Structure
```
outputs/
├── index.jsonl          # Machine-readable full index
├── index.csv            # Spreadsheet with all metadata
├── statistics.json      # Summary statistics
├── summaries/
│   ├── Machine_Learning.md
│   ├── Computer_Vision.md
│   └── ...
├── logs/
│   ├── pipeline_20251109_143000.log
│   └── moves.log
└── manifests/
    ├── Machine_Learning.manifest.json
    ├── Computer_Vision.manifest.json
    ├── repeated.manifest.json
    └── quarantined.manifest.json
```

### CSV Columns Explained

- **paper_id**: Unique identifier (content hash)
- **title**: Paper title
- **authors**: Semicolon-separated author list
- **year**: Publication year
- **venue**: Journal or conference name
- **doi**: Digital Object Identifier
- **category**: Current category
- **original_category**: Initial category from folder
- **relevance_score**: 0-10 relevance to your topic
- **include**: Boolean - recommended for inclusion
- **status**: active | moved | duplicate | quarantined
- **duplicate_of**: Paper ID if duplicate
- **is_duplicate**: Boolean flag
- **original_path**: Original file location
- **current_path**: Current file location (after moves)
- **bibtex**: BibTeX citation
- **summary_file**: Path to markdown summary
- **notes**: Additional information

### Reading Markdown Summaries

Each category has a summary file with:
- Table of contents
- Per-paper summaries including:
  - Title, authors, year, venue
  - Relevance score
  - Key contributions
  - Methods used
  - Specific relevance to your topic
  - How it helps your research
  - BibTeX citation

## Advanced Workflows

### Processing Large Collections

For 500+ papers, process incrementally:

```bash
# Process by category
for category in Machine_Learning Computer_Vision NLP; do
  python cli.py process \
    --root-dir ~/papers/$category \
    --topic "Your topic" \
    --output-dir ~/outputs/$category
done

# Merge outputs
cat ~/outputs/*/index.jsonl > combined.jsonl
```

### Updating After Adding New Papers

```bash
# Add new PDFs to categories
# Run with --resume to skip already-processed papers
python cli.py process \
  --root-dir ~/Documents/research_papers \
  --topic "Same topic" \
  --resume
```

### Adjusting After Initial Run

1. **Review outputs**: Check `outputs/index.csv`
2. **Adjust threshold**: If too many/few papers included
3. **Refine topic**: Make description more specific
4. **Clear cache**: `make clean`
5. **Re-run**: `python cli.py process ...`

### Exporting for Analysis

```bash
# Import in Python
import pandas as pd
df = pd.read_csv('outputs/index.csv')

# Filter included papers
included = df[df['include'] == True]

# Group by category
by_category = df.groupby('category').size()

# High-relevance papers
top_papers = df[df['relevance_score'] >= 8.0]
```

### Custom Post-Processing

```python
# Load JSONL
import json

papers = []
with open('outputs/index.jsonl', 'r') as f:
    for line in f:
        papers.append(json.loads(line))

# Filter by custom criteria
recent_papers = [p for p in papers if int(p.get('year', 0)) >= 2023]

# Export BibTeX for LaTeX
with open('references.bib', 'w') as f:
    for paper in papers:
        if paper['include']:
            f.write(paper['bibtex'] + '\n\n')
```

## Configuration Options

### Creating Custom Config

```bash
cp config.example.yaml my_config.yaml
# Edit my_config.yaml
python cli.py process --config-file my_config.yaml ...
```

### Key Configuration Sections

#### Ollama Models
```yaml
ollama:
  summarize_model: "llama3.1:8b"     # For summaries & classification
  embed_model: "nomic-embed-text"    # For embeddings
  temperature: 0.2                   # Lower = more focused
```

Alternative models:
- Summarization: `qwen2:7b-instruct`, `mistral:7b-instruct`
- Embeddings: `snowflake-arctic-embed` (if available)

#### Scoring Thresholds
```yaml
scoring:
  relevance_threshold: 6.5  # Include papers >= this score
  min_score: 0.0
  max_score: 10.0
```

#### Deduplication
```yaml
dedup:
  similarity_threshold: 0.95  # 0.90 = more sensitive
  num_perm: 128               # MinHash permutations
```

#### Processing
```yaml
processing:
  workers: 4           # Parallel workers
  batch_size: 32       # Embedding batch size
  ocr_language: "eng"  # Tesseract language
```

## Tips & Best Practices

### Performance
- Start with `--dry-run` to preview
- Use `--workers 2` on systems with limited RAM
- Enable `skip_ocr_if_text_exists` in config
- Pre-download PDFs if from cloud storage

### Accuracy
- Review quarantined papers - some may be false negatives
- Check `repeated/` - ensure actual duplicates
- Manually verify high-scoring papers align with topic
- Adjust threshold based on score distribution

### Organization
- Use descriptive category folder names
- Keep folder structure flat (avoid deep nesting)
- Name PDFs meaningfully (not `paper1.pdf`)
- Clean up before processing (remove non-PDFs)

### Resumability
- Cache is your friend - enable it
- Use `--resume` when re-running
- Keep cache/ and outputs/ separate
- Back up manifests before major changes

## Common Patterns

### Pattern 1: Initial Survey
```bash
# Broad topic, low threshold, review outputs
python cli.py process \
  --root-dir papers/ \
  --topic "Broad topic" \
  --relevance-threshold 5.0 \
  --dry-run

# Review, refine topic, run for real
```

### Pattern 2: Targeted Review
```bash
# Specific topic, high threshold, include only best
python cli.py process \
  --root-dir papers/ \
  --topic "Very specific aspect" \
  --relevance-threshold 7.5
```

### Pattern 3: Incremental Addition
```bash
# Initial run
python cli.py process --root-dir papers/ --topic "X"

# Add new papers to folders
# Re-run with resume
python cli.py process --root-dir papers/ --topic "X" --resume
```

## Next Steps

1. **Review outputs**: Start with `outputs/index.csv`
2. **Read summaries**: Check `outputs/summaries/`
3. **Validate moves**: Look at moved papers in manifests
4. **Adjust as needed**: Refine topic or threshold
5. **Export citations**: Use BibTeX from CSV
6. **Integrate**: Import into reference manager

For issues, see [TROUBLESHOOTING.md](TROUBLESHOOTING.md)
