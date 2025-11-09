# Troubleshooting Guide

## Common Issues and Solutions

### Installation Issues

#### ImportError: No module named 'xyz'
**Problem**: Missing Python dependencies

**Solution**:
```bash
pip install -r requirements.txt
```

#### Tesseract not found
**Problem**: OCR functionality requires Tesseract

**Solution**:
```bash
brew install tesseract
# For additional languages:
brew install tesseract-lang
```

### Service Connection Issues

#### GROBID Connection Failed
**Problem**: GROBID service not running or unreachable

**Symptoms**:
- Error: "GROBID health check failed"
- Metadata extraction returns None

**Solutions**:

1. Check if GROBID is running:
```bash
curl http://localhost:8070/api/isalive
```

2. Start GROBID:
```bash
docker run -d -p 8070:8070 --name grobid lfoppiano/grobid:0.8.0
```

3. Restart GROBID:
```bash
docker restart grobid
```

4. Check GROBID logs:
```bash
docker logs grobid
```

#### Ollama Connection Failed
**Problem**: Ollama service not running or models not available

**Symptoms**:
- Error: "Ollama connection failed"
- "Model xyz not found"

**Solutions**:

1. Check Ollama is running:
```bash
ollama list
```

2. Start Ollama (if not running):
```bash
# On macOS:
brew services start ollama
```

3. Pull required models:
```bash
ollama pull llama3.1:8b
ollama pull nomic-embed-text
```

4. Verify models are available:
```bash
ollama list
```

### Processing Issues

#### OCR Taking Too Long
**Problem**: OCR is slow on large documents

**Solutions**:

1. Enable skip_ocr_if_text_exists in config:
```yaml
processing:
  skip_ocr_if_text_exists: true
```

2. Reduce workers if memory is limited:
```bash
python cli.py process --workers 2 ...
```

3. Pre-process with faster settings:
```bash
ocrmypdf --skip-text --fast input.pdf output.pdf
```

#### Out of Memory
**Problem**: Processing too many papers in parallel

**Solutions**:

1. Reduce batch size:
```yaml
processing:
  batch_size: 16
  workers: 2
```

2. Clear cache:
```bash
make clean
```

3. Process categories separately:
```bash
# Process one category at a time
python cli.py process --root-dir /path/CategoryA --topic "..."
```

#### Embeddings Generation Fails
**Problem**: Embedding model not responding or timing out

**Solutions**:

1. Check Ollama memory usage:
```bash
docker stats  # If running Ollama in Docker
```

2. Use smaller model:
```yaml
ollama:
  embed_model: "nomic-embed-text"  # Faster than larger models
```

3. Restart Ollama:
```bash
brew services restart ollama
```

### File Moving Issues

#### Papers Not Moving Despite Reclassification
**Problem**: Dry-run mode is enabled or moves disabled

**Solutions**:

1. Check dry-run flag:
```bash
# Remove --dry-run flag
python cli.py process --root-dir ... --topic "..." 
```

2. Check config:
```yaml
move:
  enabled: true
```

#### Duplicate Manifest Entries After Move
**Problem**: This shouldn't happen with manifest tracking

**Diagnosis**:
```bash
# Check manifest files
cat outputs/manifests/*.manifest.json | jq '.entries[] | select(.paper_id=="<paper_id>")'
```

**Solution**:
The manifest system automatically prevents this. If it occurs:
1. Check logs in `outputs/logs/`
2. Verify manifest_manager.save_all() is being called
3. Report as a bug with logs

#### Files Not Found After Move
**Problem**: Paths in cache/manifests are outdated

**Solution**:
```bash
# Clear cache and re-run
make clean
python cli.py process --root-dir ... --topic "..." --resume=false
```

### Output Issues

#### CSV Has Missing Columns
**Problem**: Some papers missing metadata fields

**Solution**: This is normal. Not all PDFs have complete metadata. Check:
- GROBID extraction logs
- PDF internal metadata quality
- Enable Crossref enrichment for better metadata

#### Markdown Summaries Are Generic
**Problem**: LLM not focusing on topic

**Solutions**:

1. Make topic description more specific:
```python
topic = """
Focus on specific aspects: [list 3-5 key areas]
Methods of interest: [specific techniques]
Exclude: [what to ignore]
"""
```

2. Adjust temperature:
```yaml
ollama:
  temperature: 0.2  # More focused (lower = more deterministic)
```

3. Use different model:
```yaml
ollama:
  summarize_model: "qwen2:7b-instruct"  # Sometimes better reasoning
```

#### Low Relevance Scores Across the Board
**Problem**: Topic embedding doesn't match paper embeddings well

**Solutions**:

1. Check topic description clarity
2. Verify embedding model is working:
```bash
python -c "
import ollama
result = ollama.embeddings(model='nomic-embed-text', prompt='test')
print(len(result['embedding']))
"
```

3. Recalibrate threshold based on score distribution:
```bash
# Check statistics.json
cat outputs/statistics.json
```

### Performance Optimization

#### Pipeline Running Slowly
**Bottlenecks and Solutions**:

1. **OCR**: Skip if not needed, use --skip-ocr-if-text-exists
2. **GROBID**: Increase batch_size in config
3. **Embeddings**: Batch more aggressively (batch_size: 64)
4. **LLM Calls**: Use faster model for classification (mistral:7b)

#### Caching Not Working
**Problem**: Re-processing same papers

**Solution**:
```bash
# Check cache is enabled
cat config.yaml | grep -A3 "cache:"

# Verify cache directory exists
ls -la cache/

# Check cache DB
sqlite3 cache/cache.db ".tables"
```

### Data Quality Issues

#### BibTeX Citations Incomplete
**Problem**: GROBID extraction incomplete or failed

**Solutions**:

1. Enable Crossref enrichment:
```yaml
crossref:
  enabled: true
  email: "your@email.com"  # For polite pool (higher rate limit)
```

2. Check PDF quality (scanned vs born-digital)
3. Manual correction may be needed for some papers

#### Duplicates Not Detected
**Problem**: Near-duplicates threshold too high

**Solutions**:

1. Lower similarity threshold:
```yaml
dedup:
  similarity_threshold: 0.90  # Lower = more sensitive
```

2. Check text extraction quality:
```bash
# View extracted text
cat cache/text_extracts/* | less
```

### Debugging

#### Enable Verbose Logging
```bash
# Edit cli.py logging level
logging.basicConfig(level=logging.DEBUG)
```

#### Check Intermediate Outputs
```bash
# Cache contents
ls -la cache/

# Manifests
cat outputs/manifests/*.manifest.json | jq .

# Logs
tail -f outputs/logs/pipeline_*.log
```

#### Test Individual Components
```python
# Test GROBID
from utils.grobid_client import GrobidClient
client = GrobidClient()
print(client.is_alive())

# Test Ollama
from core.embeddings import EmbeddingGenerator
gen = EmbeddingGenerator()
print(gen.test_connection())
```

## Still Having Issues?

1. Check logs in `outputs/logs/`
2. Verify all prerequisites are installed
3. Try with `--dry-run` first
4. Test with a small sample (5-10 PDFs)
5. Clear cache and retry: `make clean`

## Performance Benchmarks

Typical processing times (2023 MacBook, M2):
- Inventory: ~1-2 seconds per 100 PDFs
- PDF parsing: ~2-5 seconds per PDF (without OCR)
- OCR: ~30-60 seconds per scanned PDF
- GROBID: ~3-5 seconds per PDF
- Embeddings: ~0.5-1 second per paper
- Scoring: ~0.1 seconds per paper
- Classification: ~2-3 seconds per paper
- Summarization: ~5-10 seconds per paper

For 100 papers: expect 30-60 minutes total
