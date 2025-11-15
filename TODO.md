# 📋 Research Assistant - TODO List

**Generated**: November 14, 2025  
**Priority**: CRITICAL → HIGH → MEDIUM → LOW

---

## 🚨 CRITICAL (Must Fix Before Production)

### 1. ✅ **Implement Gemini Rate Limiting** 
**Priority**: 🔴 CRITICAL  
**Status**: ✅ **COMPLETE**  
**Completed**: November 14, 2025

**Gemini Free Tier Limits**:
- ✅ **10 RPM** (Requests Per Minute)
- ✅ **500 RPD** (Requests Per Day)
- ✅ **1M TPM** (Tokens Per Minute)

**Implemented**:
- ✅ Created `utils/rate_limiter.py` with RateLimiter class
- ✅ RPM enforcement with artificial delays (6+ seconds between requests)
- ✅ RPD tracking with persistent state in `cache/rate_limit_state.json`
- ✅ Warnings at 50% (250) and 75% (375) of daily limit
- ✅ Interactive prompt at 500 RPD with options to pause/switch/continue
- ✅ Integrated into `utils/llm_provider.py` for all Gemini calls
- ✅ Thread-safe implementation with locks
- ✅ Tested and verified with 60s delay after 3 requests

**Files Modified**:
- `utils/rate_limiter.py` (NEW) - Complete implementation
- `utils/llm_provider.py` - Integrated rate limiter
- `config.py` - Added RateLimitConfig

---

### 2. ❌ **End-to-End Pipeline Test**
**Priority**: 🔴 CRITICAL  
**Estimated Time**: 1-2 hours  
**Issue**: New pipeline not tested with real papers

**Test Plan**:
- [ ] Select 10 test papers from existing dataset
- [ ] Run full pipeline with `--workers 1`
- [ ] Verify category generation (Pass 1)
- [ ] Verify metadata extraction (Pass 3)
- [ ] Verify multi-category scoring (Pass 3)
- [ ] Verify paper movement to folders (Pass 4)
- [ ] Verify deduplication (Pass 5)
- [ ] Verify manifests updated (Pass 6)
- [ ] Verify summaries generated (Pass 7)
- [ ] Verify index.jsonl/csv created (Pass 8)
- [ ] Test resume functionality
- [ ] Fix any bugs found

**Success Criteria**:
- ✅ All 8 passes complete without errors
- ✅ Papers in correct category folders
- ✅ Manifests have correct structure
- ✅ Index has all new fields
- ✅ Resume skips already-processed papers

---

### 3. ✅ **Category Name Validation & Sanitization**
**Priority**: 🔴 CRITICAL  
**Status**: ✅ **COMPLETE**  
**Completed**: November 15, 2025

**Implemented**:
- ✅ Created `core/category_validator.py` with CategoryValidator class
- ✅ Name sanitization: lowercase, spaces→underscores, special chars removed
- ✅ Path injection prevention: blocks `..`, `/`, `\`, `:`, `~`, wildcards
- ✅ Reserved name protection: prevents system folder names
- ✅ Fuzzy matching: "attack vector" matches "attack_vectors" (singular/plural, typos)
- ✅ Duplicate detection and prevention (case-insensitive)
- ✅ Category count validation: enforces 3-25 range
- ✅ Integrated into `core/taxonomy.py` for LLM generation
- ✅ Integrated into `core/metadata.py` for classification matching
- ✅ 25 unit tests with 96% code coverage
- ✅ 3 integration tests validating full pipeline

**Files Created**:
- `core/category_validator.py` (NEW) - Complete validation module
- `tests/test_category_validator.py` (NEW) - 25 comprehensive tests
- `tests/test_taxonomy_validation.py` (NEW) - Integration tests

**Files Modified**:
- `core/taxonomy.py` - Integrated validation
- `core/metadata.py` - Integrated matching

**Code**:
```python
def sanitize_category_name(name: str) -> str:
    """Sanitize category name to be filesystem-safe."""
    # Lowercase and strip
    name = name.lower().strip()
    # Replace spaces/dashes with underscore
    name = re.sub(r'[\s-]+', '_', name)
    # Remove all non-alphanumeric except underscore
    name = re.sub(r'[^\w_]', '', name)
    # Remove leading/trailing underscores
    name = name.strip('_')
    return name

def validate_categories(categories: Dict[str, str]) -> Dict[str, str]:
    """Validate and sanitize category dictionary."""
    if not categories or len(categories) == 0:
        raise ValueError("LLM generated 0 categories")
    
    if len(categories) > 25:
        logger.warning(f"LLM generated {len(categories)} categories (>25), may be too many")
    
    sanitized = {}
    seen = set()
    
    for name, definition in categories.items():
        clean = sanitize_category_name(name)
        
        if not clean:
            logger.warning(f"Skipping invalid category: '{name}'")
            continue
            
        if clean in seen:
            logger.warning(f"Duplicate category detected: '{name}' → '{clean}'")
            continue
            
        if clean != name:
            logger.warning(f"Sanitized category: '{name}' → '{clean}'")
            
        sanitized[clean] = definition
        seen.add(clean)
    
    if len(sanitized) < 3:
        raise ValueError(f"Only {len(sanitized)} valid categories after sanitization (need ≥3)")
    
    return sanitized
```

---

## 🔴 HIGH Priority (Critical for Production Quality)

### 4. ⚠️ **Schema-Based Summary Generation**
**Priority**: 🔴 HIGH  
**Status**: ✅ **COMPLETE**  
**Completed**: November 15, 2025

**Issue**: Unstructured LLM summaries had artifacts and inconsistent formatting

**Implemented**:
- ✅ Created `PaperSummarySchema` with structured fields
- ✅ JSON-based extraction: main_contributions, topic_relevance, key_techniques, etc.
- ✅ Consistent markdown template formatting
- ✅ Works with both Gemini (native schema) and Ollama (JSON parsing)
- ✅ Eliminates "Okay, here is..." artifacts
- ✅ 152 words vs 496 words (more focused and concise)
- ✅ No truncation issues

**Files Modified**:
- `core/summarizer.py` - Complete rewrite with schema
- `test_summary_truncation.py` (NEW) - Validation test

---

### 5. ❌ **Error Handling & Recovery**
**Priority**: 🔴 HIGH  
**Estimated Time**: 3-4 hours

**Scenarios to Handle**:
- [ ] **LLM API Failures**
  - Rate limit hit: retry with exponential backoff
  - Invalid API key: clear error message with setup instructions
  - Network timeout: retry up to 3 times
  - Malformed JSON: fallback classification
  
- [ ] **File Operations**
  - File move fails: log error, continue with others
  - Folder creation fails: clear error message
  - Permission denied: skip file, log warning
  
- [ ] **Category Generation Fails**
  - LLM error: retry once, then use fallback categories
  - Fallback: `["relevant", "related", "tangential", "other"]`
  
- [ ] **Unreadable PDFs**
  - ✅ Already handled: move to `need_human_element/`
  - Need to verify manifest updates correctly
  
- [ ] **Cache Corruption**
  - Detect corrupt cache entries
  - Skip and regenerate
  - Log warning

**Implementation**:
```python
@retry(max_attempts=3, backoff=2.0)
def safe_llm_call(prompt: str):
    """LLM call with retry logic."""
    try:
        return llm_generate(prompt)
    except RateLimitError:
        logger.warning("Rate limit hit, waiting 60s...")
        time.sleep(60)
        raise  # Retry
    except APIKeyError:
        logger.error("Invalid API key! Check .env file")
        sys.exit(1)
    except Exception as e:
        logger.error(f"LLM call failed: {e}")
        raise

def safe_file_move(src: Path, dst: Path) -> bool:
    """Move file with error handling."""
    try:
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.move(str(src), str(dst))
        return True
    except PermissionError:
        logger.error(f"Permission denied: {src} → {dst}")
        return False
    except Exception as e:
        logger.error(f"Failed to move {src}: {e}")
        return False
```

---

### 6. ⚠️ **Write Critical Tests**
**Priority**: 🔴 HIGH  
**Status**: 🟡 **IN PROGRESS** (28 tests complete)  
**Estimated Time**: 4-6 hours remaining

**Test Coverage Completed**:

**`tests/test_category_validator.py`** ✅ (NEW):
- ✅ 25 unit tests, all passing
- ✅ 96% code coverage
- ✅ Sanitization, validation, matching, duplicate detection

**`tests/test_taxonomy_validation.py`** ✅ (NEW):
- ✅ 3 integration tests
- ✅ Full pipeline validation with sanitization

**Test Coverage Still Needed**:

**`tests/test_taxonomy.py`** (NEW):
- [ ] `test_generate_categories_from_topic()` - Basic generation
- [ ] `test_category_caching()` - Cache reuse
- [ ] `test_force_regenerate()` - Ignore cache
- [ ] `test_invalid_topic()` - Empty topic handling

**`tests/test_metadata.py`** (UPDATE):
- [ ] `test_multi_category_scoring()` - All categories scored
- [ ] `test_best_category_selection()` - Highest score wins
- [ ] `test_topic_relevance_scoring()` - 1-10 range
- [ ] `test_reasoning_generation()` - Explanation provided
- [ ] `test_tie_breaking()` - Equal scores handled
- [ ] `test_llm_failure_fallback()` - Error handling

**`tests/test_manifest.py`** (UPDATE):
- [ ] `test_new_manifest_structure()` - All fields present
- [ ] `test_add_paper_with_scores()` - Scores stored
- [ ] `test_should_skip_analyzed()` - Resume logic
- [ ] `test_duplicate_tracking()` - Canonical ID

**`tests/test_cli.py`** (UPDATE):
- [ ] `test_end_to_end_pipeline()` - Full 8-pass run
- [ ] `test_resume_functionality()` - Skip analyzed papers
- [ ] `test_category_folder_creation()` - Dynamic folders
- [ ] `test_quarantine_low_relevance()` - < threshold

**`tests/test_rate_limiter.py`** (NEW):
- [ ] `test_rpm_limiting()` - Sleep enforced
- [ ] `test_rpd_tracking()` - Counter accurate
- [ ] `test_rpd_warnings()` - 50%, 75% warnings
- [ ] `test_rpd_stop()` - Stop at 500

**Target**: 60% coverage minimum  
**Current**: ~15% overall (category_validator at 96%)

---

### 7. ❌ **Enhanced Logging & Debugging**
**Priority**: 🔴 HIGH  
**Estimated Time**: 2 hours

**Add Detailed Logs**:
- [ ] **Category Generation**:
  ```
  [TAXONOMY] Generating categories for topic: "Prompt Injection Attacks"
  [TAXONOMY] LLM generated 14 categories
  [TAXONOMY] Sanitized 2 category names
  [TAXONOMY] Categories: attack_vectors, defense_mechanisms, ...
  [TAXONOMY] Cached in: cache/categories.json
  ```

- [ ] **Multi-Category Scoring**:
  ```
  [CLASSIFICATION] Paper: "Defending Against Prompt Injection"
  [CLASSIFICATION] Topic relevance: 8/10
  [CLASSIFICATION] Category scores:
    • attack_vectors: 3/10
    • defense_mechanisms: 9/10 ← BEST
    • detection_methods: 6/10
  [CLASSIFICATION] Reasoning: Paper proposes input validation...
  ```

- [ ] **Rate Limiting**:
  ```
  [RATE LIMIT] Request 1/10 this minute
  [RATE LIMIT] Sleeping 6.2 seconds to stay under 10 RPM
  [RATE LIMIT] Daily requests: 127/500 (25%)
  ```

- [ ] **Paper Movement**:
  ```
  [MOVE] paper_123.pdf → defense_mechanisms/
  [MOVE] Reason: Category score 9/10, topic relevance 8/10
  ```

**Implementation**:
- Add logging statements in critical functions
- Use structured logging (JSON format option)
- Add `--verbose` flag for extra detail
- Add timing logs for performance tracking

---

## 🟡 MEDIUM Priority (Important but Not Blocking)

### 8. ⚠️ **Performance Profiling & Optimization**
**Priority**: 🟡 MEDIUM  
**Estimated Time**: 3-4 hours

**Measure**:
- [ ] Time per pipeline pass
- [ ] API call latency
- [ ] Memory usage with workers
- [ ] Disk I/O for large files
- [ ] Cache hit rate

**Profile**:
```python
import cProfile
import pstats

profiler = cProfile.Profile()
profiler.enable()
# Run pipeline
profiler.disable()
stats = pstats.Stats(profiler)
stats.sort_stats('cumulative')
stats.print_stats(20)
```

**Optimize** (if needed):
- [ ] Batch API calls (if possible)
- [ ] Reduce redundant file I/O
- [ ] Optimize MinHash deduplication
- [ ] Cache more aggressively

**Target**: < 20 seconds per paper (average)

---

### 9. ⚠️ **Configuration Validation**
**Priority**: 🟡 MEDIUM  
**Estimated Time**: 1 hour

**Validate Inputs**:
- [ ] `min_topic_relevance`: Must be 1-10
- [ ] `workers`: Must be 1-10
- [ ] `topic`: Must be non-empty string
- [ ] `root_dir`: Must exist and be readable
- [ ] API keys: Must be present if using Gemini

**Implementation**:
```python
def validate_config(config: Config):
    """Validate configuration before pipeline starts."""
    errors = []
    
    if not 1 <= config.scoring.min_topic_relevance <= 10:
        errors.append("min_topic_relevance must be 1-10")
    
    if not 1 <= config.processing.workers <= 10:
        errors.append("workers must be 1-10")
    
    if not config.topic or not config.topic.strip():
        errors.append("topic cannot be empty")
    
    if not config.root_dir.exists():
        errors.append(f"root_dir does not exist: {config.root_dir}")
    
    if config.llm_provider == "gemini" and not config.gemini.api_key:
        errors.append("GEMINI_API_KEY not set in .env")
    
    if errors:
        for err in errors:
            logger.error(f"❌ {err}")
        sys.exit(1)
```

---

### 10. ⚠️ **Better Error Messages**
**Priority**: 🟡 MEDIUM  
**Estimated Time**: 1-2 hours

**Improve User Experience**:
- [ ] **Missing API Key**:
  ```
  ❌ ERROR: GEMINI_API_KEY not found!
  
  To fix:
  1. Create a .env file in project root
  2. Add this line:
     GEMINI_API_KEY=your_key_here
  3. Get your key from:
     https://aistudio.google.com/app/apikey
  ```

- [ ] **Rate Limit Hit**:
  ```
  ⚠️  RATE LIMIT: Reached 500 requests for today
  
  Options:
  1. Pause and resume tomorrow:
     - Press Ctrl+C to stop
     - Run again tomorrow with --resume flag
     
  2. Switch to local Ollama:
     - Install: https://ollama.com/download
     - Run: ollama pull deepseek-r1:8b
     - Restart with: --llm-provider ollama
     
  What would you like to do? [pause/ollama/continue]:
  ```

- [ ] **No Papers Found**:
  ```
  ❌ ERROR: No PDF files found in /path/to/dir
  
  Make sure:
  - PDFs are in the root directory (not subdirectories yet)
  - Files have .pdf extension
  - You have read permissions
  ```

---

### 11. ⚠️ **Resume Robustness**
**Priority**: 🟡 MEDIUM  
**Estimated Time**: 2 hours

**Improve Resume Logic**:
- [ ] Handle partial index files (corrupted)
- [ ] Resume after category generation step
- [ ] Resume after deduplication
- [ ] Clear documentation on what --resume does
- [ ] Add `--resume-from-pass N` flag

**Implementation**:
```python
# Save checkpoint after each pass
def save_checkpoint(pass_num: int, state: dict):
    checkpoint_file = cache_dir / f"checkpoint_pass{pass_num}.json"
    with open(checkpoint_file, 'w') as f:
        json.dump(state, f)

def load_checkpoint(pass_num: int) -> Optional[dict]:
    checkpoint_file = cache_dir / f"checkpoint_pass{pass_num}.json"
    if checkpoint_file.exists():
        with open(checkpoint_file, 'r') as f:
            return json.load(f)
    return None
```

---

## 🟢 LOW Priority (Nice to Have)

### 12. 💚 **Migration Tool for Old Data**
**Priority**: 🟢 LOW  
**Estimated Time**: 4-6 hours

**Script to Convert**:
- [ ] Old manifests → New structure
- [ ] Old index → New fields
- [ ] Pre-categorized folders → Flat structure

**Usage**:
```bash
python migrate.py \
  --old-output old_outputs/ \
  --new-output outputs/ \
  --topic "Your Topic"
```

---

### 13. 💚 **Dashboard / Visualization**
**Priority**: 🟢 LOW  
**Estimated Time**: 8-10 hours

**Features**:
- [ ] Category distribution pie chart
- [ ] Topic relevance histogram
- [ ] Category score heatmap
- [ ] Timeline of papers by year
- [ ] Author network graph
- [ ] Keyword cloud per category

**Tech Stack**:
- Streamlit or Dash for UI
- Plotly for charts
- NetworkX for graphs

---

### 14. 💚 **Batch Processing Mode**
**Priority**: 🟢 LOW  
**Estimated Time**: 2-3 hours

**Process Multiple Topics**:
```bash
# Process 3 topics in sequence
python cli.py batch \
  --topics "Prompt Injection,LLM Security,Adversarial ML" \
  --root-dir papers/ \
  --output-dir outputs/
```

**Implementation**:
- Loop over topics
- Generate categories for each
- Classify papers under all topics
- Multi-dimensional categorization

---

### 15. 💚 **Export Formats**
**Priority**: 🟢 LOW  
**Estimated Time**: 2 hours

**Additional Outputs**:
- [ ] BibTeX file (all papers)
- [ ] Zotero RDF export
- [ ] EndNote XML
- [ ] JSON-LD for semantic web
- [ ] SQLite database

---

### 16. 💚 **Interactive CLI**
**Priority**: 🟢 LOW  
**Estimated Time**: 3-4 hours

**Better UX**:
```bash
python cli.py interactive

? Select PDFs directory: [Browse]
? Enter research topic: Prompt Injection Attacks
? Choose LLM provider: ● Gemini  ○ Ollama
? Number of workers: 2
? Min topic relevance (1-10): 5

✓ Configuration saved
→ Starting pipeline...
```

**Tech Stack**: `questionary` or `rich`

---

## 📅 Implementation Timeline

### **Week 1: Critical Issues**
- **Day 1-2**: Rate limiting (RPM + RPD tracking)
- **Day 3**: End-to-end testing (10 papers)
- **Day 4**: Bug fixes from testing
- **Day 5**: Category name validation

### **Week 2: Quality & Testing**
- **Day 6-7**: Error handling & recovery
- **Day 8-10**: Write test suite (60% coverage)

### **Week 3: Polish**
- **Day 11**: Enhanced logging
- **Day 12**: Performance profiling
- **Day 13**: Better error messages
- **Day 14**: Configuration validation

### **Week 4: Production**
- **Day 15**: Full dataset test (500 papers)
- **Day 16**: Documentation finalization
- **Day 17**: Deployment & monitoring
- **Day 18**: Buffer for issues

---

## ✅ Definition of Done

**Pipeline is production-ready when**:

1. ✅ Rate limiting prevents API failures
2. ✅ End-to-end test passes (10 papers)
3. ✅ Full dataset test passes (500 papers)
4. ✅ Test coverage ≥ 60%
5. ✅ All critical error scenarios handled
6. ✅ Resume functionality verified
7. ✅ Category names validated/sanitized
8. ✅ Logging provides clear debugging info
9. ✅ Performance < 30s per paper average
10. ✅ Documentation complete and accurate

**Current Status**: 4/10 ✅

1. ✅ Rate limiting prevents API failures
2. ⚠️ End-to-end test passes (10 papers) - **NEEDS TESTING**
3. ⚠️ Full dataset test passes (500 papers) - **PENDING**
4. 🟡 Test coverage ≥ 60% - **CURRENT: ~25%**
5. 🟡 All critical error scenarios handled - **PARTIAL**
6. ⚠️ Resume functionality verified - **NEEDS TESTING**
7. ✅ Category names validated/sanitized
8. 🟡 Logging provides clear debugging info - **PARTIAL**
9. ⚠️ Performance < 30s per paper average - **NEEDS MEASUREMENT**
10. ✅ Documentation complete and accurate

---

## 📊 Progress Tracking

**Overall Progress**: � 45% Complete

- ✅ **Architecture**: 100% (refactoring done)
- ✅ **Rate Limiting**: 100% (fully implemented & tested)
- ✅ **Category Validation**: 100% (comprehensive validation & 96% coverage)
- ✅ **Schema-Based Summaries**: 100% (structured output working)
- 🟡 **Testing**: 25% (28 tests complete, need more coverage)
- 🟡 **Error Handling**: 30% (basic coverage)
- ✅ **Documentation**: 95% (up to date)
- 🟡 **Logging**: 50% (basic logging)

**Recent Completions** (Nov 14-15, 2025):
- ✅ Gemini rate limiting (RPM + RPD tracking)
- ✅ Category name validation & sanitization
- ✅ Schema-based summary generation
- ✅ 28 tests written with high coverage for validators

---

---

## 🎯 Immediate Next Steps (Updated)

1. ✅ **Project cleanup** - Organized docs/ and tests/ folders
2. **Test full pipeline** - Run with 5-10 papers from test_pdfs/
3. **Fix any bugs** found during testing
4. **Convert manual tests** to proper pytest tests
5. **Update README** with new structure

---

**Last Updated**: November 15, 2025  
**Next Review**: After end-to-end testing with 10 papers
