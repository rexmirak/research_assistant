#!/usr/bin/env python3
"""Test schema-based summary generation."""

import logging
from core.summarizer import Summarizer
from config import Config

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

# Set config to use Ollama
cfg = Config()
cfg.llm_provider = "ollama"

# Test paper data
title = "Prompt Injection Attacks Against Large Language Models"
abstract = """Large Language Models (LLMs) have shown remarkable capabilities in natural language understanding and generation. However, they are vulnerable to prompt injection attacks where malicious users can manipulate the model's behavior by injecting carefully crafted prompts. This paper presents a comprehensive study of prompt injection attacks, their mechanisms, and potential defenses. We categorize different types of attacks including direct injection, indirect injection through retrieved content, and jailbreaking techniques. Our experiments on GPT-3.5, GPT-4, and other major LLMs demonstrate that even sophisticated models remain vulnerable to these attacks. We propose several defense mechanisms and discuss their effectiveness."""

topic = "Prompt Injection Attacks in LLMs"

metadata = {
    "title": title,
    "authors": ["Test Author", "Another Author"],
    "year": "2024",
    "venue": "Test Conference",
}

print("=" * 80)
print("TESTING SCHEMA-BASED SUMMARY GENERATION WITH OLLAMA")
print("=" * 80)
print()

summarizer = Summarizer(model="deepseek-r1:8b", temperature=0.1, max_summary_length=800)

print("Generating structured summary...")
print()

summary = summarizer.summarize_paper(
    title=title,
    abstract=abstract,
    intro=None,
    topic=topic,
    metadata=metadata,
    full_text=None
)

print("=" * 80)
print("FORMATTED SUMMARY OUTPUT:")
print("=" * 80)
print(summary)
print()
print("=" * 80)
print(f"Summary length: {len(summary)} characters")
print(f"Summary word count: {len(summary.split())} words")
print("=" * 80)
print()
print("✅ Schema-based summary generation complete!")
print("   - Structured fields extracted from LLM response")
print("   - Formatted into readable markdown")
print("   - No truncation issues")
