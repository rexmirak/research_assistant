"""Test script for taxonomy generation."""

import os
from pathlib import Path
from core.taxonomy import TaxonomyGenerator

# Setup - use Gemini
os.environ["LLM_PROVIDER"] = "gemini"

cache_dir = Path("./cache")
output_dir = Path("./outputs")
cache_dir.mkdir(exist_ok=True)
output_dir.mkdir(exist_ok=True)

# Create generator
generator = TaxonomyGenerator(cache_dir=cache_dir, output_dir=output_dir)

# Test topic
topic = "Prompt Injection Attacks in Large Language Models"

print("=" * 80)
print("TESTING CATEGORY GENERATION")
print("=" * 80)
print(f"Topic: {topic}")
print("")

# Generate categories
print("Generating categories from topic (no papers used)...")
categories = generator.generate_categories(topic=topic, force_regenerate=True)

print(f"\nGenerated {len(categories)} categories:")
print("")

for name, definition in categories.items():
    print(f"• {name}")
    print(f"  {definition}")
    print("")

print("=" * 80)
print("SUCCESS! Check the following files:")
print(f"  - {output_dir / 'categories.json'}")
print(f"  - {cache_dir / 'categories.json'}")
print("=" * 80)
