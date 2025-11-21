#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Test recursive embedding extraction."""

import sys
import os
import numpy as np
from pathlib import Path
import tempfile
import shutil

# Set UTF-8 encoding for Windows console
if sys.platform == 'win32':
    import codecs
    sys.stdout = codecs.getwriter('utf-8')(sys.stdout.buffer, 'strict')
    sys.stderr = codecs.getwriter('utf-8')(sys.stderr.buffer, 'strict')

# Add current dir to path
sys.path.insert(0, str(Path(__file__).parent))

from gtmo_json_saver import GTMOOptimizedSaver

# Create test data with nested embeddings
test_result = {
    "article_number": 1,
    "herbert_embedding": list(np.random.rand(768)),  # Article-level embedding
    "paragraphs": [
        {
            "paragraph_number": 1,
            "herbert_embedding": list(np.random.rand(768)),  # Paragraph embedding
            "sentences": [
                {
                    "sentence_number": 1,
                    "herbert_embedding": list(np.random.rand(768))  # Sentence embedding
                },
                {
                    "sentence_number": 2,
                    "herbert_embedding": list(np.random.rand(768))  # Sentence embedding
                }
            ]
        },
        {
            "paragraph_number": 2,
            "herbert_embedding": list(np.random.rand(768)),  # Paragraph embedding
            "sentences": [
                {
                    "sentence_number": 1,
                    "herbert_embedding": list(np.random.rand(768))  # Sentence embedding
                }
            ]
        }
    ]
}

print("=" * 70)
print("Test Recursive Embedding Extraction")
print("=" * 70)

# Create temp directory for test
temp_dir = Path(tempfile.mkdtemp())
print(f"📁 Test directory: {temp_dir}")

try:
    # Initialize saver
    saver = GTMOOptimizedSaver(output_dir=str(temp_dir), save_embeddings=True)

    # Create analysis folder
    analysis_folder = saver.create_analysis_folder("test_doc")
    print(f"✅ Created analysis folder: {analysis_folder}")

    # Count embeddings before extraction
    def count_embeddings(data):
        count = 0
        if 'herbert_embedding' in data and isinstance(data['herbert_embedding'], list):
            count += 1
        if 'paragraphs' in data:
            for para in data['paragraphs']:
                count += count_embeddings(para)
        if 'sentences' in data:
            for sent in data['sentences']:
                count += count_embeddings(sent)
        return count

    before_count = count_embeddings(test_result)
    print(f"\n📊 Embeddings before extraction: {before_count}")
    print(f"   - Article level: 1")
    print(f"   - Paragraph level: 2")
    print(f"   - Sentence level: 3")

    # Extract embeddings recursively
    extracted_count = saver._extract_embeddings_recursive(test_result, "article_001")
    print(f"\n✅ Extracted {extracted_count} embeddings")

    # Save embeddings to .npz
    npz_file = saver.finalize_embeddings()
    print(f"💾 Saved embeddings to: {npz_file}")

    # Load and verify
    with np.load(npz_file) as data:
        keys = list(data.keys())
        print(f"\n🔍 Verification:")
        print(f"   Keys in .npz: {len(keys)}")
        print(f"   Expected: {before_count}")
        print(f"   Match: {'✅ YES' if len(keys) == before_count else '❌ NO'}")
        print(f"\n   All keys: {keys}")

    # Check that embeddings in dict are now references
    def check_references(data, path=""):
        refs = []
        if 'herbert_embedding' in data:
            if isinstance(data['herbert_embedding'], dict) and data['herbert_embedding'].get('_type') == 'reference':
                refs.append(f"{path}/herbert_embedding")
        if 'paragraphs' in data:
            for i, para in enumerate(data['paragraphs']):
                refs.extend(check_references(para, f"{path}/para{i}"))
        if 'sentences' in data:
            for i, sent in enumerate(data['sentences']):
                refs.extend(check_references(sent, f"{path}/sent{i}"))
        return refs

    references = check_references(test_result)
    print(f"\n📝 References created: {len(references)}")
    print(f"   Expected: {before_count}")
    print(f"   Match: {'✅ YES' if len(references) == before_count else '❌ NO'}")

    if len(keys) == before_count and len(references) == before_count:
        print("\n" + "=" * 70)
        print("✅ TEST PASSED - Recursive extraction works correctly!")
        print("=" * 70)
    else:
        print("\n" + "=" * 70)
        print("❌ TEST FAILED")
        print("=" * 70)

finally:
    # Clean up
    shutil.rmtree(temp_dir)
    print(f"\n🧹 Cleaned up test directory")
