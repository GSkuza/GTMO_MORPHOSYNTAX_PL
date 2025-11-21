#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Test HerBERT Availability Diagnostic
Sprawdza czy HerBERT jest dostępny i może generować embeddingi.
"""

import sys
import os
from datetime import datetime

# Set UTF-8 encoding for Windows console
if sys.platform == 'win32':
    import codecs
    sys.stdout = codecs.getwriter('utf-8')(sys.stdout.buffer, 'strict')
    sys.stderr = codecs.getwriter('utf-8')(sys.stderr.buffer, 'strict')

def test_herbert():
    """Test HerBERT model availability and embedding generation."""
    print("=" * 70)
    print("HerBERT Availability Diagnostic Test")
    print("=" * 70)
    print(f"Timestamp: {datetime.now()}")
    print(f"Python: {sys.version}")
    print(f"Working directory: {os.getcwd()}")
    print()

    # Test 1: Import transformers
    print("[1] Test: Import transformers...")
    try:
        from transformers import AutoTokenizer, AutoModel
        print("   [OK] transformers imported successfully")
    except ImportError as e:
        print(f"   [FAIL] Failed to import transformers: {e}")
        return False

    # Test 2: Import torch
    print("[2] Test: Import torch...")
    try:
        import torch
        print(f"   [OK] torch imported successfully (version: {torch.__version__})")
    except ImportError as e:
        print(f"   [FAIL] Failed to import torch: {e}")
        return False

    # Test 3: Load HerBERT model
    print("[3] Test: Load HerBERT model...")
    try:
        model = AutoModel.from_pretrained('allegro/herbert-base-cased')
        tokenizer = AutoTokenizer.from_pretrained('allegro/herbert-base-cased')
        model.eval()
        print("   [OK] HerBERT model loaded successfully")
    except Exception as e:
        print(f"   [FAIL] Failed to load HerBERT: {e}")
        return False

    # Test 4: Generate embedding
    print("[4] Test: Generate test embedding...")
    try:
        test_text = "To jest testowe zdanie po polsku."
        with torch.no_grad():
            inputs = tokenizer(test_text, return_tensors="pt",
                             truncation=True, max_length=512, padding=True)
            outputs = model(**inputs)
            embedding = outputs.last_hidden_state[:, 0, :].squeeze().numpy()
        print(f"   [OK] Embedding generated successfully")
        print(f"   Shape: {embedding.shape}")
        print(f"   Dtype: {embedding.dtype}")
        print(f"   Sample values: {embedding[:5]}")
    except Exception as e:
        print(f"   [FAIL] Failed to generate embedding: {e}")
        return False

    # Test 5: Check gtmo_morphosyntax availability
    print("[5] Test: Check gtmo_morphosyntax.py...")
    try:
        import gtmo_morphosyntax
        print(f"   [OK] gtmo_morphosyntax imported")
        print(f"   HERBERT_AVAILABLE: {gtmo_morphosyntax.HERBERT_AVAILABLE}")
        print(f"   GLOBAL_HERBERT_MODEL: {gtmo_morphosyntax.GLOBAL_HERBERT_MODEL is not None}")
    except Exception as e:
        print(f"   [WARNING] Could not import gtmo_morphosyntax: {e}")

    print()
    print("=" * 70)
    print("[SUCCESS] ALL TESTS PASSED - HerBERT IS AVAILABLE")
    print("=" * 70)
    return True

if __name__ == "__main__":
    success = test_herbert()
    sys.exit(0 if success else 1)
