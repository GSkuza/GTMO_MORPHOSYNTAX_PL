#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Test HerBERT Integration
"""

print("="*80)
print("TEST HERBERT INTEGRATION")
print("="*80)

# Test 1: Import dependencies
print("\n[TEST 1] Importing dependencies...")
try:
    from transformers import AutoTokenizer, AutoModel
    import torch
    import numpy as np
    print("✔ transformers, torch, numpy imported")
except ImportError as e:
    print(f"✗ Import failed: {e}")
    exit(1)

# Test 2: Load HerBERT model
print("\n[TEST 2] Loading HerBERT model...")
try:
    HERBERT_MODEL_NAME = "allegro/herbert-base-cased"
    herbert_tokenizer = AutoTokenizer.from_pretrained(HERBERT_MODEL_NAME)
    herbert_model = AutoModel.from_pretrained(HERBERT_MODEL_NAME)
    herbert_model.eval()
    print(f"✔ HerBERT loaded: {HERBERT_MODEL_NAME}")
    print(f"   Model type: {type(herbert_model).__name__}")
    print(f"   Tokenizer vocab size: {herbert_tokenizer.vocab_size}")
except Exception as e:
    print(f"✗ Model loading failed: {e}")
    exit(1)

# Test 3: Generate embeddings for test sentences
print("\n[TEST 3] Generating embeddings...")
test_sentences = [
    "Konstytucja Rzeczypospolitej Polskiej jest najwyższym prawem.",
    "Sąd rozpatruje sprawę w składzie trzech sędziów.",
    "Prawo do obrony jest fundamentalnym prawem każdego obywatela."
]

embeddings = []
for i, text in enumerate(test_sentences):
    try:
        with torch.no_grad():
            inputs = herbert_tokenizer(text, return_tensors="pt", 
                                      truncation=True, max_length=512, 
                                      padding=True)
            outputs = herbert_model(**inputs)
            embedding = outputs.last_hidden_state[:, 0, :].squeeze().numpy()
        
        embeddings.append(embedding)
        print(f"✔ Sentence {i+1}: shape={embedding.shape}, mean={embedding.mean():.4f}, std={embedding.std():.4f}")
        
        # Verify shape
        if embedding.shape != (768,):
            print(f"✗ ERROR: Expected shape (768,), got {embedding.shape}")
        
    except Exception as e:
        print(f"✗ Embedding generation failed for sentence {i+1}: {e}")

# Test 4: Check embedding properties
print("\n[TEST 4] Verifying embedding properties...")
if len(embeddings) == 3:
    emb_array = np.array(embeddings)
    print(f"✔ Generated {len(embeddings)} embeddings")
    print(f"   Shape: {emb_array.shape}")
    print(f"   Overall mean: {emb_array.mean():.4f}")
    print(f"   Overall std: {emb_array.std():.4f}")
    print(f"   Min value: {emb_array.min():.4f}")
    print(f"   Max value: {emb_array.max():.4f}")
    
    # Check similarity between sentences
    from numpy.linalg import norm
    
    def cosine_similarity(a, b):
        return np.dot(a, b) / (norm(a) * norm(b))
    
    print(f"\n   Cosine similarities:")
    print(f"   Sent 1 vs Sent 2: {cosine_similarity(embeddings[0], embeddings[1]):.4f}")
    print(f"   Sent 1 vs Sent 3: {cosine_similarity(embeddings[0], embeddings[2]):.4f}")
    print(f"   Sent 2 vs Sent 3: {cosine_similarity(embeddings[1], embeddings[2]):.4f}")

# Test 5: Test with gtmo_morphosyntax.py
print("\n[TEST 5] Testing integration with gtmo_morphosyntax.py...")
try:
    from gtmo_morphosyntax import analyze_quantum_with_axioms
    
    test_text = "Sąd wydaje wyrok w imieniu Rzeczypospolitej Polskiej."
    print(f"   Analyzing: '{test_text}'")
    
    result = analyze_quantum_with_axioms(test_text)
    
    if 'herbert_embedding' in result:
        emb = np.array(result['herbert_embedding'])
        print(f"✔ herbert_embedding found in result")
        print(f"   Shape: {emb.shape}")
        print(f"   Mean: {emb.mean():.4f}")
        print(f"   Std: {emb.std():.4f}")
        
        if emb.shape[0] == 768:
            print(f"✔ Correct dimension (768)")
        else:
            print(f"✗ Wrong dimension: {emb.shape}")
    else:
        print(f"✗ herbert_embedding NOT found in result")
        print(f"   Available keys: {list(result.keys())}")
        
except Exception as e:
    print(f"✗ Integration test failed: {e}")
    import traceback
    traceback.print_exc()

print("\n" + "="*80)
print("TEST COMPLETE")
print("="*80)
