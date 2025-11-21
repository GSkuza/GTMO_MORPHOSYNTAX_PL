#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Example: Loading HerBERT Embeddings from Analysis
==================================================

Demonstrates how to load and use HerBERT embeddings saved in .npz format.
"""

import numpy as np
from pathlib import Path
import json


def load_embeddings(analysis_folder: str):
    """
    Load HerBERT embeddings from an analysis folder.

    Args:
        analysis_folder: Path to analysis folder containing herbert_embeddings.npz

    Returns:
        Dictionary mapping keys to embedding arrays
    """
    embeddings_file = Path(analysis_folder) / "herbert_embeddings.npz"

    if not embeddings_file.exists():
        print(f"❌ No embeddings file found: {embeddings_file}")
        return {}

    # Load all embeddings
    with np.load(embeddings_file) as data:
        embeddings = {key: data[key] for key in data.files}

    size_kb = embeddings_file.stat().st_size / 1024
    print(f"✅ Loaded {len(embeddings)} embeddings from {embeddings_file.name}")
    print(f"   File size: {size_kb:.1f} KB (compressed)")
    print(f"   Keys: {list(embeddings.keys())[:5]}..." if len(embeddings) > 5 else f"   Keys: {list(embeddings.keys())}")

    return embeddings


def get_embedding_for_article(analysis_folder: str, article_number: int) -> np.ndarray:
    """
    Get HerBERT embedding for a specific article.

    Args:
        analysis_folder: Path to analysis folder
        article_number: Article number (1-indexed)

    Returns:
        Embedding array (768D) or None if not found
    """
    embeddings_file = Path(analysis_folder) / "herbert_embeddings.npz"

    if not embeddings_file.exists():
        return None

    key = f"article_{article_number:03d}"

    with np.load(embeddings_file) as data:
        if key in data:
            return data[key]

    return None


def compute_similarity(embedding1: np.ndarray, embedding2: np.ndarray) -> float:
    """
    Compute cosine similarity between two embeddings.

    Args:
        embedding1: First embedding (768D)
        embedding2: Second embedding (768D)

    Returns:
        Cosine similarity [-1, 1]
    """
    # Normalize embeddings
    norm1 = np.linalg.norm(embedding1)
    norm2 = np.linalg.norm(embedding2)

    if norm1 == 0 or norm2 == 0:
        return 0.0

    # Cosine similarity
    return np.dot(embedding1, embedding2) / (norm1 * norm2)


def example_usage():
    """Example usage of embedding loading."""
    import sys

    if len(sys.argv) < 2:
        print("Usage: python load_embeddings_example.py <analysis_folder>")
        print("\nExample:")
        print("  python load_embeddings_example.py gtmo_results/analysis_21112025_no1_document")
        sys.exit(1)

    analysis_folder = sys.argv[1]

    # Load all embeddings
    print("=" * 70)
    print("Loading HerBERT Embeddings")
    print("=" * 70)
    embeddings = load_embeddings(analysis_folder)

    if not embeddings:
        print("No embeddings found!")
        sys.exit(1)

    # Show embedding info
    first_key = list(embeddings.keys())[0]
    first_embedding = embeddings[first_key]

    print(f"\n📊 Embedding Details:")
    print(f"   Shape: {first_embedding.shape}")
    print(f"   Data type: {first_embedding.dtype}")
    print(f"   Memory per embedding: {first_embedding.nbytes / 1024:.2f} KB")

    # Compute similarities between consecutive articles
    if len(embeddings) > 1:
        print(f"\n🔍 Similarity Analysis:")
        keys = sorted(embeddings.keys())
        for i in range(len(keys) - 1):
            emb1 = embeddings[keys[i]]
            emb2 = embeddings[keys[i+1]]
            similarity = compute_similarity(emb1, emb2)
            print(f"   {keys[i]} ↔ {keys[i+1]}: {similarity:.4f}")

    # Example: Load specific article embedding
    print(f"\n📝 Load Specific Article:")
    article_emb = get_embedding_for_article(analysis_folder, 1)
    if article_emb is not None:
        print(f"   Article 1 embedding shape: {article_emb.shape}")
        print(f"   First 10 values: {article_emb[:10]}")


if __name__ == "__main__":
    example_usage()
