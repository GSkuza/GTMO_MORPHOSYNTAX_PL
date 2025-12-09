#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
HerBERT to GTMO 9D Mapper
=========================

This module provides integration between HerBERT (Polish BERT model) and the GTMO
9-dimensional morphosyntactic space. It enables semantic embeddings for Polish text
using the Allegro HerBERT model.

Requirements:
    - transformers
    - torch
    - numpy

Author: GTMO Project
License: MIT
"""

import numpy as np
from typing import List, Optional, Dict, Any
from concurrent.futures import ThreadPoolExecutor
import warnings

try:
    from transformers import AutoTokenizer, AutoModel
    import torch
    TRANSFORMERS_AVAILABLE = True
except ImportError:
    TRANSFORMERS_AVAILABLE = False
    warnings.warn(
        "transformers and/or torch not installed. "
        "Install with: pip install transformers torch"
    )


class GTMOCoordinates9D:
    """
    Represents 9-dimensional GTMO morphosyntactic coordinates.

    The 9 dimensions correspond to:
    1. Grammatical category (noun, verb, adjective, etc.)
    2. Case (for nominals)
    3. Number (singular, plural)
    4. Gender (masculine, feminine, neuter)
    5. Person (1st, 2nd, 3rd)
    6. Tense (for verbs)
    7. Aspect (perfective, imperfective)
    8. Voice (active, passive)
    9. Mood (indicative, imperative, conditional)
    """

    def __init__(self, coordinates: np.ndarray):
        """
        Initialize GTMO 9D coordinates.

        Args:
            coordinates: numpy array of shape (9,) with coordinate values
        """
        if coordinates.shape != (9,):
            raise ValueError(f"Expected 9D coordinates, got shape {coordinates.shape}")
        self.coordinates = coordinates

    def to_array(self) -> np.ndarray:
        """Return coordinates as numpy array."""
        return self.coordinates

    def __repr__(self) -> str:
        return f"GTMOCoordinates9D({self.coordinates})"


class HerBERTtoGTMO9DMapper:
    """
    Maps HerBERT embeddings to GTMO 9-dimensional morphosyntactic space.

    This class loads the HerBERT model and provides methods to:
    - Generate embeddings for Polish text
    - Extract semantic features
    - Compute similarity metrics
    """

    def __init__(
        self,
        herbert_model_name: str = "allegro/herbert-base-cased",
        device: Optional[str] = None,
        max_length: int = 512
    ):
        """
        Initialize the HerBERT to GTMO mapper.

        Args:
            herbert_model_name: Name of the HuggingFace HerBERT model
            device: Device to use ('cuda', 'cpu', or None for auto-detect)
            max_length: Maximum sequence length for tokenization

        Raises:
            RuntimeError: If transformers/torch are not installed
            Exception: If model loading fails
        """
        if not TRANSFORMERS_AVAILABLE:
            raise RuntimeError(
                "transformers and torch are required for HerBERT integration. "
                "Install with: pip install transformers torch"
            )

        # Set device
        if device is None:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device

        print(f"[*] Loading HerBERT model '{herbert_model_name}' on {self.device}...")

        try:
            # Load tokenizer and model
            self.tokenizer = AutoTokenizer.from_pretrained(herbert_model_name)
            self.model = AutoModel.from_pretrained(herbert_model_name)
            self.model.to(self.device)
            self.model.eval()  # Set to evaluation mode

            self.max_length = max_length
            self.embedding_dim = self.model.config.hidden_size

            print(f"[OK] HerBERT model loaded (embedding dim: {self.embedding_dim})")

        except Exception as e:
            raise Exception(f"Failed to load HerBERT model: {e}")

    def get_herbert_embedding(
        self,
        text: str,
        pooling: str = "mean"
    ) -> np.ndarray:
        """
        Generate HerBERT embedding for the given text.

        Args:
            text: Input text (Polish)
            pooling: Pooling strategy ('mean', 'cls', 'max')
                - 'mean': Average all token embeddings
                - 'cls': Use [CLS] token embedding
                - 'max': Max pooling over token embeddings

        Returns:
            numpy array of shape (embedding_dim,) containing the text embedding
        """
        if not text or not text.strip():
            # Return zero vector for empty text
            return np.zeros(self.embedding_dim, dtype=np.float32)

        try:
            # Tokenize
            inputs = self.tokenizer(
                text,
                return_tensors="pt",
                truncation=True,
                max_length=self.max_length,
                padding=True
            )

            # Move to device
            inputs = inputs.to(self.device)

            # Get embeddings (no gradient computation needed)
            with torch.no_grad():
                hidden_states = self.model(inputs["input_ids"], attention_mask=inputs["attention_mask"])[0]

            # Apply pooling
            if pooling == "cls":
                # Use [CLS] token (first token)
                embedding = hidden_states[:, 0, :].cpu().numpy()
            elif pooling == "max":
                # Max pooling over sequence
                embedding = torch.max(hidden_states, dim=1)[0].cpu().numpy()
            else:  # mean pooling (default)
                # Average over sequence length
                embedding = torch.mean(hidden_states, dim=1).cpu().numpy()

            return embedding.astype(np.float32)

        except Exception as e:
            warnings.warn(f"Error generating HerBERT embedding: {e}")
            return np.zeros(self.embedding_dim, dtype=np.float32)

    def get_batch_embeddings(
        self,
        texts: List[str],
        pooling: str = "mean",
        batch_size: int = 8
    ) -> np.ndarray:
        """
        Generate HerBERT embeddings for multiple texts efficiently.

        Args:
            texts: List of input texts
            pooling: Pooling strategy (see get_herbert_embedding)
            batch_size: Number of texts to process at once

        Returns:
            numpy array of shape (len(texts), embedding_dim)
        """
        embeddings = np.zeros((len(texts), self.embedding_dim), dtype=np.float32)

        for i in range(0, len(texts), batch_size):
            batch_texts = texts[i:i + batch_size]
            inputs = self.tokenizer(
                batch_texts,
                return_tensors="pt",
                truncation=True,
                max_length=self.max_length,
                padding=True
            )

            inputs = inputs.to(self.device)

            with torch.no_grad():
                hidden_states = self.model(inputs["input_ids"], attention_mask=inputs["attention_mask"])[0]

            if pooling == "cls":
                batch_embeddings = hidden_states[:, 0, :].cpu().numpy()
            elif pooling == "max":
                batch_embeddings = torch.max(hidden_states, dim=1)[0].cpu().numpy()
            else:  # mean pooling (default)
                batch_embeddings = torch.mean(hidden_states, dim=1).cpu().numpy()

            embeddings[i:i + batch_size] = batch_embeddings

        return embeddings

    def compute_similarity(
        self,
        text1: str,
        text2: str,
        metric: str = "cosine"
    ) -> float:
        """
        Compute similarity between two texts using their HerBERT embeddings.

        Args:
            text1: First text
            text2: Second text
            metric: Similarity metric ('cosine', 'euclidean', 'dot')

        Returns:
            Similarity score (higher = more similar for cosine/dot)
        """
        # Get embeddings in parallel
        with ThreadPoolExecutor() as executor:
            emb1 = executor.submit(self.get_herbert_embedding, text1).result()
            emb2 = executor.submit(self.get_herbert_embedding, text2).result()

        # Flatten embeddings if needed
        emb1 = emb1.flatten()
        emb2 = emb2.flatten()

        if metric == "cosine":
            # Cosine similarity
            norm1, norm2 = np.linalg.norm(emb1), np.linalg.norm(emb2)
            if norm1 == 0 or norm2 == 0:
                return 0.0
            return float(np.dot(emb1, emb2) / (norm1 * norm2))

        elif metric == "euclidean":
            # Euclidean distance (lower = more similar)
            return float(np.linalg.norm(emb1 - emb2))

        elif metric == "dot":
            # Dot product
            return float(np.dot(emb1, emb2))

        else:
            raise ValueError(f"Unknown metric: {metric}")

    def get_embedding_stats(self, embedding: np.ndarray) -> Dict[str, float]:
        """
        Compute statistics for an embedding vector.

        Args:
            embedding: numpy array embedding vector

        Returns:
            Dictionary with statistics (magnitude, mean, std, etc.)
        """
        # Flatten embedding if needed
        emb = embedding.flatten()
        stats = {
            "magnitude": float(np.linalg.norm(emb)),
            "mean": float(np.mean(emb)),
            "std": float(np.std(emb)),
            "min": float(np.min(emb)),
            "max": float(np.max(emb)),
            "sparsity": float(np.count_nonzero(np.abs(emb) < 1e-6) / len(emb))
        }
        return stats


# Convenience function for quick embedding generation
def get_embedding(text: str, model_name: str = "allegro/herbert-base-cased") -> np.ndarray:
    """
    Quick function to get HerBERT embedding for text.
    Creates a new mapper instance - use HerBERTtoGTMO9DMapper directly for batch processing.

    Args:
        text: Input text
        model_name: HuggingFace model name

    Returns:
        numpy array embedding
    """
    mapper = HerBERTtoGTMO9DMapper(herbert_model_name=model_name)
    return mapper.get_herbert_embedding(text)


if __name__ == "__main__":
    # Example usage
    print("=" * 80)
    print("HerBERT to GTMO 9D Mapper - Example Usage")
    print("=" * 80)

    # Test if dependencies are available
    if not TRANSFORMERS_AVAILABLE:
        print("\n[ERROR] Missing dependencies!")
        print("Install with: pip install transformers torch")
        exit(1)

    # Create mapper
    print("\n1. Initializing HerBERT mapper...")
    mapper = HerBERTtoGTMO9DMapper()

    # Test texts (Polish)
    texts = [
        "Konstytucja Rzeczypospolitej Polskiej jest najwyższym prawem.",
        "Sejm i Senat sprawują władzę ustawodawczą.",
        "Prezydent jest najwyższym przedstawicielem państwa."
    ]

    print("\n2. Generating embeddings for sample texts...")
    for i, text in enumerate(texts, 1):
        emb = mapper.get_herbert_embedding(text)
        stats = mapper.get_embedding_stats(emb)
        print(f"\nText {i}: {text}")
        print(f"  Embedding shape: {emb.shape}")
        print(f"  Magnitude: {stats['magnitude']:.3f}")
        print(f"  Mean: {stats['mean']:.6f}")

    print("\n3. Computing pairwise similarities...")
    for i in range(len(texts)):
        for j in range(i + 1, len(texts)):
            sim = mapper.compute_similarity(texts[i], texts[j])
            print(f"  Text {i+1} <-> Text {j+1}: {sim:.3f}")

    print("\n[OK] Example completed successfully!")
