#!/usr/bin/env python3
"""
Extract quantum entanglements - ID-only version (without full word mapping).

Since the full 728-word mapping is not available in the JSON structure,
this version exports entanglements by quantum ID only.
"""

import json
import csv
import numpy as np
from pathlib import Path
from typing import List, Tuple
import sys


def load_entanglement_matrix(json_path: str) -> np.ndarray:
    """Load entanglement matrix from JSON."""
    print(f"Loading JSON from: {json_path}")
    with open(json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    qe = data['quantum_enhanced']
    matrix_data = qe['entanglement']['entanglement_matrix']
    matrix = np.array(matrix_data, dtype=np.float64)

    print(f"[OK] Loaded matrix of shape {matrix.shape}")
    return matrix, data


def get_available_word_samples(data: dict) -> dict:
    """Get whatever word samples are available from sentences."""
    word_map = {}

    for paragraph in data.get('paragraphs', []):
        for sentence in paragraph.get('sentences', []):
            qe = sentence.get('quantum_enhanced', {})
            samples = qe.get('wavefunction_samples', [])

            for idx, sample in enumerate(samples):
                # We don't know the global quantum ID, just collect what we can
                word = sample.get('word')
                if word:
                    word_map[word] = word_map.get(word, 0) + 1

    print(f"[INFO] Found {len(word_map)} unique words in samples")
    print(f"[INFO] Top 10 most frequent: {sorted(word_map.items(), key=lambda x: x[1], reverse=True)[:10]}")
    return word_map


def get_top_entanglements_by_id(matrix: np.ndarray, top_n: int = 100) -> List[Tuple[int, int, float]]:
    """Get top N entanglements by quantum ID only."""
    print(f"Computing top {top_n} entanglements...")

    n = matrix.shape[0]
    entanglements = []

    # Upper triangle only (avoid duplicates)
    for i in range(n):
        for j in range(i + 1, n):
            strength = matrix[i][j]
            if strength > 0.0:
                entanglements.append((i, j, strength))

    # Sort by strength descending
    entanglements.sort(key=lambda x: x[2], reverse=True)

    print(f"[OK] Found {len(entanglements)} non-zero entanglements")
    return entanglements[:top_n]


def export_to_csv(entanglements: List[Tuple], output_path: str):
    """Export to CSV."""
    print(f"Exporting to: {output_path}")

    with open(output_path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(['Rank', 'Quantum_ID_A', 'Quantum_ID_B', 'Entanglement_Strength'])

        for rank, (id_a, id_b, strength) in enumerate(entanglements, 1):
            writer.writerow([rank, id_a, id_b, f"{strength:.6f}"])

    print(f"[OK] Exported {len(entanglements)} entanglements")


def analyze_entanglement_patterns(entanglements: List[Tuple], matrix: np.ndarray):
    """Analyze patterns in entanglements."""
    print("\n" + "="*80)
    print("ENTANGLEMENT PATTERN ANALYSIS")
    print("="*80)

    strengths = [e[2] for e in entanglements[:100]]
    unique_strengths = sorted(set(strengths), reverse=True)

    print(f"\nUnique strength values (top 20): {unique_strengths[:20]}")
    print(f"\nStrength distribution:")
    print(f"  Max: {max(strengths):.6f}")
    print(f"  Min: {min(strengths):.6f}")
    print(f"  Mean: {np.mean(strengths):.6f}")
    print(f"  Std: {np.std(strengths):.6f}")

    # Most entangled IDs
    id_counts = {}
    for id_a, id_b, _ in entanglements[:100]:
        id_counts[id_a] = id_counts.get(id_a, 0) + 1
        id_counts[id_b] = id_counts.get(id_b, 0) + 1

    top_ids = sorted(id_counts.items(), key=lambda x: x[1], reverse=True)[:20]
    print(f"\nMost frequently entangled quantum IDs:")
    for qid, count in top_ids:
        print(f"  ID {qid}: {count} connections")

    print("="*80 + "\n")


def print_top_entanglements(entanglements: List[Tuple], top: int = 30):
    """Print top entanglements."""
    print("\n" + "="*80)
    print(f"TOP {top} QUANTUM ENTANGLEMENTS (by ID)")
    print("="*80)
    print(f"{'Rank':<6} {'ID_A':<10} {'ID_B':<10} {'Strength':<12}")
    print("-"*80)

    for rank, (id_a, id_b, strength) in enumerate(entanglements[:top], 1):
        print(f"{rank:<6} {id_a:<10} {id_b:<10} {strength:.6f}")

    print("="*80 + "\n")


def main():
    json_file = r"d:\GTMO_MORPHOSYNTAX\gtmo_results\analysis_15112025_no1_projekt-poselski_yyyy_(1)\article_004.json"
    output_csv = r"d:\GTMO_MORPHOSYNTAX\gtmo_results\analysis_15112025_no1_projekt-poselski_yyyy_(1)\quantum_entanglements_by_id.csv"

    # Load matrix
    matrix, data = load_entanglement_matrix(json_file)

    # Get available word samples (for info only)
    get_available_word_samples(data)

    # Get top entanglements
    entanglements = get_top_entanglements_by_id(matrix, top_n=500)

    # Export
    export_to_csv(entanglements, output_csv)

    # Analyze patterns
    analyze_entanglement_patterns(entanglements, matrix)

    # Print top
    print_top_entanglements(entanglements, top=50)

    print("[SUCCESS] Complete!")


if __name__ == "__main__":
    main()
