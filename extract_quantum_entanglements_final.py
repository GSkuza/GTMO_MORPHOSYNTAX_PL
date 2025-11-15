#!/usr/bin/env python3
"""
Final version: Extract quantum word entanglements from GTMØ analysis.

This version:
1. Aggregates ALL quantum words from all sentences
2. Maps them to quantum IDs by position
3. Generates ranked entanglements with actual words
4. Exports to CSV and Excel
"""

import json
import csv
import numpy as np
import pandas as pd
from pathlib import Path
from typing import List, Dict, Tuple
from collections import Counter
import sys


class QuantumEntanglementAnalyzer:
    """Analyze quantum word entanglements from GTMØ JSON."""

    def __init__(self, json_path: str):
        self.json_path = Path(json_path)
        self.data = None
        self.all_quantum_words = []  # Sequential list of all quantum words
        self.word_to_ids = {}  # word -> list of quantum IDs
        self.entanglement_matrix = None

    def load_data(self):
        """Load JSON data."""
        print(f"Loading: {self.json_path}")
        with open(self.json_path, 'r', encoding='utf-8') as f:
            self.data = json.load(f)
        print("[OK] Loaded")

    def extract_all_quantum_words(self):
        """
        Extract ALL quantum words by iterating through all sentences.
        Build a complete list in order.
        """
        print("Extracting quantum words from all sentences...")

        quantum_id = 0

        for para in self.data.get('paragraphs', []):
            for sent in para.get('sentences', []):
                qe = sent.get('quantum_enhanced', {})
                samples = qe.get('wavefunction_samples', [])

                for sample in samples:
                    word = sample.get('word')
                    if word:
                        self.all_quantum_words.append({
                            'id': quantum_id,
                            'word': word,
                            'amplitude': sample.get('amplitude'),
                            'phase': sample.get('phase'),
                            'coords': sample.get('coords')
                        })

                        # Build reverse mapping
                        if word not in self.word_to_ids:
                            self.word_to_ids[word] = []
                        self.word_to_ids[word].append(quantum_id)

                        quantum_id += 1

        print(f"[OK] Extracted {len(self.all_quantum_words)} quantum words")
        print(f"[INFO] Unique words: {len(self.word_to_ids)}")

        # Show most common words
        word_counts = Counter([w['word'] for w in self.all_quantum_words])
        print(f"[INFO] Top 10 common words: {word_counts.most_common(10)}")

    def load_entanglement_matrix(self):
        """Load entanglement matrix from root level."""
        print("Loading entanglement matrix...")

        qe = self.data.get('quantum_enhanced', {})
        ent = qe.get('entanglement', {})
        matrix_data = ent.get('entanglement_matrix', [])

        if not matrix_data:
            raise ValueError("No entanglement_matrix found")

        self.entanglement_matrix = np.array(matrix_data, dtype=np.float64)
        print(f"[OK] Matrix shape: {self.entanglement_matrix.shape}")

    def get_top_entanglements(self, top_n: int = 100) -> List[Dict]:
        """
        Get top N most entangled word pairs.

        Returns list of dicts with full information.
        """
        print(f"Computing top {top_n} entanglements...")

        n_words = len(self.all_quantum_words)
        n_matrix = self.entanglement_matrix.shape[0]

        if n_matrix != n_words:
            print(f"[WARNING] Matrix size ({n_matrix}) != extracted words ({n_words})")
            print(f"[INFO] Using min({n_matrix}, {n_words}) for safety")
            n = min(n_matrix, n_words)
        else:
            n = n_words

        entanglements = []

        # Upper triangle only
        for i in range(n):
            for j in range(i + 1, n):
                strength = self.entanglement_matrix[i, j]

                if strength > 0.0:
                    word_a = self.all_quantum_words[i]['word'] if i < n_words else f"ID_{i}"
                    word_b = self.all_quantum_words[j]['word'] if j < n_words else f"ID_{j}"

                    entanglements.append({
                        'id_a': i,
                        'id_b': j,
                        'word_a': word_a,
                        'word_b': word_b,
                        'strength': strength
                    })

        # Sort by strength
        entanglements.sort(key=lambda x: x['strength'], reverse=True)

        print(f"[OK] Found {len(entanglements)} non-zero entanglements")

        # Analyze unique strengths
        unique_strengths = sorted(set(e['strength'] for e in entanglements), reverse=True)
        print(f"[INFO] Unique strength values: {len(unique_strengths)}")
        if len(unique_strengths) <= 20:
            print(f"[INFO] All unique values: {unique_strengths}")
        else:
            print(f"[INFO] Top 20: {unique_strengths[:20]}")

        return entanglements[:top_n]

    def export_csv(self, entanglements: List[Dict], output_path: str):
        """Export to CSV."""
        print(f"Exporting CSV: {output_path}")

        with open(output_path, 'w', newline='', encoding='utf-8') as f:
            writer = csv.DictWriter(f, fieldnames=[
                'Rank', 'Word_A', 'Word_B', 'Entanglement_Strength',
                'Quantum_ID_A', 'Quantum_ID_B'
            ])
            writer.writeheader()

            for rank, ent in enumerate(entanglements, 1):
                writer.writerow({
                    'Rank': rank,
                    'Word_A': ent['word_a'],
                    'Word_B': ent['word_b'],
                    'Entanglement_Strength': f"{ent['strength']:.6f}",
                    'Quantum_ID_A': ent['id_a'],
                    'Quantum_ID_B': ent['id_b']
                })

        print(f"[OK] Exported {len(entanglements)} rows")

    def export_excel(self, entanglements: List[Dict], output_path: str):
        """Export to Excel with formatting."""
        print(f"Exporting Excel: {output_path}")

        df = pd.DataFrame([
            {
                'Rank': rank,
                'Word A': ent['word_a'],
                'Word B': ent['word_b'],
                'Entanglement': round(ent['strength'], 6),
                'ID_A': ent['id_a'],
                'ID_B': ent['id_b']
            }
            for rank, ent in enumerate(entanglements, 1)
        ])

        with pd.ExcelWriter(output_path, engine='openpyxl') as writer:
            df.to_excel(writer, sheet_name='Entanglements', index=False)

        print(f"[OK] Exported Excel")

    def print_summary(self, entanglements: List[Dict], top: int = 30):
        """Print summary table."""
        print("\n" + "="*90)
        print(f"TOP {top} QUANTUM ENTANGLEMENTS")
        print("="*90)
        print(f"{'Rank':<6} {'Word A':<25} {'Word B':<25} {'Strength':<15} {'IDs':<15}")
        print("-"*90)

        for rank, ent in enumerate(entanglements[:top], 1):
            ids_str = f"{ent['id_a']},{ent['id_b']}"
            print(f"{rank:<6} {ent['word_a']:<25} {ent['word_b']:<25} "
                  f"{ent['strength']:<15.6f} {ids_str:<15}")

        print("="*90 + "\n")

    def analyze_word_centrality(self, entanglements: List[Dict], top: int = 20):
        """
        Analyze which words are most central (most entangled with others).
        """
        print("\nWORD CENTRALITY ANALYSIS")
        print("="*70)

        word_connections = Counter()

        for ent in entanglements:
            word_connections[ent['word_a']] += 1
            word_connections[ent['word_b']] += 1

        print(f"\nTop {top} most connected words:")
        print(f"{'Rank':<6} {'Word':<30} {'Connections':<12}")
        print("-"*70)

        for rank, (word, count) in enumerate(word_connections.most_common(top), 1):
            print(f"{rank:<6} {word:<30} {count:<12}")

        print("="*70 + "\n")

    def run(self, output_csv: str, output_excel: str, top_n: int = 100):
        """Run full analysis pipeline."""
        try:
            self.load_data()
            self.extract_all_quantum_words()
            self.load_entanglement_matrix()

            entanglements = self.get_top_entanglements(top_n)

            self.export_csv(entanglements, output_csv)
            self.export_excel(entanglements, output_excel)

            self.print_summary(entanglements, top=30)
            self.analyze_word_centrality(entanglements, top=20)

            print("[SUCCESS] Analysis complete!")
            return entanglements

        except Exception as e:
            print(f"[ERROR] {type(e).__name__}: {e}")
            import traceback
            traceback.print_exc()
            sys.exit(1)


def main():
    json_file = r"d:\GTMO_MORPHOSYNTAX\gtmo_results\analysis_15112025_no1_projekt-poselski_yyyy_(1)\article_004.json"
    output_csv = r"d:\GTMO_MORPHOSYNTAX\gtmo_results\analysis_15112025_no1_projekt-poselski_yyyy_(1)\quantum_entanglements_final.csv"
    output_excel = r"d:\GTMO_MORPHOSYNTAX\gtmo_results\analysis_15112025_no1_projekt-poselski_yyyy_(1)\quantum_entanglements_final.xlsx"

    analyzer = QuantumEntanglementAnalyzer(json_file)
    analyzer.run(
        output_csv=output_csv,
        output_excel=output_excel,
        top_n=100
    )


if __name__ == "__main__":
    main()
