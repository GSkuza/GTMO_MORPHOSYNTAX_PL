#!/usr/bin/env python3
"""
Extract and rank quantum word entanglements from GTMØ analysis JSON.

This script:
1. Loads large JSON files in chunks
2. Extracts word-to-ID mapping from wavefunction_samples
3. Parses the entanglement matrix
4. Generates TOP-N most entangled word pairs
5. Exports to CSV
"""

import json
import csv
import numpy as np
from pathlib import Path
from typing import List, Dict, Tuple, Optional
import sys


class QuantumEntanglementExtractor:
    """Extract quantum entanglements from GTMØ JSON analysis."""

    def __init__(self, json_path: str):
        """
        Initialize extractor.

        Args:
            json_path: Path to the JSON file with quantum analysis
        """
        self.json_path = Path(json_path)
        self.word_mapping: List[str] = []
        self.entanglement_matrix: Optional[np.ndarray] = None

    def load_json(self) -> dict:
        """
        Load entire JSON file.

        Returns:
            Parsed JSON data
        """
        print(f"Loading JSON from: {self.json_path}")
        with open(self.json_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        print(f"[OK] JSON loaded successfully")
        return data

    def extract_word_mapping(self, data: dict) -> List[str]:
        """
        Extract word list from wavefunction_samples across all sentences.

        Args:
            data: Parsed JSON data

        Returns:
            List of words in order (index = quantum ID)
        """
        print("Extracting word mapping...")

        words = []

        # Check if paragraphs exist
        if 'paragraphs' not in data:
            raise KeyError("No 'paragraphs' section found in JSON")

        # Iterate through paragraphs and sentences
        for para_idx, paragraph in enumerate(data['paragraphs']):
            if 'sentences' not in paragraph:
                continue

            for sent_idx, sentence in enumerate(paragraph['sentences']):
                if 'quantum_enhanced' not in sentence:
                    continue

                qe = sentence['quantum_enhanced']
                if 'wavefunction_samples' not in qe:
                    continue

                # Extract words from this sentence
                samples = qe['wavefunction_samples']
                sentence_words = [sample['word'] for sample in samples]
                words.extend(sentence_words)

        if len(words) == 0:
            raise ValueError("No quantum words found in any sentence")

        print(f"[OK] Extracted {len(words)} quantum words from sentences")
        return words

    def extract_entanglement_matrix(self, data: dict) -> np.ndarray:
        """
        Extract entanglement matrix.

        Args:
            data: Parsed JSON data

        Returns:
            NumPy array with entanglement values
        """
        print("Extracting entanglement matrix...")

        qe = data['quantum_enhanced']

        if 'entanglement' not in qe:
            raise KeyError("No 'entanglement' section found")

        if 'entanglement_matrix' not in qe['entanglement']:
            raise KeyError("No 'entanglement_matrix' found")

        matrix_data = qe['entanglement']['entanglement_matrix']
        matrix = np.array(matrix_data, dtype=np.float64)

        print(f"[OK] Extracted matrix of shape {matrix.shape}")
        return matrix

    def get_top_entanglements(self, top_n: int = 100) -> List[Tuple[str, str, float, int, int]]:
        """
        Get top N most entangled word pairs.

        Args:
            top_n: Number of top pairs to return

        Returns:
            List of tuples: (word_a, word_b, strength, id_a, id_b)
        """
        print(f"Computing top {top_n} entanglements...")

        if self.entanglement_matrix is None or len(self.word_mapping) == 0:
            raise ValueError("Must load data first")

        n = len(self.word_mapping)
        entanglements = []

        # Iterate upper triangle of symmetric matrix (avoid duplicates)
        for i in range(n):
            for j in range(i + 1, n):
                strength = self.entanglement_matrix[i][j]

                # Only include non-zero entanglements
                if strength > 0.0:
                    entanglements.append((
                        self.word_mapping[i],
                        self.word_mapping[j],
                        strength,
                        i,
                        j
                    ))

        # Sort by strength (descending)
        entanglements.sort(key=lambda x: x[2], reverse=True)

        print(f"[OK] Found {len(entanglements)} non-zero entanglements")
        return entanglements[:top_n]

    def export_to_csv(self, entanglements: List[Tuple], output_path: str):
        """
        Export entanglements to CSV file.

        Args:
            entanglements: List of entanglement tuples
            output_path: Path for output CSV
        """
        print(f"Exporting to CSV: {output_path}")

        with open(output_path, 'w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)

            # Header
            writer.writerow([
                'Rank',
                'Word_A',
                'Word_B',
                'Entanglement_Strength',
                'Quantum_ID_A',
                'Quantum_ID_B'
            ])

            # Data rows
            for rank, (word_a, word_b, strength, id_a, id_b) in enumerate(entanglements, 1):
                writer.writerow([rank, word_a, word_b, f"{strength:.6f}", id_a, id_b])

        print(f"[OK] Exported {len(entanglements)} entanglements to CSV")

    def print_summary(self, entanglements: List[Tuple], top: int = 10):
        """
        Print summary of top entanglements.

        Args:
            entanglements: List of entanglement tuples
            top: Number of top pairs to display
        """
        print(f"\n{'='*80}")
        print(f"TOP {top} QUANTUM ENTANGLEMENTS")
        print(f"{'='*80}")
        print(f"{'Rank':<6} {'Word A':<20} {'Word B':<20} {'Strength':<10}")
        print(f"{'-'*80}")

        for rank, (word_a, word_b, strength, id_a, id_b) in enumerate(entanglements[:top], 1):
            print(f"{rank:<6} {word_a:<20} {word_b:<20} {strength:.6f}")

        print(f"{'='*80}\n")

    def run(self, output_csv: str, top_n: int = 100, display_top: int = 20):
        """
        Run full extraction pipeline.

        Args:
            output_csv: Path for output CSV file
            top_n: Number of top entanglements to extract
            display_top: Number to display in console
        """
        try:
            # Load data
            data = self.load_json()

            # Extract word mapping
            self.word_mapping = self.extract_word_mapping(data)

            # Extract entanglement matrix
            self.entanglement_matrix = self.extract_entanglement_matrix(data)

            # Validate dimensions
            expected_size = len(self.word_mapping)
            actual_size = self.entanglement_matrix.shape[0]

            if actual_size != expected_size:
                print(f"[WARNING] Matrix size ({actual_size}) != word count ({expected_size})")

            # Get top entanglements
            entanglements = self.get_top_entanglements(top_n)

            # Export to CSV
            self.export_to_csv(entanglements, output_csv)

            # Print summary
            self.print_summary(entanglements, display_top)

            print(f"[SUCCESS] Extraction complete!")
            return entanglements

        except Exception as e:
            print(f"[ERROR] {type(e).__name__}: {e}")
            import traceback
            traceback.print_exc()
            sys.exit(1)


def main():
    """Main entry point."""

    # Configuration
    json_file = r"d:\GTMO_MORPHOSYNTAX\gtmo_results\analysis_15112025_no1_projekt-poselski_yyyy_(1)\article_004.json"
    output_csv = r"d:\GTMO_MORPHOSYNTAX\gtmo_results\analysis_15112025_no1_projekt-poselski_yyyy_(1)\quantum_entanglements_top100.csv"

    # Run extractor
    extractor = QuantumEntanglementExtractor(json_file)
    extractor.run(
        output_csv=output_csv,
        top_n=100,
        display_top=20
    )


if __name__ == "__main__":
    main()
