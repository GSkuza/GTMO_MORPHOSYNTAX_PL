#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
GTMØ Enhanced Quantum Metrics Module
====================================
Detailed quantum analysis with phase, amplitude, frequency, and entanglement.
"""

import numpy as np
import cmath
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Performance limits
MAX_QUANTUM_STATES = 50  # Limit to prevent O(n²) performance issues with very long texts


# =============================================================================
# QUANTUM STATE REPRESENTATION
# =============================================================================

@dataclass
class QuantumState:
    """Represents a quantum state in GTMØ space."""
    amplitude: complex  # Complex amplitude (magnitude + phase)
    coords: np.ndarray  # D-S-E coordinates
    word: str           # Associated word/token
    index: int          # Position in text


    def _classify_superposition(self, degree: float, phase_coherence: Optional[float] = None) -> str:
        """
        Classify type of superposition.

        Args:
            degree: Superposition degree (overlap between states)
            phase_coherence: Phase coherence (0-1), if available

        Returns:
            Classification string

        Logic:
            - Phase coherence > 0.8 → COHERENT (even if overlap is low)
            - Phase coherence < 0.3 → DECOHERENT (even if overlap is high)
            - Otherwise, use overlap-based classification
        """
        try:
            # Priority: Phase coherence determines coherence vs decoherence
            if phase_coherence is not None:
                if phase_coherence >= 0.8:
                    # High phase coherence = COHERENT state
                    if degree >= 0.7:
                        return 'MAXIMALLY_ENTANGLED_COHERENT'
                    else:
                        return 'COHERENT_SUPERPOSITION'
                elif phase_coherence <= 0.3:
                    # Low phase coherence = DECOHERENT
                    return 'DECOHERENT'

            # Fallback: use overlap-based classification
            if degree < 0.2:
                return 'DECOHERENT'
            elif degree < 0.4:
                return 'WEAKLY_SUPERPOSED'
            elif degree < 0.7:
                return 'COHERENTLY_SUPERPOSED'
            else:
                return 'MAXIMALLY_ENTANGLED'
        except TypeError:
            logger.error("TypeError in _classify_superposition")
            return 'DECOHERENT'


# =============================================================================
# INTEGRATION FUNCTION
# =============================================================================

def analyze_quantum_enhanced(text: str,
                            words: List[str],
                            coords_per_word: List[np.ndarray],
                            base_coherence: float = 0.5) -> Dict:
    """
    Perform enhanced quantum analysis on text.

    Args:
        text: Original text
        words: List of words
        coords_per_word: D-S-E coordinates for each word
        base_coherence: Base coherence value

    Returns:
        Enhanced quantum analysis results
    """
    if len(words) != len(coords_per_word):
        raise ValueError("Number of words must match number of coordinate sets")

    analyzer = EnhancedQuantumAnalyzer()

    # Create quantum states for all words (with performance limit)
    quantum_states = []

    # If too many words, sample evenly to stay within MAX_QUANTUM_STATES
    num_words = len(words)
    if num_words > MAX_QUANTUM_STATES:
        # Sample evenly across the text
        indices = np.linspace(0, num_words-1, MAX_QUANTUM_STATES, dtype=int)
        logger.info(f"Limiting quantum states: {num_words} words → {MAX_QUANTUM_STATES} states (sampled)")
        words_to_process = [(words[i], coords_per_word[i], i) for i in indices]
    else:
        words_to_process = [(w, c, i) for i, (w, c) in enumerate(zip(words, coords_per_word))]

    for word, coords, i in words_to_process:
        # Simple importance: longer words = higher importance
        importance = min(len(word) / 10.0, 1.0)

        state = analyzer.create_quantum_word_state(
            word=word,
            coords=coords,
            index=i,
            coherence=base_coherence,
            importance=importance
        )

        quantum_states.append(state)

    # Detailed coherence analysis
    coherence_detailed = analyzer.calculate_coherence_detailed(quantum_states)

    # Pairwise entanglement
    entanglement_analysis = analyzer.calculate_pairwise_entanglement(quantum_states)

    # Superposition analysis
    superposition_analysis = analyzer.analyze_superposition_states(quantum_states)

    # Extract phase and amplitude distributions
    phases = [cmath.phase(s.amplitude) for s in quantum_states]
    amplitudes = [abs(s.amplitude) for s in quantum_states]
    frequencies = [analyzer.calculate_frequency(s.coords) for s in quantum_states]

    # Quantum observables
    observables = {
        'mean_phase': float(np.mean(phases)),
        'std_phase': float(np.std(phases)),
        'mean_amplitude': float(np.mean(amplitudes)),
        'std_amplitude': float(np.std(amplitudes)),
        'mean_frequency': float(np.mean(frequencies)),
        'max_frequency': float(max(frequencies)),
        'min_frequency': float(min(frequencies))
    }

    # Wave function representation (first 5 states as example)
    wavefunction_samples = [
        {
            'word': s.word,
            'amplitude': abs(s.amplitude),
            'phase': cmath.phase(s.amplitude),
            'frequency': analyzer.calculate_frequency(s.coords),
            'coords': s.coords.tolist()
        }
        for s in quantum_states[:5]
    ]

    return {
        'num_quantum_states': len(quantum_states),
        'coherence_detailed': coherence_detailed,
        'entanglement': entanglement_analysis,
        'superposition': superposition_analysis,
        'quantum_observables': observables,
        'wavefunction_samples': wavefunction_samples,
        'quantum_classification': superposition_analysis['superposition_type']
    }


if __name__ == "__main__":
    print("GTMØ Enhanced Quantum Metrics Module")
    print("=" * 60)

    # Test quantum analysis
    analyzer = EnhancedQuantumAnalyzer()

    # Create test states
    test_words = ['prawo', 'ustawa', 'artykuł', 'konstytucja']
    test_coords = [
        np.array([0.85, 0.80, 0.15]),  # Legal terms - high D, low E
        np.array([0.88, 0.82, 0.12]),
        np.array([0.90, 0.85, 0.10]),
        np.array([0.95, 0.90, 0.05]),
    ]

    states = []
    for i, (word, coords) in enumerate(zip(test_words, test_coords)):
        state = analyzer.create_quantum_word_state(
            word=word,
            coords=coords,
            index=i,
            coherence=0.8,
            importance=1.0
        )
        states.append(state)
        print(f"\nWord: {word}")
        print(f"  Amplitude: {abs(state.amplitude):.3f}")
        print(f"  Phase: {cmath.phase(state.amplitude):.3f} rad")
        print(f"  Frequency: {analyzer.calculate_frequency(coords):.3f}")

    # Coherence analysis
    print("\n" + "=" * 60)
    coherence = analyzer.calculate_coherence_detailed(states)
    print("Coherence Analysis:")
    print(f"  Total coherence: {coherence['total_coherence']:.3f}")
    print(f"  Phase coherence: {coherence['phase_coherence']:.3f}")
    print(f"  Amplitude coherence: {coherence['amplitude_coherence']:.3f}")
    print(f"  Spatial coherence: {coherence['spatial_coherence']:.3f}")

    # Entanglement analysis
    print("\n" + "=" * 60)
    entanglement = analyzer.calculate_pairwise_entanglement(states)
    print("Entanglement Analysis:")
    print(f"  Mean entanglement: {entanglement['mean_entanglement']:.3f}")
    print(f"  Max entanglement: {entanglement['max_entanglement']:.3f}")
    print(f"  Highly entangled pairs: {len(entanglement['highly_entangled_pairs'])}")

    # Superposition analysis
    print("\n" + "=" * 60)
    superposition = analyzer.analyze_superposition_states(states)
    print("Superposition Analysis:")
    print(f"  Is superposed: {superposition['is_superposed']}")
    print(f"  Superposition degree: {superposition['superposition_degree']:.3f}")
    print(f"  Type: {superposition['superposition_type']}")

    print("\n" + "=" * 60)
