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

@dataclass(frozen=True)
class QuantumState:
    """Represents a quantum state in GTMØ space."""
    amplitude: complex  # Complex amplitude (magnitude + phase)
    coords: np.ndarray  # D-S-E coordinates
    word: str           # Associated word/token
    index: int          # Position in text


class EnhancedQuantumAnalyzer:
    """Enhanced quantum analysis with detailed metrics."""

    def __init__(self):
        """Initialize enhanced quantum analyzer."""
        self.planck_h = 6.62607015e-34  # Planck constant (symbolic)
        self.hbar = self.planck_h / (2 * np.pi)

    def calculate_phase(self, coords: np.ndarray, word: str) -> float:
        """
        Calculate quantum phase based on semantic position.

        Args:
            coords: D-S-E coordinates
            word: Word being analyzed

        Returns:
            Phase angle in radians [0, 2π]
        """
        # Phase depends on position in semantic space
        # D influences azimuthal angle, E influences polar angle
        d, s, e = coords

        # Map to spherical coordinates
        # θ (polar) from E: 0 (low entropy) to π (high entropy)
        theta = e * np.pi

        # φ (azimuthal) from D: 0 to 2π
        phi = d * 2 * np.pi

        # Combined phase with stability modulation
        phase = (phi + theta * s) % (2 * np.pi)

        return float(phase)

    def calculate_amplitude(self,
                          coords: np.ndarray,
                          coherence: float,
                          word_importance: float = 1.0) -> float:
        """
        Calculate quantum amplitude (probability amplitude).

        Args:
            coords: D-S-E coordinates
            coherence: Coherence measure
            word_importance: Importance weight [0, 1]

        Returns:
            Amplitude value [0, 1]
        """
        d, s, e = coords

        # Amplitude combines determination and stability
        # High D and S = high amplitude
        base_amplitude = np.sqrt(d * s) * (1 - 0.5 * e)

        # Modulate by coherence
        amplitude = base_amplitude * np.sqrt(coherence) * word_importance

        return float(np.clip(amplitude, 0.0, 1.0))

    def calculate_frequency(self, coords: np.ndarray, velocity: Optional[np.ndarray] = None) -> float:
        """
        Calculate semantic frequency (de Broglie-like).

        Args:
            coords: D-S-E coordinates
            velocity: Velocity in semantic space (optional)

        Returns:
            Frequency in arbitrary units
        """
        d, s, e = coords

        # Base frequency from entropy (high entropy = high frequency)
        base_freq = e * 10.0  # Scale to reasonable range

        # If velocity available, add Doppler-like shift
        if velocity is not None:
            velocity_mag = np.linalg.norm(velocity)
            # Positive velocity increases frequency
            freq = base_freq * (1.0 + 0.1 * velocity_mag)
        else:
            freq = base_freq

        return float(freq)

    def create_quantum_word_state(self,
                                  word: str,
                                  coords: np.ndarray,
                                  index: int,
                                  coherence: float,
                                  importance: float = 1.0) -> QuantumState:
        """
        Create quantum state representation for a word.

        Args:
            word: The word
            coords: D-S-E coordinates
            index: Position in text
            coherence: Coherence value
            importance: Word importance

        Returns:
            QuantumState object
        """
        phase = self.calculate_phase(coords, word)
        amplitude_mag = self.calculate_amplitude(coords, coherence, importance)

        # Create complex amplitude: A = |A| * e^(iφ)
        amplitude = amplitude_mag * cmath.exp(1j * phase)

        return QuantumState(
            amplitude=amplitude,
            coords=coords,
            word=word,
            index=index
        )

    def calculate_entanglement_entropy(self,
                                      state1: QuantumState,
                                      state2: QuantumState) -> float:
        """
        Calculate entanglement entropy between two quantum states.

        Args:
            state1: First quantum state
            state2: Second quantum state

        Returns:
            Entanglement entropy [0, ∞]
        """
        # Distance in semantic space
        distance = np.linalg.norm(state1.coords - state2.coords)

        # Phase difference
        phase_diff = abs(cmath.phase(state1.amplitude) - cmath.phase(state2.amplitude))
        phase_diff = min(phase_diff, 2*np.pi - phase_diff)  # Wrap to [0, π]

        # Amplitude correlation
        amp1 = abs(state1.amplitude)
        amp2 = abs(state2.amplitude)
        amp_product = amp1 * amp2

        # Entanglement decreases with distance, increases with amplitude correlation
        # and phase alignment
        if distance > 0:
            entanglement = amp_product * (1 - phase_diff / np.pi) / (1 + distance)
        else:
            entanglement = amp_product

        # Convert to entropy (von Neumann-like)
        if entanglement > 0 and entanglement < 1:
            entropy = -entanglement * np.log(entanglement) - (1-entanglement) * np.log(1-entanglement + 1e-10)
        else:
            entropy = 0.0

        return float(np.clip(entropy, 0.0, 10.0))

    def calculate_pairwise_entanglement(self, states: List[QuantumState]) -> Dict:
        """
        Calculate pairwise entanglement between all quantum states.

        Args:
            states: List of quantum states

        Returns:
            Entanglement matrix and statistics
        """
        n = len(states)

        if n < 2:
            return {
                'entanglement_matrix': [],
                'mean_entanglement': 0.0,
                'max_entanglement': 0.0,
                'total_entanglement': 0.0,
                'num_states': n
            }

        # Calculate entanglement matrix
        entanglement_matrix = np.zeros((n, n))

        for i in range(n):
            for j in range(i+1, n):
                entropy = self.calculate_entanglement_entropy(states[i], states[j])
                entanglement_matrix[i, j] = entropy
                entanglement_matrix[j, i] = entropy  # Symmetric

        # Statistics
        upper_triangle = entanglement_matrix[np.triu_indices(n, k=1)]

        return {
            'entanglement_matrix': entanglement_matrix.tolist(),
            'mean_entanglement': float(upper_triangle.mean()) if len(upper_triangle) > 0 else 0.0,
            'max_entanglement': float(upper_triangle.max()) if len(upper_triangle) > 0 else 0.0,
            'total_entanglement': float(upper_triangle.sum()),
            'num_states': n,
            'highly_entangled_pairs': self._find_highly_entangled_pairs(states, entanglement_matrix)
        }

    def _find_highly_entangled_pairs(self,
                                    states: List[QuantumState],
                                    entanglement_matrix: np.ndarray,
                                    threshold: float = 0.5) -> List[Dict]:
        """
        Find highly entangled word pairs.

        Args:
            states: List of quantum states
            entanglement_matrix: Entanglement matrix
            threshold: Threshold for "high" entanglement

        Returns:
            List of highly entangled pairs
        """
        pairs = []
        n = len(states)

        for i in range(n):
            for j in range(i+1, n):
                entanglement = entanglement_matrix[i, j]

                if entanglement > threshold:
                    pairs.append({
                        'word1': states[i].word,
                        'word2': states[j].word,
                        'index1': states[i].index,
                        'index2': states[j].index,
                        'entanglement': float(entanglement),
                        'distance': int(abs(states[j].index - states[i].index))
                    })

        # Sort by entanglement (descending)
        pairs.sort(key=lambda x: x['entanglement'], reverse=True)

        return pairs[:10]  # Return top 10

    def calculate_coherence_detailed(self, states: List[QuantumState]) -> Dict:
        """
        Calculate detailed coherence metrics.

        Args:
            states: List of quantum states

        Returns:
            Detailed coherence analysis
        """
        if not states:
            return {
                'total_coherence': 0.0,
                'phase_coherence': 0.0,
                'amplitude_coherence': 0.0,
                'spatial_coherence': 0.0
            }

        # Extract phases and amplitudes
        phases = np.array([cmath.phase(s.amplitude) for s in states])
        amplitudes = np.array([abs(s.amplitude) for s in states])
        coords_array = np.array([s.coords for s in states])

        # Phase coherence: how aligned are the phases?
        phase_coherence = np.abs(np.mean(np.exp(1j * phases)))

        # Amplitude coherence: how uniform are the amplitudes?
        amplitude_coherence = np.exp(-np.var(amplitudes))

        # Spatial coherence: how clustered are the states in D-S-E space?
        if len(coords_array) > 1:
            # Calculate variance of the coordinates
            spatial_variance = np.var(coords_array, axis=0)
            # Calculate spatial coherence
            spatial_coherence = np.exp(-np.sum(spatial_variance))
        else:
            spatial_coherence = 1.0  # Single state = perfect coherence

        # Total coherence: weighted average
        total_coherence = (
            0.4 * phase_coherence +
            0.3 * amplitude_coherence +
            0.3 * spatial_coherence
        )

        return {
            'total_coherence': float(total_coherence),
            'phase_coherence': float(phase_coherence),
            'amplitude_coherence': float(amplitude_coherence),
            'spatial_coherence': float(spatial_coherence)
        }

    def analyze_superposition_states(self, states: List[QuantumState]) -> Dict:
        """
        Analyze superposition characteristics of quantum states.

        Args:
            states: List of quantum states

        Returns:
            Dictionary with superposition analysis
        """
        if not states:
            return {
                'superposition_type': 'none',
                'superposition_degree': 0.0,
                'dominant_state_index': -1,
                'state_probabilities': []
            }

        # Calculate probabilities from amplitudes
        amplitudes = np.array([abs(s.amplitude) for s in states])
        probabilities = amplitudes ** 2
        probabilities = probabilities / np.sum(probabilities) if np.sum(probabilities) > 0 else probabilities

        # Superposition degree: entropy-based measure
        # High entropy = strong superposition, low entropy = collapsed state
        entropy = -np.sum(probabilities * np.log2(probabilities + 1e-10))
        max_entropy = np.log2(len(states)) if len(states) > 1 else 1.0
        superposition_degree = entropy / max_entropy if max_entropy > 0 else 0.0

        # Dominant state
        dominant_idx = int(np.argmax(probabilities))
        dominant_prob = probabilities[dominant_idx]

        # Classification
        if dominant_prob > 0.9:
            superposition_type = 'collapsed'
        elif superposition_degree > 0.8:
            superposition_type = 'strong_superposition'
        elif superposition_degree > 0.5:
            superposition_type = 'moderate_superposition'
        else:
            superposition_type = 'weak_superposition'

        return {
            'superposition_type': superposition_type,
            'superposition_degree': float(superposition_degree),
            'dominant_state_index': dominant_idx,
            'dominant_probability': float(dominant_prob),
            'state_probabilities': probabilities.tolist(),
            'entropy': float(entropy)
        }

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
    import sys
    sys.stdout.reconfigure(encoding='utf-8')
    print("GTMO Enhanced Quantum Metrics Module")
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
    print(f"  Superposition type: {superposition['superposition_type']}")
    print(f"  Superposition degree: {superposition['superposition_degree']:.3f}")
    print(f"  Dominant state index: {superposition['dominant_state_index']}")
    print(f"  Dominant probability: {superposition['dominant_probability']:.3f}")
    print(f"  Entropy: {superposition['entropy']:.3f}")

    print("\n" + "=" * 60)
