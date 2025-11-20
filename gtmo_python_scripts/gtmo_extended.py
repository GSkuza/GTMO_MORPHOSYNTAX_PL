#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
GTMØ Extended Analysis Module - Quantum & Temporal Extensions
=============================================================
Extends base GTMØ with temporality, rhetorical inversions, and quantum states.
Compatible with existing gtmo_morphosyntax.py
"""

import numpy as np
from typing import Dict, Tuple, List, Optional, Any
import logging
import re

# Try to import base module
try:
    from gtmo_morphosyntax import gtmo_analyze, CASE_COORDS, POS_COORDS
    BASE_MODULE_AVAILABLE = True
except ImportError:
    BASE_MODULE_AVAILABLE = False
    print("Warning: gtmo_morphosyntax.py not found, using standalone mode")

# Try to import NLP tools
try:
    import spacy
    nlp = spacy.load('pl_core_news_lg')
except:
    try:
        nlp = spacy.load('pl_core_news_sm')
    except:
        nlp = None

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# =============================================================================
# TEMPORALITY ANALYSIS
# =============================================================================

TEMPORAL_COORDS = {
    # Time -> [temporal_determination, temporal_stability, temporal_entropy]
    'past': np.array([0.85, 0.90, 0.15]),      # Past - high determination
    'present': np.array([0.70, 0.60, 0.40]),   # Present - medium determination
    'future': np.array([0.40, 0.30, 0.85]),    # Future - high entropy
    'conditional': np.array([0.35, 0.25, 0.90]), # Conditional - max entropy
    'imperative': np.array([0.95, 0.50, 0.20]), # Imperative - high intention determination
    'aorist': np.array([0.90, 0.95, 0.10]),    # Aorist - closed event
    'iterative': np.array([0.60, 0.80, 0.35])  # Iterative - cyclical
}

def analyze_temporality(text: str, doc=None) -> Tuple[np.ndarray, Dict]:
    """
    Analyze temporality of utterance.
    
    Args:
        text: Text to analyze
        doc: spaCy doc object (optional)
        
    Returns:
        Temporal coordinates and metadata
    """
    if doc is None and nlp is not None:
        doc = nlp(text)
    
    if doc is None:
        return np.array([0.5, 0.5, 0.5]), {"error": "No NLP model available"}
    
    temporal_markers = []
    temporal_metadata = {
        'tenses': {},
        'aspects': {},
        'dominant_time': None
    }
    
    for token in doc:
        if token.pos_ == 'VERB':
            morph = token.morph.to_dict()
            
            # Extract tense
            tense = morph.get('Tense', '')
            if tense:
                temporal_metadata['tenses'][tense] = temporal_metadata['tenses'].get(tense, 0) + 1
                
                if tense == 'Past':
                    temporal_markers.append(TEMPORAL_COORDS['past'])
                elif tense == 'Pres':
                    temporal_markers.append(TEMPORAL_COORDS['present'])
                elif tense == 'Fut':
                    temporal_markers.append(TEMPORAL_COORDS['future'])
            
            # Extract mood
            mood = morph.get('Mood', '')
            if mood == 'Cnd':
                temporal_markers.append(TEMPORAL_COORDS['conditional'])
            elif mood == 'Imp':
                temporal_markers.append(TEMPORAL_COORDS['imperative'])
            
            # Extract aspect
            aspect = morph.get('Aspect', '')
            if aspect:
                temporal_metadata['aspects'][aspect] = temporal_metadata['aspects'].get(aspect, 0) + 1
                
                # Perfective aspect increases stability
                if aspect == 'Perf' and temporal_markers:
                    temporal_markers[-1] = temporal_markers[-1] * np.array([1.0, 1.2, 0.8])
                    temporal_markers[-1] = np.clip(temporal_markers[-1], 0, 1)
    
    # Calculate dominant time
    if temporal_metadata['tenses']:
        temporal_metadata['dominant_time'] = max(temporal_metadata['tenses'], 
                                                key=temporal_metadata['tenses'].get)
    
    if temporal_markers:
        avg_coords = np.mean(temporal_markers, axis=0)
        return np.clip(avg_coords, 0, 1), temporal_metadata
    
    return np.array([0.5, 0.5, 0.5]), temporal_metadata


# =============================================================================
# RHETORICAL MODE DETECTION (IRONY/SARCASM/PARADOX)
# =============================================================================

RHETORICAL_PATTERNS = {
    'irony_markers': [
        'oczywiście', 'jakże', 'ależ', 'no pewnie', 'świetnie', 'super', 'bosko',
        'wspaniale', 'cudownie', 'genialnie', 'no tak', 'ach tak', 'nie do wiary', 'serio?',
    ],
    'paradox_markers': [
        'jednocześnie', 'zarazem', 'a jednak', 'mimo to',
        'wbrew', 'choć', 'aczkolwiek', 'niemniej', 'wszelako', 'pomimo', 'paradoksalnie'
    ],
    'sarcasm_patterns': [
        r'\b(super|świetny|genialny|wspaniały)\b.*\b(ale|tylko|że)\b',
        r'\".*\"',  # Quoted text often sarcastic
        r'\.{3,}',  # Multiple dots
        r'\?{2,}',  # Multiple question marks
        r'!{2,}'    # Multiple exclamation marks
    ],
    'negativity_context': [
        'znowu', 'znów', 'kolejny', 'kolejna', 'kolejne', 'jeszcze raz', 'bez przerwy', 'bez rezutatu'
        'po raz kolejny', 'jak zawsze', 'można było się tego spodziewać','jak zwykle', 'oczywiście',
        'niestety', 'szkoda', 'porażka', 'bez sensu', 'beznadziejnie','przykro', 'źle', 'kiepsko'
    ]
}

def detect_rhetorical_mode(text: str, base_coords: np.ndarray) -> Tuple[np.ndarray, str, Dict]:
    """
    Detect irony/sarcasm (inversion) or paradox (preservation).
    
    Args:
        text: Text to analyze
        base_coords: Base coordinates [D, S, E]
        
    Returns:
        Transformed coordinates, mode name, and metadata
    """
    text_lower = text.lower()
    metadata = {
        'irony_score': 0,
        'paradox_score': 0,
        'sarcasm_score': 0,
        'context_score': 0,
        'detected_markers': []
    }
    
    # Detect irony markers
    for marker in RHETORICAL_PATTERNS['irony_markers']:
        if marker in text_lower:
            metadata['irony_score'] += 1.5  # Increased weight
            metadata['detected_markers'].append(('irony', marker))
    
    # Detect sarcasm patterns
    for pattern in RHETORICAL_PATTERNS['sarcasm_patterns']:
        if re.search(pattern, text, re.IGNORECASE):
            metadata['sarcasm_score'] += 1.5
            metadata['detected_markers'].append(('sarcasm', pattern))
    
    # Check for negative context with positive words (strong irony indicator)
    has_positive = any(word in text_lower for word in [
        'super', 'świetnie', 'wspaniale', 'cudownie', 'genialnie', 
        'rewelacyjnie', 'fantastycznie', 'ekstra', 'bombowo'
    ])
    has_negative_context = any(word in text_lower for word in RHETORICAL_PATTERNS['negativity_context'])
    
    if has_positive and has_negative_context:
        metadata['context_score'] += 2
        metadata['detected_markers'].append(('context', 'positive_with_negative'))
        metadata['irony_score'] += 2  # Strong indicator of irony
    
    # Detect paradox markers
    for marker in RHETORICAL_PATTERNS['paradox_markers']:
        if marker in text_lower:
            metadata['paradox_score'] += 1
            metadata['detected_markers'].append(('paradox', marker))
    
    # Check for semantic contradictions
    if has_semantic_contradiction(text):
        metadata['irony_score'] += 2
        metadata['detected_markers'].append(('contradiction', 'semantic'))
    
    # Decision and transformation
    total_irony_sarcasm = metadata['irony_score'] + metadata['sarcasm_score'] + metadata['context_score']
    
    # Lower threshold for better detection
    if total_irony_sarcasm > 1.5:
        # INVERSION for irony/sarcasm
        inverted_coords = np.array([
            1.0 - base_coords[0],  # D: certainty -> uncertainty
            1.0 - base_coords[1],  # S: stability -> instability
            1.0 - base_coords[2]   # E: entropy -> neg-entropy
        ])
        return inverted_coords, 'irony', metadata
    
    elif metadata['paradox_score'] > 1:
        # PRESERVATION for paradox with entropy boost
        paradox_coords = base_coords.copy()
        paradox_coords[2] = min(paradox_coords[2] * 1.5, 1.0)  # Increase entropy
        return paradox_coords, 'paradox', metadata
    
    return base_coords, 'literal', metadata

def has_semantic_contradiction(text: str) -> bool:
    """Detect semantic contradiction suggesting irony."""
    contradictory_pairs = [
        (['świetnie', 'wspaniale', 'genialnie', 'super'], 
         ['źle', 'kiepsko', 'tragicznie', 'okropnie', 'strasznie']),
        (['kocham', 'uwielbiam'], ['nienawidzę', 'nie znoszę']),
        (['piękny', 'ładny', 'śliczny'], ['brzydki', 'szpetny', 'ohydny'])
    ]
    
    text_lower = text.lower()
    for positive_words, negative_words in contradictory_pairs:
        has_positive = any(word in text_lower for word in positive_words)
        has_negative = any(word in text_lower for word in negative_words)
        if has_positive and has_negative:
            return True
    
    return False


# =============================================================================
# QUANTUM COORDINATES
# =============================================================================

class QuantumCoordinates:
    """Quantum representation of GTMØ coordinates."""
    
    def __init__(self, superposition_threshold: float = 0.3):
        """
        Initialize quantum coordinates system.
        
        Args:
            superposition_threshold: Threshold for superposition activation
        """
        self.superposition_threshold = superposition_threshold
        self.ambiguity_indicators = [
            'może', 'chyba', 'prawdopodobnie', 'możliwe', 'ewentualnie',
            'lub', 'albo', 'czy', 'bądź', 'względnie',
            'zarówno', 'jak i', 'ani', 'niby', 'jakby',
            'przypuszczalnie', 'domniemanie', 'najprawdopodobniej'
        ]
    
    def create_superposition(self, interpretations: List[np.ndarray], 
                           probabilities: Optional[List[float]] = None) -> Dict:
        """
        Create superposition of states for ambiguous utterances.
        
        Args:
            interpretations: List of possible interpretations [D,S,E,T]
            probabilities: Probabilities of each interpretation
            
        Returns:
            Quantum representation of coordinates
        """
        if not interpretations:
            raise ValueError("At least one interpretation required")
        
        # Default equal probabilities if not provided
        if probabilities is None:
            probabilities = [1.0 / len(interpretations)] * len(interpretations)
        
        # Normalize probabilities
        probs = np.array(probabilities)
        probs = probs / probs.sum()
        
        # Base state (weighted average)
        base_state = np.average(interpretations, weights=probs, axis=0)
        
        # Covariance matrix - measures uncertainty/blur
        if len(interpretations) > 1:
            # Ensure interpretations is 2D array
            interp_array = np.array(interpretations)
            if interp_array.ndim == 1:
                interp_array = interp_array.reshape(1, -1)
            
            try:
                covariance = np.cov(interp_array.T, aweights=probs)
            except:
                # Fallback for single interpretation
                covariance = np.zeros((len(base_state), len(base_state)))
        else:
            covariance = np.zeros((len(base_state), len(base_state)))
        
        # Von Neumann entropy - measure of quantum uncertainty
        try:
            eigenvalues = np.linalg.eigvalsh(covariance)
            eigenvalues = eigenvalues[eigenvalues > 1e-10]
            if len(eigenvalues) > 0:
                eigenvalues_norm = eigenvalues / eigenvalues.sum()
                von_neumann_entropy = -np.sum(eigenvalues_norm * np.log(eigenvalues_norm + 1e-10))
            else:
                von_neumann_entropy = 0.0
        except:
            von_neumann_entropy = 0.0
        
        # Determine if in superposition
        uncertainty = np.sqrt(np.trace(covariance))
        is_superposed = uncertainty > self.superposition_threshold
        
        return {
            'base_state': base_state.tolist(),
            'superposition': is_superposed,
            'states': [state.tolist() for state in interpretations],
            'probabilities': probs.tolist(),
            'covariance': covariance.tolist(),
            'von_neumann_entropy': float(von_neumann_entropy),
            'uncertainty': float(uncertainty),
            'collapsed': False
        }
    
    def detect_ambiguity(self, text: str, variance_threshold: float = 0.1) -> Tuple[bool, Dict]:
        """
        Detect if text requires quantum representation.
        
        Args:
            text: Text to analyze
            variance_threshold: Threshold for variance-based detection
            
        Returns:
            Boolean and metadata about ambiguity
        """
        text_lower = text.lower()
        metadata = {
            'ambiguity_markers': [],
            'marker_count': 0,
            'question_marks': text.count('?'),
            'ellipsis': text.count('...')
        }
        
        # Check linguistic markers
        for marker in self.ambiguity_indicators:
            if marker in text_lower:
                metadata['ambiguity_markers'].append(marker)
                metadata['marker_count'] += 1
        
        # Additional ambiguity from punctuation
        ambiguity_score = metadata['marker_count'] + \
                         metadata['question_marks'] * 0.5 + \
                         metadata['ellipsis'] * 0.3
        
        needs_superposition = ambiguity_score > 2
        metadata['ambiguity_score'] = ambiguity_score
        metadata['needs_superposition'] = needs_superposition
        
        return needs_superposition, metadata
    
    def collapse_wavefunction(self, quantum_state: Dict, 
                            context: Optional[str] = None) -> np.ndarray:
        """
        Collapse wave function to specific state based on context.
        
        Args:
            quantum_state: Quantum state dictionary
            context: Additional context forcing collapse
            
        Returns:
            Collapsed coordinates
        """
        if quantum_state.get('collapsed', False):
            return np.array(quantum_state.get('final_state', quantum_state['base_state']))
        
        if not quantum_state.get('superposition', False):
            return np.array(quantum_state['base_state'])
        
        if context:
            # Context determines state selection
            # For now, use simple heuristic based on context sentiment
            context_lower = context.lower()
            
            # Positive context -> lower entropy state
            if any(word in context_lower for word in ['pewny', 'jasny', 'oczywisty']):
                # Find state with lowest entropy
                states = np.array(quantum_state['states'])
                if len(states[0]) >= 3:  # Has entropy dimension
                    chosen_idx = np.argmin(states[:, 2])
                else:
                    chosen_idx = 0
            else:
                # Random collapse according to probabilities
                chosen_idx = np.random.choice(
                    len(quantum_state['states']),
                    p=quantum_state['probabilities']
                )
        else:
            # Random collapse
            chosen_idx = np.random.choice(
                len(quantum_state['states']),
                p=quantum_state['probabilities']
            )
        
        chosen_state = np.array(quantum_state['states'][chosen_idx])
        
        # Mark as collapsed
        quantum_state['collapsed'] = True
        quantum_state['final_state'] = chosen_state.tolist()
        quantum_state['collapse_index'] = int(chosen_idx)
        
        return chosen_state


# =============================================================================
# MAIN EXTENDED ANALYSIS
# =============================================================================

def gtmo_analyze_extended(text: str, enable_quantum: bool = True) -> Dict:
    """
    Extended GTMØ analysis with temporality, inversions, and quantum states.
    
    Args:
        text: Text to analyze
        enable_quantum: Enable quantum superposition for ambiguous texts
        
    Returns:
        Extended analysis results
    """
    if not text or not text.strip():
        raise ValueError("Empty text provided")
    
    result = {
        'text': text,
        'mode': 'extended'
    }
    
    # Get base analysis if available
    if BASE_MODULE_AVAILABLE:
        try:
            base_result = gtmo_analyze(text)
            base_coords = np.array([
                base_result['coordinates']['determination'],
                base_result['coordinates']['stability'],
                base_result['coordinates']['entropy']
            ])
            result['base_analysis'] = base_result
        except Exception as e:
            logger.error(f"Base analysis failed: {e}")
            # Fallback coordinates
            base_coords = np.array([0.5, 0.5, 0.5])
            result['base_analysis'] = {'error': str(e)}
    else:
        # Standalone mode - use defaults
        base_coords = np.array([0.5, 0.5, 0.5])
        result['base_analysis'] = {'mode': 'standalone'}
    
    # Temporal analysis
    temporal_coords, temporal_meta = analyze_temporality(text)
    result['temporal_analysis'] = {
        'coordinates': temporal_coords.tolist(),
        'metadata': temporal_meta
    }
    
    # Rhetorical mode detection
    rhetorical_coords, rhetorical_mode, rhetorical_meta = detect_rhetorical_mode(text, base_coords)
    result['rhetorical_analysis'] = {
        'mode': rhetorical_mode,
        'coordinates': rhetorical_coords.tolist(),
        'metadata': rhetorical_meta
    }
    
    # Combine coordinates (4D)
    combined_coords = np.concatenate([
        rhetorical_coords,  # Use rhetorical-adjusted coords
        [temporal_coords[0]]  # Add temporal determination
    ])
    
    # Quantum analysis if enabled
    if enable_quantum:
        qc = QuantumCoordinates()
        needs_quantum, quantum_meta = qc.detect_ambiguity(text)
        
        if needs_quantum:
            # Generate alternative interpretations
            interpretations = generate_alternative_interpretations(text, combined_coords)
            quantum_state = qc.create_superposition(
                interpretations['states'],
                interpretations.get('probabilities')
            )
            
            result['quantum_analysis'] = {
                'is_superposed': True,
                'quantum_state': quantum_state,
                'metadata': quantum_meta
            }
            
            # Use base state from superposition
            final_coords = np.array(quantum_state['base_state'])
        else:
            result['quantum_analysis'] = {
                'is_superposed': False,
                'metadata': quantum_meta
            }
            final_coords = combined_coords
    else:
        final_coords = combined_coords
    
    # Final coordinates (ensure 4D)
    if len(final_coords) < 4:
        final_coords = np.concatenate([final_coords, [0.5] * (4 - len(final_coords))])
    
    result['coordinates'] = {
        'determination': float(final_coords[0]),
        'stability': float(final_coords[1]),
        'entropy': float(final_coords[2]),
        'temporality': float(final_coords[3]) if len(final_coords) > 3 else 0.5
    }
    
    return result


def generate_alternative_interpretations(text: str, base_coords: np.ndarray) -> Dict:
    """
    Generate alternative interpretations for ambiguous text.
    
    Args:
        text: Text to analyze
        base_coords: Base coordinates
        
    Returns:
        Dictionary with alternative states and probabilities
    """
    interpretations = [base_coords]
    probabilities = [0.5]  # Base interpretation has highest probability
    
    # Generate variations based on text characteristics
    text_lower = text.lower()
    
    # If question, add high-entropy interpretation
    if '?' in text:
        high_entropy = base_coords.copy()
        high_entropy[2] = min(high_entropy[2] * 1.5, 1.0)
        interpretations.append(high_entropy)
        probabilities.append(0.3)
    
    # If conditional markers, add low-determination interpretation
    if any(word in text_lower for word in ['może', 'chyba', 'gdyby']):
        low_determination = base_coords.copy()
        low_determination[0] *= 0.7
        interpretations.append(low_determination)
        probabilities.append(0.2)
    
    return {
        'states': interpretations,
        'probabilities': probabilities
    }


# =============================================================================
# BATCH PROCESSING
# =============================================================================

def batch_analyze_extended(texts: List[str], enable_quantum: bool = True) -> List[Dict]:
    """
    Analyze multiple texts with extended features.
    
    Args:
        texts: List of texts to analyze
        enable_quantum: Enable quantum analysis
        
    Returns:
        List of analysis results
    """
    results = []
    
    for i, text in enumerate(texts, 1):
        logger.info(f"Processing text {i}/{len(texts)}")
        try:
            result = gtmo_analyze_extended(text, enable_quantum)
            results.append(result)
        except Exception as e:
            logger.error(f"Failed to analyze text {i}: {e}")
            results.append({
                'text': text,
                'error': str(e)
            })
    
    return results


# =============================================================================
# TEST SUITE
# =============================================================================

def run_tests():
    """Run test suite for extended features."""
    test_cases = [
        # Temporal tests
        ("Wczoraj padało deszcz.", "past"),
        ("Jutro będzie lepiej.", "future"),
        ("Gdyby tylko wiedział...", "conditional"),
        
        # Irony tests
        ("Świetnie, znowu spóźniony autobus...", "irony"),
        ("Ależ cudownie się składa!", "irony"),
        
        # Paradox tests
        ("Kocham cię i jednocześnie nienawidzę.", "paradox"),
        ("Jest piękny, a jednak brzydki.", "paradox"),
        
        # Quantum ambiguity tests
        ("Może przyjdę, a może nie.", "quantum"),
        ("Chyba tak, ale nie jestem pewien.", "quantum"),
    ]
    
    print("=" * 60)
    print("GTMØ Extended Module - Test Suite")
    print("=" * 60)
    
    for text, expected_type in test_cases:
        print(f"\nTesting: {text}")
        print(f"Expected: {expected_type}")
        
        try:
            result = gtmo_analyze_extended(text)
            coords = result['coordinates']
            
            print(f"Coordinates: D={coords['determination']:.3f}, "
                  f"S={coords['stability']:.3f}, "
                  f"E={coords['entropy']:.3f}, "
                  f"T={coords['temporality']:.3f}")
            
            if 'rhetorical_analysis' in result:
                print(f"Rhetorical mode: {result['rhetorical_analysis']['mode']}")
            
            if 'quantum_analysis' in result and result['quantum_analysis']['is_superposed']:
                print(f"Quantum state: SUPERPOSED")
                print(f"Von Neumann entropy: {result['quantum_analysis']['quantum_state']['von_neumann_entropy']:.3f}")
            
        except Exception as e:
            print(f"ERROR: {e}")
    
    print("\n" + "=" * 60)
    print("Test suite completed")


def detect_rhetorical_anomaly(
    text: str,
    morph_coords: np.ndarray,
    syntax_coords: np.ndarray,
    morph_metadata: Dict
) -> Tuple[np.ndarray, str]:
    """
    Wykryj anomalie retoryczne przez analizę rozbieżności strukturalnych.
    """
    # 1. Analiza rozbieżności między morfologią a składnią
    divergence = np.linalg.norm(morph_coords - syntax_coords)

    # 2. Sprawdź dysproporcje w przypadkach
    cases = morph_metadata.get('cases', {})
    case_variance = np.var(list(cases.values())) if cases else 0.0
    # Wysoka wariancja = nienaturalny rozkład przypadków

    # 3. Sprawdź nadmiar przymiotników wartościujących
    pos = morph_metadata.get('pos', {})
    adj_ratio = pos.get('adj', 0) / max(sum(pos.values()), 1)

    # 4. Entropia morfologiczna vs składniowa
    entropy_mismatch = abs(morph_coords[2] - syntax_coords[2])

    # IRONIA: gdy struktura pozytywna (wysokie D, S) 
    # ale z anomaliami dystrybucyjnymi
    if (
        morph_coords[0] > 0.7 and  # Wysoka determinacja morfologiczna
        divergence > 0.15 and      # Ale rozbieżność ze składnią
        adj_ratio > 0.4            # Nadmiar przymiotników
    ):
        # Inwersja - struktura kłamie o intencji
        return 1.0 - morph_coords, 'ironic_anomaly'

    # PARADOKS: wysoka entropia przy wysokiej stabilności
    if morph_coords[1] > 0.7 and morph_coords[2] > 0.7:
        return morph_coords * np.array([0.8, 1.0, 1.2]), 'paradox'

    return morph_coords, 'literal'


if __name__ == "__main__":
    # Check dependencies
    print("GTMØ Extended Analysis Module")
    print("-" * 40)
    print(f"Base module available: {BASE_MODULE_AVAILABLE}")
    print(f"spaCy available: {nlp is not None}")
    print("-" * 40)
    
    # Run tests
    run_tests()
