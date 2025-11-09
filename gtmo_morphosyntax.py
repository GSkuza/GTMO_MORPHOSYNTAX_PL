#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
GTMØ QUANTUM MORPHOSYNTAX ENGINE - INTEGRATED VERSION WITH AXIOMS
=================================================================
Integration of Polish morphosyntax analysis with quantum superposition semantics
and complete 13 axiom system acting as immune system.
"""

import sys
import numpy as np
from typing import Dict, Tuple, List, Optional, Union, Any
from dataclasses import dataclass
import cmath
from enum import Enum
import logging
import json
import hashlib
from datetime import datetime

# Fix Windows console encoding for Unicode characters (with error handling)
if sys.platform == 'win32':
    import io
    try:
        if not sys.stdout.closed:
            sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
        if not sys.stderr.closed:
            sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8')
    except (ValueError, AttributeError):
        # Skip if streams are already wrapped or closed
        pass

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Import rhetorical analysis module
try:
    from gtmo_pure_rhetoric import GTMORhetoricalAnalyzer
    RHETORICAL_ANALYZER_AVAILABLE = True
    # print("✔ GTMØ Rhetorical Analyzer loaded")
except ImportError:
    RHETORICAL_ANALYZER_AVAILABLE = False
    # print("✗ GTMØ Rhetorical Analyzer not available")

# Import domain dictionary module
try:
    from gtmo_domain_dictionary import DomainDictionary
    DOMAIN_DICTIONARY_AVAILABLE = True
    # print("✔ GTMØ Domain Dictionary loaded")
except ImportError:
    DOMAIN_DICTIONARY_AVAILABLE = False
    # print("✗ GTMØ Domain Dictionary not available")

# Import constitutional duality calculator
try:
    from gtmo_constitutional_duality import ConstitutionalDualityCalculator
    CONSTITUTIONAL_DUALITY_AVAILABLE = True
    # print("✔ GTMØ Constitutional Duality Calculator loaded")
except ImportError:
    CONSTITUTIONAL_DUALITY_AVAILABLE = False
    # print("✗ GTMØ Constitutional Duality Calculator not available")

# Required imports
try:
    import morfeusz2
    morfeusz = morfeusz2.Morfeusz()
    # print("✔ Morfeusz2 loaded")
except ImportError:
    morfeusz = None
    # print("✗ Morfeusz2 missing: pip install morfeusz2")

try:
    import spacy
    nlp = spacy.load('pl_core_news_lg')
    # print("✔ spaCy loaded")
except:
    try:
        nlp = spacy.load('pl_core_news_sm')
        # print("✔ spaCy (small) loaded")
    except:
        nlp = None
        # print("✗ spaCy missing: pip install spacy && python -m spacy download pl_core_news_lg")

# GTMØ Theoretical Constants
PHI = (1 + np.sqrt(5)) / 2
SQRT_2_INV = 1 / np.sqrt(2)  
SINGULARITY_COORDS = np.array([1.0, 1.0, 0.0])  
COGNITIVE_CENTER = np.array([0.5, 0.5, 0.5])    
ENTROPY_THRESHOLD_SINGULARITY = 0.001             
BOUNDARY_THICKNESS = 0.02                        
META_REFLECTION_THRESHOLD = 0.95                 
DECOHERENCE_RATE = 0.02      
ENTANGLEMENT_THRESHOLD = 0.7  

# GTMØ coordinates for Polish cases
CASE_COORDS = {
    'nom': np.array([0.95, 0.92, 0.08]),  
    'gen': np.array([0.55, 0.25, 0.88]),  
    'dat': np.array([0.72, 0.65, 0.35]),  
    'acc': np.array([0.89, 0.85, 0.15]),  
    'ins': np.array([0.42, 0.18, 0.95]),  
    'loc': np.array([0.78, 0.95, 0.12]),  
    'voc': np.array([0.65, 0.35, 0.75])   
}

# GTMØ coordinates for Polish POS tags
POS_COORDS = {
    'subst': np.array([0.80, 0.85, 0.20]),  
    'adj': np.array([0.65, 0.68, 0.32]),    
    'verb': np.array([0.70, 0.45, 0.65]),   
    'adv': np.array([0.52, 0.38, 0.68]),    
    'num': np.array([0.95, 0.90, 0.10]),    
    'pron': np.array([0.68, 0.52, 0.53]),   
    'prep': np.array([0.76, 0.75, 0.24]),   
    'conj': np.array([0.65, 0.85, 0.20]),   
    'part': np.array([0.40, 0.26, 0.84]),   
    'interp': np.array([0.95, 0.95, 0.05])
}

# GTMØ coordinates for temporal analysis
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

# Patterns for rhetorical analysis (irony/sarcasm/paradox)
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
        'znowu', 'znów', 'kolejny', 'kolejna', 'kolejne', 'jeszcze raz', 'bez przerwy', 'bez rezultatu',
        'po raz kolejny', 'jak zawsze', 'można było się tego spodziewać','jak zwykle', 'oczywiście',
        'niestety', 'szkoda', 'porażka', 'bez sensu', 'beznadziejnie','przykro', 'źle', 'kiepsko'
    ]
}

class QuantumState(Enum):
    SUPERPOSITION = "⟨ψ|"
    COLLAPSED = "|ψ⟩" 
    ENTANGLED = "⟨ψ₁ψ₂|"
    DECOHERENT = "|ψ_mixed⟩"

class AxiomActivationLevel(Enum):
    DORMANT = 0.0
    MONITORING = 0.3
    ACTIVE = 0.7
    CRITICAL = 1.0

@dataclass
class QuantumSemanticState:
    amplitudes: Dict[str, complex]  
    phase: float                    
    coherence: float               
    entangled_with: List[str]      
    measurement_count: int         

@dataclass
class AxiomState:
    axiom_id: str
    activation_level: float
    last_activation_reason: str
    activation_count: int
    violations_prevented: int
    meta_reflection_triggered: bool

class SingularityError(Exception):
    pass

class EpistemicBoundaryError(Exception):
    pass

class GTMOAxiomSystem:
    """Complete implementation of GTMØ 13 Executable Axioms."""
    
    def __init__(self, enable_adaptive_learning: bool = True):
        self.enable_adaptive_learning = enable_adaptive_learning
        
        self.axiom_states = {
            f"AX{i}": AxiomState(f"AX{i}", 0.0, "", 0, 0, False) 
            for i in range(13)
        }
        
        self.adaptive_memory = {
            'contradiction_patterns': [],
            'boundary_states': [],
            'successful_avoidances': [],
            'meta_reflections': []
        }
        
        self.attractors = {
            'Ψᴷ': np.array([0.85, 0.85, 0.15]),  
            'Ψʰ': np.array([0.15, 0.15, 0.85]),  
            'Ψᴺ': np.array([0.50, 0.30, 0.90]),  
            'Ø': SINGULARITY_COORDS,              
            'Ψ~': np.array([0.50, 0.50, 0.80]),  
        }
        
        self.attractor_basins = {
            'Ψᴷ': 0.15, 'Ψʰ': 0.20, 'Ψᴺ': 0.25, 
            'Ø': 0.10, 'Ψ~': 0.18
        }
    
    def execute_all_axioms(self, system_state: Dict) -> Dict:
        """Execute all 13 axioms in sequence."""
        # Reset activation levels for each run
        for state in self.axiom_states.values():
            state.activation_level = AxiomActivationLevel.DORMANT.value
            state.last_activation_reason = ""
            state.meta_reflection_triggered = False
        modified_state = system_state.copy()

        for i in range(13):
            axiom_id = f"AX{i}"
            try:
                modified_state = self._execute_single_axiom(axiom_id, modified_state)
            except (SingularityError, EpistemicBoundaryError) as e:
                self._handle_axiom_violation(axiom_id, str(e), modified_state)

        modified_state['axiom_execution_summary'] = {
            'axioms_activated': sum(1 for state in self.axiom_states.values() 
                                  if state.activation_level > 0.3),
            'violations_prevented': sum(state.violations_prevented 
                                      for state in self.axiom_states.values()),
            'meta_reflections': sum(1 for state in self.axiom_states.values() 
                                  if state.meta_reflection_triggered)
        }

        return modified_state
    
    def _execute_single_axiom(self, axiom_id: str, state: Dict) -> Dict:
        """Execute single axiom based on ID."""
        axiom_methods = {
            "AX0": self._ax0_systemic_uncertainty,
            "AX1": self._ax1_ontological_difference,
            "AX2": self._ax2_translogical_isolation,
            "AX3": self._ax3_epistemic_singularity,
            "AX4": self._ax4_non_representability,
            "AX5": self._ax5_topological_boundary,
            "AX6": self._ax6_heuristic_extremum,
            "AX7": self._ax7_meta_closure,
            "AX8": self._ax8_not_limit_point,
            "AX9": self._ax9_operator_irreducibility,
            "AX10": self._ax10_meta_operator_definition,
            "AX11": self._ax11_adaptive_learning,
            "AX12": self._ax12_topological_classification
        }
        
        if axiom_id in axiom_methods:
            return axiom_methods[axiom_id](state)
        return state
    
    def _extract_coordinates(self, state: Dict) -> Optional[np.ndarray]:
        """Extract coordinates from state."""
        if 'coordinates' in state:
            coords_dict = state['coordinates']
            if isinstance(coords_dict, dict):
                return np.array([
                    coords_dict.get('determination', 0.5),
                    coords_dict.get('stability', 0.5),
                    coords_dict.get('entropy', 0.5)
                ])
            elif isinstance(coords_dict, (list, tuple, np.ndarray)):
                return np.array(coords_dict)
        return None
    
    def _update_coordinates(self, state: Dict, new_coords: np.ndarray) -> Dict:
        """Update coordinates in state."""
        state['coordinates'] = {
            'determination': float(new_coords[0]),
            'stability': float(new_coords[1]),
            'entropy': float(new_coords[2])
        }
        return state
    
    def _activate_axiom(self, axiom_id: str, reason: str):
        """Mark axiom as activated."""
        if axiom_id in self.axiom_states:
            self.axiom_states[axiom_id].activation_level = AxiomActivationLevel.ACTIVE.value
            self.axiom_states[axiom_id].last_activation_reason = reason
            self.axiom_states[axiom_id].activation_count += 1
    
    def _handle_axiom_violation(self, axiom_id: str, violation: str, state: Dict):
        """Handle detected axiom violation."""
        if axiom_id in self.axiom_states:
            self.axiom_states[axiom_id].violations_prevented += 1
        
        state['axiom_violations'] = state.get('axiom_violations', [])
        state['axiom_violations'].append({
            'axiom': axiom_id,
            'violation': violation,
            'timestamp': datetime.now().isoformat()
        })
    
    # Complete axiom implementations
    def _ax0_systemic_uncertainty(self, state: Dict) -> Dict:
        coords = self._extract_coordinates(state)
        if coords is None:
            return state
        
        # Introduce quantum superposition
        uncertainty_vector = np.array([-0.01, -0.005, 0.015])
        new_coords = coords + uncertainty_vector
        new_coords = np.clip(new_coords, 0, 1)
        
        return self._update_coordinates(state, new_coords)
    
    def _ax1_ontological_difference(self, state: Dict) -> Dict:
        coords = self._extract_coordinates(state)
        if coords is None:
            return state
        
        distance_to_singularity = np.linalg.norm(coords - SINGULARITY_COORDS)
        if distance_to_singularity < 0.05:
            self._activate_axiom("AX1", f"Preventing singularity approach")
            direction = coords - SINGULARITY_COORDS
            if np.linalg.norm(direction) > 0:
                push_direction = direction / np.linalg.norm(direction)
                new_coords = coords + push_direction * 0.02
                new_coords = np.clip(new_coords, 0, 1)
                return self._update_coordinates(state, new_coords)
        
        return state
    
    def _ax2_translogical_isolation(self, state: Dict) -> Dict:
        """Prevent definable functions from producing Singularity."""
        coords = self._extract_coordinates(state)
        if coords is None:
            return state
        
        # Check if approaching singularity through computation
        if 'operation_result' in state:
            result_coords = self._extract_coordinates({'coordinates': state['operation_result']})
            if result_coords is not None:
                distance = np.linalg.norm(result_coords - SINGULARITY_COORDS)
                if distance < 0.01:
                    self._activate_axiom("AX2", "Blocking translogical path to Singularity")
                    state['operation_result'] = COGNITIVE_CENTER.tolist()
                    raise SingularityError("AX2: Translogical isolation prevents path to Ø")
        
        return state
    
    def _ax3_epistemic_singularity(self, state: Dict) -> Dict:
        """Prevent claims of knowing Singularity."""
        coords = self._extract_coordinates(state)
        if coords is None:
            return state
            
        distance = np.linalg.norm(coords - SINGULARITY_COORDS)
        if distance < 0.1 and 'knowledge_claims' in state:
            self._activate_axiom("AX3", "Blocking epistemic claims near Singularity")
            new_coords = coords * 0.9 + COGNITIVE_CENTER * 0.1
            return self._update_coordinates(state, new_coords)
        
        return state
    
    def _ax4_non_representability(self, state: Dict) -> Dict:
        """Prevent standard representation of Singularity."""
        coords = self._extract_coordinates(state)
        if coords is None:
            return state
            
        if 'representation' in state and isinstance(state['representation'], (int, float)):
            distance = np.linalg.norm(coords - SINGULARITY_COORDS)
            if distance < 0.1:
                self._activate_axiom("AX4", "Blocking standard representation of Ø")
                state['representation'] = {'type': 'indefinite'}
        
        return state
    
    def _ax5_topological_boundary(self, state: Dict) -> Dict:
        """Maintain Singularity at boundary."""
        coords = self._extract_coordinates(state)
        if coords is None:
            return state
            
        if coords[0] > 0.9 and coords[1] > 0.9 and coords[2] < 0.1:
            distance = np.linalg.norm(coords - SINGULARITY_COORDS)
            if distance < BOUNDARY_THICKNESS:
                self._activate_axiom("AX5", "Maintaining Ø at boundary")
                if distance < 0.01:
                    new_coords = np.array([0.98, 0.98, 0.02])
                    return self._update_coordinates(state, new_coords)
        
        return state
    
    def _ax6_heuristic_extremum(self, state: Dict) -> Dict:
        """Enforce minimal entropy for Singularity."""
        coords = self._extract_coordinates(state)
        if coords is None:
            return state
            
        distance = np.linalg.norm(coords - SINGULARITY_COORDS)
        if distance < 0.1:
            current_entropy = coords[2]
            if current_entropy > ENTROPY_THRESHOLD_SINGULARITY:
                self._activate_axiom("AX6", f"Enforcing minimal entropy near Ø")
                proximity_factor = (0.1 - distance) / 0.1
                target_entropy = current_entropy * (1 - proximity_factor * 0.9)
                target_entropy = max(target_entropy, ENTROPY_THRESHOLD_SINGULARITY)
                new_coords = coords.copy()
                new_coords[2] = target_entropy
                return self._update_coordinates(state, new_coords)
        
        return state
    
    def _ax7_meta_closure(self, state: Dict) -> Dict:
        """Trigger meta-reflection near singularity."""
        coords = self._extract_coordinates(state)
        if coords is None:
            return state
        
        distance_to_singularity = np.linalg.norm(coords - SINGULARITY_COORDS)
        
        if distance_to_singularity < 0.1:
            self._activate_axiom("AX7", "Triggering meta-cognitive self-evaluation")
            meta_uncertainty = min(0.1, (0.1 - distance_to_singularity) * 2)
            new_coords = coords + np.array([-meta_uncertainty, -meta_uncertainty, meta_uncertainty])
            new_coords = np.clip(new_coords, 0, 1)
            
            state = self._update_coordinates(state, new_coords)
            state['meta_reflection'] = {
                'triggered_by': 'AX7_proximity_to_singularity',
                'uncertainty_applied': meta_uncertainty
            }
            
            self.axiom_states["AX7"].meta_reflection_triggered = True
        
        return state
    
    def _ax8_not_limit_point(self, state: Dict) -> Dict:
        """Prevent sequences converging to Singularity."""
        if 'trajectory' in state and len(state['trajectory']) > 3:
            trajectory = np.array(state['trajectory'])
            recent_points = trajectory[-3:]
            distances = [np.linalg.norm(point - SINGULARITY_COORDS) for point in recent_points]
            
            if len(distances) >= 2 and distances[-1] < distances[-2] and distances[-1] < 0.05:
                self._activate_axiom("AX8", "Preventing trajectory convergence to Ø")
                current_coords = recent_points[-1]
                direction_away = current_coords - SINGULARITY_COORDS
                direction_away = direction_away / (np.linalg.norm(direction_away) + 1e-8)
                new_coords = current_coords + direction_away * 0.02
                new_coords = np.clip(new_coords, 0, 1)
                return self._update_coordinates(state, new_coords)
        
        return state
    
    def _ax9_operator_irreducibility(self, state: Dict) -> Dict:
        """Prevent standard operators from acting on Singularity."""
        coords = self._extract_coordinates(state)
        if coords is None:
            return state
            
        if 'operation' in state:
            distance = np.linalg.norm(coords - SINGULARITY_COORDS)
            if distance < 0.05:
                blocked_ops = ['add', 'subtract', 'multiply', 'divide']
                if any(op in str(state['operation']).lower() for op in blocked_ops):
                    self._activate_axiom("AX9", f"Blocking operator near Ø")
                    state['operation_result'] = {'type': 'irreducible'}
                    raise SingularityError(f"AX9: Operator cannot act on Ø")
        
        return state
    
    def _ax10_meta_operator_definition(self, state: Dict) -> Dict:
        """Allow only meta-operators near Singularity."""
        coords = self._extract_coordinates(state)
        if coords is None:
            return state
            
        distance = np.linalg.norm(coords - SINGULARITY_COORDS)
        if distance < 0.05 and 'meta_operation' in state:
            allowed_meta_ops = ['psi_gtmo', 'e_gtmo', 'topology_classification']
            if any(allowed in str(state['meta_operation']).lower() for allowed in allowed_meta_ops):
                self._activate_axiom("AX10", f"Allowing meta-operator near Ø")
            else:
                raise SingularityError("AX10: Only meta-operators can act near Ø")
        
        return state
    
    def _ax11_adaptive_learning(self, state: Dict) -> Dict:
        """Learn from boundary encounters."""
        if not self.enable_adaptive_learning:
            return state
            
        coords = self._extract_coordinates(state)
        if coords is None:
            return state
            
        boundary_distances = np.minimum(coords, 1 - coords)
        min_boundary_distance = np.min(boundary_distances)
        
        if min_boundary_distance < BOUNDARY_THICKNESS:
            self._activate_axiom("AX11", "Learning from boundary encounter")
            self.adaptive_memory['boundary_states'].append({
                'coordinates': coords.tolist(),
                'boundary_distance': float(min_boundary_distance)
            })
            
            if len(self.adaptive_memory['boundary_states']) > 5:
                similar_states = [s for s in self.adaptive_memory['boundary_states'][-10:]
                                if np.linalg.norm(np.array(s['coordinates']) - coords) < 0.1]
                
                if len(similar_states) > 2:
                    avoidance_strength = min(0.05, len(similar_states) * 0.01)
                    center_direction = COGNITIVE_CENTER - coords
                    center_direction = center_direction / (np.linalg.norm(center_direction) + 1e-8)
                    new_coords = coords + center_direction * avoidance_strength
                    new_coords = np.clip(new_coords, 0, 1)
                    return self._update_coordinates(state, new_coords)
        
        return state
    
    def _ax12_topological_classification(self, state: Dict) -> Dict:
        """Classify knowledge via topological attractors."""
        coords = self._extract_coordinates(state)
        if coords is None:
            return state
        
        nearest_attractor = None
        min_distance = float('inf')
        
        for attractor_name, attractor_coords in self.attractors.items():
            distance = np.linalg.norm(coords - attractor_coords)
            if distance < min_distance:
                min_distance = distance
                nearest_attractor = attractor_name
        
        if nearest_attractor and min_distance <= self.attractor_basins[nearest_attractor]:
            self._activate_axiom("AX12", f"Classifying via attractor: {nearest_attractor}")
            
            state['topological_classification'] = {
                'attractor': nearest_attractor,
                'distance': float(min_distance),
                'in_basin': True
            }
            
            if min_distance > 0:
                direction = (self.attractors[nearest_attractor] - coords)
                direction = direction / np.linalg.norm(direction)
                pull_strength = 0.01 * (1 - min_distance / self.attractor_basins[nearest_attractor])
                new_coords = coords + direction * pull_strength
                new_coords = np.clip(new_coords, 0, 1)
                state = self._update_coordinates(state, new_coords)
        
        return state

# =============================================================================
# TEMPORAL ANALYSIS FUNCTIONS
# =============================================================================

def analyze_temporality(text: str, doc=None) -> Tuple[np.ndarray, Dict]:
    """
    Analyze temporality of utterance using spaCy.

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
# ENHANCED RHETORICAL ANALYSIS FUNCTIONS
# =============================================================================

def has_semantic_contradiction(text: str) -> bool:
    """Detect semantic contradiction suggesting irony."""
    import re
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

def detect_enhanced_rhetorical_mode(text: str, base_coords: np.ndarray, morph_metadata: Dict) -> Tuple[np.ndarray, str, Dict]:
    """
    Enhanced detection of irony/sarcasm (inversion) or paradox (preservation).

    Args:
        text: Text to analyze
        base_coords: Base coordinates [D, S, E]
        morph_metadata: Morphological metadata

    Returns:
        Transformed coordinates, mode name, and metadata
    """
    import re
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

# =============================================================================
# QUANTUM SUPERPOSITION FOR AMBIGUITY
# =============================================================================

class QuantumAmbiguityAnalyzer:
    """Quantum representation for ambiguous utterances."""

    def __init__(self, superposition_threshold: float = 0.3):
        self.superposition_threshold = superposition_threshold
        self.ambiguity_indicators = [
            'może', 'chyba', 'prawdopodobnie', 'możliwe', 'ewentualnie',
            'lub', 'albo', 'czy', 'bądź', 'względnie',
            'zarówno', 'jak i', 'ani', 'niby', 'jakby',
            'przypuszczalnie', 'domniemanie', 'najprawdopodobniej'
        ]

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

    def create_superposition(self, interpretations: List[np.ndarray],
                           probabilities: Optional[List[float]] = None) -> Dict:
        """
        Create superposition of states for ambiguous utterances.

        Args:
            interpretations: List of possible interpretations [D,S,E]
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
            interp_array = np.array(interpretations)
            if interp_array.ndim == 1:
                interp_array = interp_array.reshape(1, -1)

            try:
                covariance = np.cov(interp_array.T, aweights=probs)
            except:
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

class QuantumMorphosyntaxEngine:
    def calculate_adaptive_weights(self, text: str, morph_meta: Dict, synt_meta: Dict) -> Tuple[float, float]:
        word_count = len(text.split())
        # Krótkie fragmenty (1-3 słowa)
        if word_count <= 3:
            morph_weight = 0.40  # Zmniejszona z 0.64
            # Ale jeśli to tylko liczby/interpunkcja, jeszcze mniej
            if morph_meta.get('pos', {}).get('interp', 0) > word_count/2:
                morph_weight = 0.25
        # Średnie zdania (4-15 słów)
        elif word_count <= 15:
            morph_weight = 0.64  # Standardowa
        # Długie zdania (16-30 słów)
        elif word_count <= 30:
            morph_weight = 0.55  # Więcej składni
        # Bardzo długie okresy (30+ słów)
        else:
            morph_weight = 0.45  # Składnia dominuje
        synt_weight = 1.0 - morph_weight
        return morph_weight, synt_weight

    def __init__(self, domain_dictionary: Optional['DomainDictionary'] = None):
        self.case_coords = CASE_COORDS
        self.pos_coords = POS_COORDS
        self.quantum_states = {}
        self.entanglement_matrix = {}
        self.decoherence_history = {}
        self.pauli_x = np.array([[0, 1], [1, 0]], dtype=complex)
        self.pauli_y = np.array([[0, -1j], [1j, 0]], dtype=complex)
        self.pauli_z = np.array([[1, 0], [0, -1]], dtype=complex)
        self.axiom_system = GTMOAxiomSystem(enable_adaptive_learning=True)

        # Initialize rhetorical analyzer if available
        if RHETORICAL_ANALYZER_AVAILABLE:
            self.rhetorical_analyzer = GTMORhetoricalAnalyzer()
        else:
            self.rhetorical_analyzer = None

        # Initialize domain dictionary
        self.domain_dictionary = domain_dictionary
        self.use_domain_filtering = domain_dictionary is not None

        # Initialize constitutional duality calculator
        if CONSTITUTIONAL_DUALITY_AVAILABLE:
            self.constitutional_calculator = ConstitutionalDualityCalculator()
        else:
            self.constitutional_calculator = None

        # Initialize quantum ambiguity analyzer
        self.quantum_ambiguity_analyzer = QuantumAmbiguityAnalyzer()

    def analyze_morphology_quantum(self, text: str) -> Tuple[np.ndarray, Dict, Dict[str, QuantumSemanticState]]:
        """Morphological analysis with quantum superposition."""
        if not morfeusz:
            raise Exception("Morfeusz2 not available")
        
        coords_list = []
        case_counts = {}
        pos_counts = {}
        debug_info = []
        word_quantum_states = {}
        
        try:
            analyses = morfeusz.analyse(text)
            
            # Group analyses by word form
            word_analyses = {}
            for start, end, (form, lemma, tag, labels, qualifiers) in analyses:
                if not form or form.isspace():
                    continue
                
                if form not in word_analyses:
                    word_analyses[form] = []
                word_analyses[form].append((start, end, form, lemma, tag, labels, qualifiers))
            
            # Process each word with quantum superposition
            for form, analyses_list in word_analyses.items():
                # Apply domain dictionary filtering if available
                if self.use_domain_filtering and self.domain_dictionary:
                    # Filter analyses using domain knowledge
                    filtered_analyses = analyses_list.copy()

                    # Use domain dictionary to prioritize
                    domain_scores = []
                    for analysis in filtered_analyses:
                        _, _, _, lemma, tag, _, _ = analysis
                        score = 0

                        # Check if lemma is domain term
                        if self.domain_dictionary.is_domain_term(lemma):
                            score += 5

                        # Check if tag matches domain tags
                        domain_tags = self.domain_dictionary.get_domain_tags_for_word(lemma)
                        if tag in domain_tags:
                            score += 10

                        domain_scores.append(score)

                    # If we have domain-specific analyses, prioritize them
                    if max(domain_scores) > 0:
                        # Keep only analyses with highest scores
                        max_score = max(domain_scores)
                        analyses_list = [
                            filtered_analyses[i] for i, s in enumerate(domain_scores)
                            if s >= max_score * 0.7  # Keep analyses with score >= 70% of max
                        ]

                word_case_frequencies = {}
                word_pos_frequencies = {}

                for start, end, form, lemma, tag, labels, qualifiers in analyses_list:
                    debug_info.append(f"{lemma}:{tag}")
                    tag_parts = tag.split(':')
                    main_pos = tag_parts[0]
                    
                    # Extract case
                    for part in tag_parts:
                        if part in CASE_COORDS:
                            word_case_frequencies[part] = word_case_frequencies.get(part, 0) + 1
                            case_counts[part] = case_counts.get(part, 0) + 1
                            break
                        elif '.' in part:
                            sub_parts = part.split('.')
                            for sub_part in sub_parts:
                                if sub_part in CASE_COORDS:
                                    word_case_frequencies[sub_part] = word_case_frequencies.get(sub_part, 0) + 1
                                    case_counts[sub_part] = case_counts.get(sub_part, 0) + 1
                                    break
                    
                    # Extract POS
                    if main_pos in POS_COORDS:
                        word_pos_frequencies[main_pos] = word_pos_frequencies.get(main_pos, 0) + 1
                        pos_counts[main_pos] = pos_counts.get(main_pos, 0) + 1
                
                # Create quantum superposition state for this word
                if word_case_frequencies:
                    quantum_state = self.create_superposition_state(form, word_case_frequencies)
                    word_quantum_states[form] = quantum_state
                    
                    # Collapse to get coordinates
                    collapsed_coords = self.collapse_superposition(form, list(word_case_frequencies.keys()))
                    coords_list.append(collapsed_coords)
                
                # Add POS coordinates
                for pos, freq in word_pos_frequencies.items():
                    coords_list.append(POS_COORDS[pos])
            
            if coords_list:
                final_coords = np.mean(coords_list, axis=0)
            else:
                final_coords = np.array([0.5, 0.5, 0.5])
                
            metadata = {
                'total_analyses': len(analyses),
                'cases': case_counts,
                'pos': pos_counts,
                'ambiguity': len(analyses) / len(text.split()) if text.split() else 1.0,
                'debug_tags': debug_info[:10],
                'coords_count': len(coords_list),
                'quantum_words': len(word_quantum_states)
            }
            
            return final_coords, metadata, word_quantum_states
            
        except Exception as e:
            raise Exception(f"Morphological quantum analysis failed: {e}")
    
    def analyze_syntax_quantum(self, text: str, word_quantum_states: Dict[str, QuantumSemanticState]) -> Tuple[np.ndarray, Dict, List[Tuple[str, str, float]]]:
        """Syntactic analysis with quantum entanglement detection."""
        # Fallback analysis if spaCy fails
        def fallback_syntax_analysis(text: str) -> Tuple[np.ndarray, Dict]:
            """Heuristic syntax analysis without spaCy."""
            words = text.split()
            word_count = len(words)
            sentences = text.replace('!', '.').replace('?', '.').split('.')
            sentences = [s.strip() for s in sentences if s.strip()]
            sent_count = len(sentences)
            
            # Calculate average sentence length
            avg_sent_len = word_count / sent_count if sent_count > 0 else word_count
            
            # DETERMINATION based on sentence structure indicators
            determination = 0.5
            
            # Short, simple sentences = higher determination
            if avg_sent_len < 7:
                determination = 0.75
            elif avg_sent_len < 15:
                determination = 0.6
            else:
                determination = 0.45
            
            # Questions reduce determination
            if '?' in text:
                determination *= 0.8
            
            # Exclamations can indicate strong determination or emotion
            if '!' in text:
                determination *= 0.95
            
            # STABILITY based on punctuation and structure
            punct_count = sum(1 for char in text if char in '.,;:!?-–—')
            punct_ratio = punct_count / word_count if word_count > 0 else 0
            
            # Moderate punctuation = stable structure
            if punct_ratio < 0.1:
                stability = 0.7  # Too few punctuation
            elif punct_ratio < 0.2:
                stability = 0.8  # Good balance
            else:
                stability = 0.5  # Too much punctuation
            
            # Multiple sentences reduce stability slightly
            if sent_count > 1:
                stability *= (0.95 ** (sent_count - 1))
            
            # ENTROPY based on complexity indicators
            entropy = 0.3  # Base entropy
            
            # Long sentences increase entropy
            if avg_sent_len > 15:
                entropy += min((avg_sent_len - 15) / 30, 0.3)
            
            # Commas indicate complex structure
            comma_count = text.count(',')
            if comma_count > 0:
                entropy += min(comma_count * 0.05, 0.2)
            
            # Conjunctions indicate complexity
            conjunctions = ['i', 'oraz', 'ale', 'lecz', 'jednak', 'czy', 'lub', 'albo', 'ani', 'więc', 'zatem']
            conj_count = sum(1 for word in words if word.lower() in conjunctions)
            if conj_count > 0:
                entropy += min(conj_count * 0.04, 0.15)
            
            coords = np.array([determination, stability, entropy])
            coords = np.clip(coords, 0.05, 0.95)
            
            metadata = {
                'method': 'fallback_heuristic',
                'words': word_count,
                'sentences': sent_count,
                'avg_sent_length': avg_sent_len,
                'punctuation_ratio': punct_ratio,
                'conjunction_count': conj_count
            }
            
            return coords, metadata
        
        # Try spaCy first
        if nlp:
            try:
                doc = nlp(text)
                
                # Check if spaCy actually parsed dependencies
                has_dependencies = False
                for token in doc:
                    if token.dep_ and token.dep_ != "":
                        has_dependencies = True
                        break
                
                if not has_dependencies:
                    # print("  SYNTAX: spaCy returned no dependencies, using fallback")
                    pass
                    coords, metadata = fallback_syntax_analysis(text)
                    return coords, metadata, []
                
                # Count dependencies
                dep_counts = {}
                max_depth = 0
                avg_depth = 0
                depth_variance = 0
                depth_list = []
                entanglements = []
                total_tokens = len(doc)
                
                for token in doc:
                    dep = token.dep_
                    if dep:  # Only count non-empty dependencies
                        dep_counts[dep] = dep_counts.get(dep, 0) + 1

                    # Enhanced depth calculation with full tree analysis
                    depth = len(list(token.ancestors))
                    depth_list.append(depth)
                    max_depth = max(max_depth, depth)
                    
                    # Check for syntactic entanglement
                    if token.head != token:
                        head_form = token.head.text.lower()
                        token_form = token.text.lower()
                        
                        if head_form in word_quantum_states and token_form in word_quantum_states:
                            interference = self.measure_quantum_interference(head_form, token_form)
                            if interference > ENTANGLEMENT_THRESHOLD:
                                self.create_entanglement(head_form, token_form, coupling_strength=0.5)
                                entanglements.append((head_form, token_form, interference))

                # Calculate depth statistics
                if depth_list:
                    avg_depth = np.mean(depth_list)
                    depth_variance = np.var(depth_list)
                else:
                    avg_depth = 0
                    depth_variance = 0

                # If we got this far but have no meaningful dependencies, use fallback
                if not dep_counts or len(dep_counts) < 2:
                    print("  SYNTAX: Too few dependency types found, using fallback")
                    coords, metadata = fallback_syntax_analysis(text)
                    metadata['spacy_attempted'] = True
                    return coords, metadata, entanglements
                
                # Calculate coordinates based on dependencies
                # DETERMINATION
                core_deps = ['ROOT', 'nsubj', 'nsubj:pass', 'obj', 'iobj', 'csubj']
                core_count = sum(dep_counts.get(dep, 0) for dep in core_deps)
                
                determination = 0.4 + (core_count / total_tokens) * 1.5
                determination = min(determination, 0.9)
                
                # Questions and negations
                if any(token.text == '?' for token in doc):
                    determination *= 0.8
                if 'neg' in dep_counts:
                    determination *= 0.85
                
                # STABILITY
                num_dep_types = len(dep_counts)
                # Polish might have different dependency patterns
                if num_dep_types < 4:
                    stability = 0.75
                elif num_dep_types < 8:
                    stability = 0.6
                else:
                    stability = 0.45
                
                sentences = list(doc.sents)
                if len(sentences) > 1:
                    stability *= (0.95 ** (len(sentences) - 1))
                
                # ENTROPY
                entropy = 0.25  # Base entropy
                
                if max_depth > 2:
                    entropy += min((max_depth - 2) * 0.08, 0.25)
                
                # Complex dependencies
                complex_deps = ['ccomp', 'xcomp', 'advcl', 'acl', 'relcl', 'conj', 'cc', 'csubj']
                complex_count = sum(dep_counts.get(d, 0) for d in complex_deps)
                if complex_count > 0:
                    entropy += min(complex_count * 0.04, 0.2)
                
                coords = np.array([determination, stability, entropy])
                
                # Apply entanglement effects
                if entanglements:
                    ent_factor = len(entanglements) / total_tokens
                    coords[1] *= (1 + ent_factor * 0.05)
                    coords[2] *= (1 + ent_factor * 0.03)
                
                coords = np.clip(coords, 0.05, 0.95)
                
                print(f"  SYNTAX_DEPS: {list(dep_counts.keys())[:8]}")  # First 8 deps
                print(f"  SYNTAX_VALUES: core={core_count}, types={num_dep_types}, depth={max_depth}")
                print(f"  SYNTAX_COORDS: [{coords[0]:.3f}, {coords[1]:.3f}, {coords[2]:.3f}]")
                
                metadata = {
                    'method': 'spacy_dependencies',
                    'tokens': total_tokens,
                    'sentences': len(sentences),
                    'max_depth': max_depth,
                    'avg_depth': float(avg_depth),
                    'depth_variance': float(depth_variance),
                    'dependencies': dep_counts,
                    'dep_types': num_dep_types,
                    'core_ratio': core_count / total_tokens if total_tokens > 0 else 0
                }
                
                return coords, metadata, entanglements
                
            except Exception as e:
                print(f"  SYNTAX_ERROR: {str(e)}, using fallback")
                coords, metadata = fallback_syntax_analysis(text)
                metadata['error'] = str(e)
                return coords, metadata, []
        
        else:
            # No spaCy available, use fallback
            print("  SYNTAX: spaCy not available, using fallback")
            coords, metadata = fallback_syntax_analysis(text)
            return coords, metadata, []
    
    def create_superposition_state(self, word: str, case_frequencies: Dict[str, int]) -> QuantumSemanticState:
        """Create quantum superposition state for word."""
        total_observations = sum(case_frequencies.values())
        if total_observations == 0:
            amplitudes = {case: SQRT_2_INV for case in self.case_coords.keys()}
        else:
            amplitudes = {}
            for case, freq in case_frequencies.items():
                probability = freq / total_observations
                amplitude = np.sqrt(probability) * cmath.exp(1j * np.random.uniform(0, 2*np.pi))
                amplitudes[case] = amplitude
        
        phase = np.angle(sum(amplitudes.values()))
        coherence = abs(sum(amplitudes.values()))**2 / sum(abs(amp)**2 for amp in amplitudes.values())
        
        quantum_state = QuantumSemanticState(
            amplitudes=amplitudes,
            phase=phase,
            coherence=coherence,
            entangled_with=[],
            measurement_count=0
        )
        
        self.quantum_states[word] = quantum_state
        return quantum_state
    
    def collapse_superposition(self, word: str, observed_cases: List[str]) -> np.ndarray:
        """Collapse quantum superposition to classical coordinates."""
        if word not in self.quantum_states:
            coords_list = [self.case_coords[case] for case in observed_cases if case in self.case_coords]
            return np.mean(coords_list, axis=0) if coords_list else np.array([0.5, 0.5, 0.5])
        
        quantum_state = self.quantum_states[word]
        
        total_probability = 0
        weighted_coords = np.zeros(3)
        
        for case in observed_cases:
            if case in quantum_state.amplitudes and case in self.case_coords:
                probability = abs(quantum_state.amplitudes[case])**2
                total_probability += probability
                weighted_coords += probability * self.case_coords[case]
        
        if total_probability > 0:
            collapsed_coords = weighted_coords / total_probability
        else:
            coords_list = [self.case_coords[case] for case in observed_cases if case in self.case_coords]
            collapsed_coords = np.mean(coords_list, axis=0) if coords_list else np.array([0.5, 0.5, 0.5])
        
        quantum_state.measurement_count += 1
        self._apply_decoherence(word)
        
        return collapsed_coords
    
    def _apply_decoherence(self, word: str):
        """Apply quantum decoherence over time."""
        if word not in self.quantum_states:
            return
        
        quantum_state = self.quantum_states[word]
        quantum_state.coherence *= (1 - DECOHERENCE_RATE)
        
        for case in quantum_state.amplitudes:
            phase_noise = np.random.normal(0, DECOHERENCE_RATE)
            quantum_state.amplitudes[case] *= cmath.exp(1j * phase_noise)
    
    def create_entanglement(self, word1: str, word2: str, coupling_strength: float = 0.5):
        """Create quantum entanglement between words."""
        if word1 not in self.quantum_states or word2 not in self.quantum_states:
            return
        
        state1 = self.quantum_states[word1]
        state2 = self.quantum_states[word2]
        
        for case in state1.amplitudes:
            if case in state2.amplitudes:
                coupled_amp = coupling_strength * (state1.amplitudes[case] + state2.amplitudes[case])
                state1.amplitudes[case] = coupled_amp
                state2.amplitudes[case] = coupled_amp.conjugate()
        
        if word2 not in state1.entangled_with:
            state1.entangled_with.append(word2)
        if word1 not in state2.entangled_with:
            state2.entangled_with.append(word1)
        
        self.entanglement_matrix[(word1, word2)] = coupling_strength
        self.entanglement_matrix[(word2, word1)] = coupling_strength
    
    def measure_quantum_interference(self, word1: str, word2: str) -> float:
        """Measure quantum interference between two words."""
        if word1 not in self.quantum_states or word2 not in self.quantum_states:
            return 0.0
        
        state1 = self.quantum_states[word1]
        state2 = self.quantum_states[word2]
        
        interference = 0.0
        for case in state1.amplitudes:
            if case in state2.amplitudes:
                combined = state1.amplitudes[case] + state2.amplitudes[case]
                individual_sum = abs(state1.amplitudes[case])**2 + abs(state2.amplitudes[case])**2
                combined_prob = abs(combined)**2
                interference += abs(combined_prob - individual_sum)
        
        return interference
    
    def _classify_interpretation(self, coords: Dict[str, float]) -> Dict[str, str]:
        """Classify interpretation based on coordinates."""
        d, s, e = coords['determination'], coords['stability'], coords['entropy']
        
        # Determine levels
        det_level = "high" if d > 0.7 else ("moderate" if d > 0.4 else "low")
        stab_level = "high" if s > 0.7 else ("moderate" if s > 0.4 else "low")
        ent_level = "low" if e < 0.3 else ("mixed" if e < 0.6 else "high")
        
        # Overall quality assessment
        if d > 0.7 and s > 0.7 and e < 0.3:
            overall = "high_quality"
        elif d > 0.5 and s > 0.5 and e < 0.5:
            overall = "moderate_quality"
        elif d < 0.3 or s < 0.3 or e > 0.7:
            overall = "low_quality"
        else:
            overall = "mixed_quality"
        
        # Calculate overall score
        overall_score = (d * 0.4 + s * 0.3 + (1 - e) * 0.3)
        
        return {
            'determination': det_level,
            'stability': stab_level,
            'entropy': ent_level,
            'overall': overall,
            'overall_score': round(overall_score, 4)
        }
    
    def _classify_superposition_type(self, word_quantum_states: Dict[str, QuantumSemanticState]) -> str:
        """Classify type of quantum superposition in the text."""
        if not word_quantum_states:
            return "VACUUM_STATE"
        
        total_coherence = sum(qs.coherence for qs in word_quantum_states.values())
        avg_coherence = total_coherence / len(word_quantum_states)
        
        entangled_count = sum(len(qs.entangled_with) for qs in word_quantum_states.values()) / 2
        
        if avg_coherence > 0.8:
            if entangled_count > len(word_quantum_states) / 2:
                return "HIGHLY_ENTANGLED_COHERENT"
            else:
                return "COHERENT_SUPERPOSITION"
        elif avg_coherence > 0.5:
            if entangled_count > 0:
                return "PARTIALLY_ENTANGLED_MIXED"
            else:
                return "MIXED_SUPERPOSITION"
        else:
            return "DECOHERENT_CLASSICAL"
    
    def gtmo_analyze_quantum(self, text: str, source_file: Optional[Dict] = None) -> Dict:
        """
        Main GTMØ quantum analysis function with axiom integration.
        Returns JSON-compatible result.
        """
        if not text or not text.strip():
            raise Exception("Empty text")
        
        print(f"🌟 Quantum analyzing: {text[:50]}...")
        
        # Run morphological analysis with quantum superposition
        morph_coords, morph_meta, word_quantum_states = self.analyze_morphology_quantum(text)
        
        # Run syntactic analysis with entanglement detection
        synt_coords, synt_meta, entanglements = self.analyze_syntax_quantum(text, word_quantum_states)

        # TEMPORAL ANALYSIS
        doc = nlp(text) if nlp else None
        temporal_coords, temporal_meta = analyze_temporality(text, doc)

        # Calculate quantum coherence metrics
        total_quantum_coherence = 0.0
        if word_quantum_states:
            total_quantum_coherence = np.mean([qs.coherence for qs in word_quantum_states.values()])
        
        # Use adaptive fusion weights instead of fixed ones
        morph_weight, synt_weight = self.calculate_adaptive_weights(text, morph_meta, synt_meta)
        
        # Add quantum coherence adjustment
        morph_weight += 0.1 * total_quantum_coherence
        morph_weight = np.clip(morph_weight, 0.2, 0.8)
        synt_weight = 1.0 - morph_weight
        
        # Debug: show weight calculation
        print(f"  ADAPTIVE_WEIGHTS: morph={morph_weight:.3f}, synt={synt_weight:.3f}")
        print(f"  WEIGHT_FACTORS: words={len(text.split())}, depth={synt_meta.get('max_depth', 0)}, ambiguity={morph_meta.get('ambiguity', 1.0):.2f}")
        
        final_coords = morph_weight * morph_coords + synt_weight * synt_coords
        final_coords = np.clip(final_coords, 0, 1)

        # ENHANCED RHETORICAL ANALYSIS (Extended from gtmo_extended.py)
        rhetorical_coords, rhetorical_mode, rhetorical_metadata_extended = detect_enhanced_rhetorical_mode(
            text, final_coords, morph_meta
        )

        # If irony/paradox detected, update final coordinates
        if rhetorical_mode in ['irony', 'paradox']:
            final_coords = rhetorical_coords
            print(f"  🎭 EXTENDED RHETORICAL: {rhetorical_mode.upper()} detected")
            print(f"     Coordinates transformed: D={final_coords[0]:.3f}, S={final_coords[1]:.3f}, E={final_coords[2]:.3f}")

        # QUANTUM AMBIGUITY ANALYSIS
        needs_quantum, quantum_ambiguity_meta = self.quantum_ambiguity_analyzer.detect_ambiguity(text)

        quantum_superposition_state = None
        if needs_quantum:
            # Generate alternative interpretations
            alt_interpretations = generate_alternative_interpretations(text, final_coords)
            quantum_superposition_state = self.quantum_ambiguity_analyzer.create_superposition(
                alt_interpretations['states'],
                alt_interpretations.get('probabilities')
            )
            print(f"  ⚛️ QUANTUM SUPERPOSITION: {len(alt_interpretations['states'])} states detected")
            print(f"     Von Neumann entropy: {quantum_superposition_state['von_neumann_entropy']:.4f}")

        # Quantum tensor: T_quantum = D × S × (1-E)
        D = float(final_coords[0])
        S = float(final_coords[1])
        E = float(final_coords[2])
        T_quantum = D * S * (1 - E)
        tensor_print = f"  T_QUANTUM: {D:.3f} × {S:.3f} × {1-E:.3f} = {T_quantum:.3f}"
        
        # Create state for axiom system
        axiom_state = {
            'coordinates': {
                'determination': float(final_coords[0]),
                'stability': float(final_coords[1]),
                'entropy': float(final_coords[2])
            },
            'quantum_coherence': total_quantum_coherence,
            'word_quantum_states': word_quantum_states,
            'entanglements': entanglements,
            'context': 'morphosyntax_analysis',
            'text': text,
            # Dodane wyzwalacze aksjomatów (heurystycznie lub testowo)
            'operation_result': [float(final_coords[0]), float(final_coords[1]), float(final_coords[2])],
            'knowledge_claims': ["Wiem, że..." if "wiem" in text.lower() else ""],
            'representation': 1.0 if any(x in text.lower() for x in ["reprezentacja", "przedstawia"]) else 0.5,
            'trajectory': [np.array([float(final_coords[0]), float(final_coords[1]), float(final_coords[2])]), np.array([0.9,0.9,0.1]), np.array([0.8,0.8,0.2]), np.array([0.7,0.7,0.3])],
            'operation': 'add' if "+" in text else 'none',
            'meta_operation': 'psi_gtmo' if "meta" in text.lower() else 'none',
        }
        
        # Apply axiom system
        axiom_result = self.axiom_system.execute_all_axioms(axiom_state)
        
        # Extract final coordinates after axiom intervention
        if 'coordinates' in axiom_result:
            final_coords = np.array([
                axiom_result['coordinates']['determination'],
                axiom_result['coordinates']['stability'],
                axiom_result['coordinates']['entropy']
            ])
        
        # Prepare result
        timestamp = datetime.now()
        
        result = {
            "version": "2.0",
            "analysis_type": "GTMØ",
            "timestamp": timestamp.isoformat(),
            "content": {
                "text": text,
                "length": len(text),
                "word_count": len(text.split())
            },
            "coordinates": {
                "determination": round(float(final_coords[0]), 6),
                "stability": round(float(final_coords[1]), 6),
                "entropy": round(float(final_coords[2]), 6)
            },
            "analysis_metadata": {
                "analyzed_at": timestamp.isoformat(),
                "sequence_number": 0,  # Should be set by caller
                "daily_date": timestamp.strftime("%d%m%Y")
            },
            "additional_metrics": {
                "total_analyses": morph_meta.get('total_analyses', 0),
                "cases": morph_meta.get('cases', {}),
                "pos": morph_meta.get('pos', {}),
                "ambiguity": morph_meta.get('ambiguity', 1.0),
                "debug_tags": morph_meta.get('debug_tags', []),
                "coords_count": morph_meta.get('coords_count', 0)
            },
            "interpretation": self._classify_interpretation({
                "determination": final_coords[0],
                "stability": final_coords[1],
                "entropy": final_coords[2]
            })
        }
        
        # Add explicit top-level depth and ambiguity for convenience
        try:
            result["depth"] = synt_meta.get('max_depth', 0)
            result["ambiguity"] = morph_meta.get('ambiguity', 1.0)
        except Exception:
            # keep result intact if metadata missing
            pass

        # Add source file info if provided
        if source_file:
            result["source_file"] = source_file
        
        # Add quantum metrics
        result["quantum_metrics"] = {
            "total_coherence": round(float(total_quantum_coherence), 4),
            "quantum_words": len(word_quantum_states),
            "entanglements": len(entanglements),
            "superposition_type": self._classify_superposition_type(word_quantum_states)
        }
        
        # Add axiom summary
        if 'axiom_execution_summary' in axiom_result:
            result["axiom_protection"] = axiom_result['axiom_execution_summary']
        
        # Add topological classification if present
        if 'topological_classification' in axiom_result:
            result["topology"] = axiom_result['topological_classification']

        # Add temporal analysis
        result["temporal_analysis"] = {
            "coordinates": {
                "determination": round(float(temporal_coords[0]), 4),
                "stability": round(float(temporal_coords[1]), 4),
                "entropy": round(float(temporal_coords[2]), 4)
            },
            "tenses": temporal_meta.get('tenses', {}),
            "aspects": temporal_meta.get('aspects', {}),
            "dominant_time": temporal_meta.get('dominant_time', None)
        }

        # Add extended rhetorical analysis
        result["extended_rhetorical_analysis"] = {
            "mode": rhetorical_mode,
            "irony_score": round(rhetorical_metadata_extended.get('irony_score', 0.0), 4),
            "paradox_score": round(rhetorical_metadata_extended.get('paradox_score', 0.0), 4),
            "sarcasm_score": round(rhetorical_metadata_extended.get('sarcasm_score', 0.0), 4),
            "context_score": round(rhetorical_metadata_extended.get('context_score', 0.0), 4),
            "detected_markers": rhetorical_metadata_extended.get('detected_markers', [])
        }

        # Add quantum ambiguity analysis
        result["quantum_ambiguity"] = {
            "needs_superposition": needs_quantum,
            "ambiguity_score": quantum_ambiguity_meta.get('ambiguity_score', 0.0),
            "ambiguity_markers": quantum_ambiguity_meta.get('ambiguity_markers', []),
            "marker_count": quantum_ambiguity_meta.get('marker_count', 0)
        }

        if quantum_superposition_state:
            result["quantum_ambiguity"]["superposition_state"] = {
                "von_neumann_entropy": round(quantum_superposition_state['von_neumann_entropy'], 4),
                "uncertainty": round(quantum_superposition_state['uncertainty'], 4),
                "num_states": len(quantum_superposition_state['states']),
                "probabilities": [round(p, 4) for p in quantum_superposition_state['probabilities']]
            }

        # Add enhanced depth metrics
        result["depth_metrics"] = {
            "max_depth": synt_meta.get('max_depth', 0),
            "avg_depth": round(synt_meta.get('avg_depth', 0.0), 4),
            "depth_variance": round(synt_meta.get('depth_variance', 0.0), 4)
        }

        print(f"  FINAL: D={D:.3f}, S={S:.3f}, E={E:.3f}")
        print(tensor_print)

        # ========================================================================
        # RHETORICAL ANALYSIS: Irony/Paradox Detection
        # ========================================================================
        if self.rhetorical_analyzer:
            try:
                # Prepare syntax coords (use synt_coords directly)
                syntax_coords = synt_coords

                # Perform rhetorical analysis
                transformed_coords, rhetorical_mode, rhetorical_metadata = self.rhetorical_analyzer.analyze_rhetorical_mode(
                    text=text,
                    morph_coords=final_coords,
                    syntax_coords=syntax_coords,
                    morph_metadata=morph_meta,
                    syntax_metadata=synt_meta
                )

                # Add rhetorical analysis to result
                result["rhetorical_analysis"] = {
                    "mode": rhetorical_mode,
                    "irony_score": round(rhetorical_metadata.get('irony_score', 0.0), 4),
                    "paradox_score": round(rhetorical_metadata.get('paradox_score', 0.0), 4),
                    "structural_divergence": round(rhetorical_metadata.get('structural_divergence', 0.0), 4),
                    "pos_anomalies": {
                        "adj_ratio": round(rhetorical_metadata.get('pos_anomalies', {}).get('adj_ratio', 0.0), 4),
                        "verb_ratio": round(rhetorical_metadata.get('pos_anomalies', {}).get('verb_ratio', 0.0), 4),
                        "anomaly_score": round(rhetorical_metadata.get('pos_anomalies', {}).get('anomaly_score', 0.0), 4)
                    }
                }

                # If irony detected, show coordinate transformation
                if rhetorical_mode == 'irony':
                    result["rhetorical_analysis"]["coordinate_inversion"] = {
                        "original": {
                            "determination": round(float(final_coords[0]), 4),
                            "stability": round(float(final_coords[1]), 4),
                            "entropy": round(float(final_coords[2]), 4)
                        },
                        "inverted": {
                            "determination": round(float(transformed_coords[0]), 4),
                            "stability": round(float(transformed_coords[1]), 4),
                            "entropy": round(float(transformed_coords[2]), 4)
                        }
                    }
                    result["rhetorical_analysis"]["irony_indicators"] = rhetorical_metadata.get('irony_analysis', {}).get('irony_indicators', [])
                    print(f"  🎭 IRONY DETECTED: score={rhetorical_metadata.get('irony_score', 0.0):.2f}")
                    print(f"     Coordinates INVERTED: D={transformed_coords[0]:.3f}, S={transformed_coords[1]:.3f}, E={transformed_coords[2]:.3f}")

                # If paradox detected, show details
                elif rhetorical_mode == 'paradox':
                    result["rhetorical_analysis"]["paradox_indicators"] = rhetorical_metadata.get('paradox_analysis', {}).get('paradox_indicators', [])
                    result["rhetorical_analysis"]["symmetry_score"] = round(rhetorical_metadata.get('paradox_analysis', {}).get('symmetry_score', 0.0), 4)
                    print(f"  ⚖️ PARADOX DETECTED: score={rhetorical_metadata.get('paradox_score', 0.0):.2f}")

                else:
                    print(f"  📝 LITERAL MODE (no rhetorical transformation)")

            except Exception as e:
                print(f"  ⚠️ Rhetorical analysis failed: {e}")
                result["rhetorical_analysis"] = {
                    "mode": "error",
                    "error": str(e)
                }

        # ========================================================================
        # CONSTITUTIONAL METRICS: Complete CD-CI Duality Implementation
        # ========================================================================
        # Using dedicated ConstitutionalDualityCalculator class

        if self.constitutional_calculator:
            ambiguity = morph_meta.get('ambiguity', 1.0)
            depth = synt_meta.get('max_depth', 1)

            # Calculate metrics using dedicated calculator
            const_metrics = self.constitutional_calculator.calculate_metrics(
                ambiguity=ambiguity,
                depth=depth,
                D=D,
                S=S,
                E=E,
                inflectional_forms_count=morph_meta.get('total_analyses')
            )

            # Print summary
            print(f"  CONSTITUTIONAL_DEFINITENESS: {const_metrics.CD:.4f}")
            print(f"  CONSTITUTIONAL_INDEFINITENESS: {const_metrics.CI:.4f}")
            print(f"  DUALITY_CHECK: CI × CD = {const_metrics.duality_product:.4f} ≈ Depth² = {const_metrics.duality_theoretical} (error: {const_metrics.duality_error:.2%})")
            print(f"  SEMANTIC_ACCESSIBILITY: {const_metrics.SA:.4f} ({const_metrics.SA*100:.1f}% - {const_metrics.sa_category.value})")
            print(f"  CI_DECOMPOSITION: Morphological={const_metrics.CI_morphological:.2f}, Syntactic={const_metrics.CI_syntactic:.2f}, Semantic={const_metrics.CI_semantic:.2f}")
            print(f"  CLASSIFICATION: {const_metrics.structure_classification.value} (CD/CI = {const_metrics.cd_ci_ratio:.4f})")

            # Dodaj tensor do JSON
            result["quantum_tensor"] = {
                "value": round(T_quantum, 6),
                "formula": f"{D:.3f} × {S:.3f} × {1-E:.3f} = {T_quantum:.3f}"
            }

            # Dodaj Constitutional Metrics do JSON using dedicated calculator
            result["constitutional_metrics"] = const_metrics.to_dict()

            # Jawne czynniki geometryczne na poziomie zdania
            try:
                result["geometric_balance"] = round(float(const_metrics.geometric_balance), 6)
                result["geometric_tension"] = round(float(const_metrics.geometric_tension), 6)
            except Exception:
                pass

            # Mark critical blocks with low semantic accessibility
            try:
                if const_metrics.SA < 0.3:
                    result["critical_block"] = True
                    result["critical_reason"] = "LOW_SA"
            except Exception:
                pass

        else:
            # Fallback gdy Constitutional Calculator niedostępny
            print("  ⚠️ Constitutional Duality Calculator not available - skipping CI-CD metrics")

            # Dodaj tensor do JSON
            result["quantum_tensor"] = {
                "value": round(T_quantum, 6),
                "formula": f"{D:.3f} × {S:.3f} × {1-E:.3f} = {T_quantum:.3f}"
            }

            result["constitutional_metrics"] = {
                "error": "ConstitutionalDualityCalculator not available"
            }

        return result


# ==================================================
# UTILITY FUNCTIONS
# ==================================================

def load_markdown_file(file_path: str) -> List[str]:
    """
    Load and parse a markdown file into sentences.
    
    Args:
        file_path: Path to the markdown file
        
    Returns:
        List of sentences from the file
    """
    import re
    
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        # Remove markdown formatting
        content = re.sub(r'#+ ', '', content)  # Remove headers
        content = re.sub(r'\*\*(.*?)\*\*', r'\1', content)  # Remove bold
        content = re.sub(r'\*(.*?)\*', r'\1', content)  # Remove italic
        content = re.sub(r'\[(.*?)\]\(.*?\)', r'\1', content)  # Remove links
        content = re.sub(r'`(.*?)`', r'\1', content)  # Remove code
        
        # Split into sentences using spaCy if available
        if nlp:
            doc = nlp(content)
            sentences = [sent.text.strip() for sent in doc.sents if sent.text.strip()]
        else:
            # Fallback: simple sentence splitting
            sentences = re.split(r'[.!?]+', content)
            sentences = [s.strip() for s in sentences if s.strip()]
        
        # Filter out very short sentences (less than 10 characters)
        sentences = [s for s in sentences if len(s) >= 10]
        
        print(f"📄 Loaded {len(sentences)} sentences from {file_path}")
        return sentences
        
    except Exception as e:
        print(f"❌ Error loading file {file_path}: {e}")
        return []

# ==================================================
# MAIN ANALYSIS FUNCTIONS
# ==================================================

def analyze_quantum_with_axioms(text: str, source_file: str = "unknown") -> Dict:
    """
    Główna funkcja analizy z aksjomatami GTMØ
    """
    engine = QuantumMorphosyntaxEngine()
    return engine.gtmo_analyze_quantum(text, source_file)

def batch_analyze_quantum_with_axioms(texts: List[str], source_file: str = "batch") -> List[Dict]:
    """
    Analiza wsadowa z aksjomatami GTMØ
    """
    engine = QuantumMorphosyntaxEngine()
    results = []
    
    for i, text in enumerate(texts):
        source_info = f"{source_file}_sentence_{i+1}"
        result = engine.gtmo_analyze_quantum(text, source_info)
        results.append(result)
    
    return results

# ==================================================
# TEST IMPLEMENTATION
# ==================================================

if __name__ == "__main__":
    import json
    import sys
    import os
    from gtmo_json_saver import GTMOOptimizedSaver
    
    # Check if file argument provided
    if len(sys.argv) > 1:
        file_path = sys.argv[1]
        
        if not os.path.exists(file_path):
            print(f"❌ File not found: {file_path}")
            sys.exit(1)
        
        print(f"🔍 Loading file: {file_path}")
        
        try:
            # Load markdown file
            sentences = load_markdown_file(file_path)
            
            if not sentences:
                print("❌ No sentences found in file")
                sys.exit(1)
            
            print(f"📄 Found {len(sentences)} sentences")
            print("🌟 Starting GTMØ Quantum Analysis...")
            print("=" * 70)
            
            if not morfeusz or not nlp:
                print("❌ Missing required components. Install:")
                print("pip install morfeusz2 spacy")
                print("python -m spacy download pl_core_news_lg")
                sys.exit(1)
            
            # Initialize saver
            saver = GTMOOptimizedSaver()

            # Create analysis folder for this document
            analysis_folder = saver.create_analysis_folder(os.path.basename(file_path))
            print(f"📁 Created analysis folder: {analysis_folder}")

            # Store all sentence analyses for full document
            sentence_analyses = []

            # Analyze each sentence and save individually
            for i, sentence in enumerate(sentences, 1):
                print(f"\n🌌 Analyzing sentence {i}/{len(sentences)}")
                print(f"Text: {sentence[:60]}{'...' if len(sentence) > 60 else ''}")

                try:
                    # Analyze single sentence
                    result = analyze_quantum_with_axioms(sentence, os.path.basename(file_path))
                    result["sentence_number"] = i
                    result["total_sentences"] = len(sentences)

                    # Save individual sentence result
                    saved_file = saver.save_sentence_analysis(result, sentence, i)
                    print(f"✅ Saved sentence to: {saved_file}")

                    # Store for full document analysis
                    sentence_analyses.append(result)

                except Exception as e:
                    print(f"❌ Error analyzing sentence {i}: {e}")
                    continue

            # Save full document analysis
            if sentence_analyses:
                try:
                    full_doc_file = saver.save_full_document_analysis(
                        source_file=file_path,
                        sentences=sentences,
                        sentence_analyses=sentence_analyses
                    )
                    print(f"\n📄 Saved full document analysis to: {full_doc_file}")
                except Exception as e:
                    print(f"❌ Error saving full document: {e}")

            print(f"\n🎯 Analysis complete! Check '{analysis_folder}' for results.")
            
        except Exception as e:
            print(f"❌ Error processing file: {e}")
            sys.exit(1)
    
    else:
        # Test basic functionality (original code)
        test_texts = [
            "Rzeczpospolita Polska przestrzega wiążącego ją prawa międzynarodowego.",
            "Dzień był letni i świąteczny. Wszystko na świecie jaśniało, kwitło, pachniało, śpiewało.",
            "Badania lingwistyczne nad entropią językową.",
            "Kocham cię bardzo mocno i tak samo nienawidzę!",
            "To zdanie nie istnieje.",
            "Świnia to ptak, a świnia to ssak.",
        ]
        
        print("GTMØ QUANTUM MORPHOSYNTAX ENGINE WITH AXIOMS - Test Run")
        print("=" * 70)
        
        if not morfeusz or not nlp:
            print("Missing required components. Install:")
            print("pip install morfeusz2 spacy")
            print("python -m spacy download pl_core_news_lg")
        else:
            results = batch_analyze_quantum_with_axioms(test_texts, "test_file.md")
            
            print("\n📊 RESULTS (JSON format):")
            print("=" * 70)
            for r in results:
                print(json.dumps(r, indent=2, ensure_ascii=False))
                print("-" * 70)
