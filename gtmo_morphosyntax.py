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

# Import file loader for markdown parsing
try:
    from gtmo_file_loader import load_markdown_file
    FILE_LOADER_AVAILABLE = True
except ImportError:
    FILE_LOADER_AVAILABLE = False
    # Fallback will be defined below if needed

# Import topological attractors module
try:
    from gtmo_topological_attractors import (
        TopologicalAttractorAnalyzer,
        TemporalEvolutionAnalyzer,
        analyze_topological_context
    )
    TOPOLOGICAL_ATTRACTORS_AVAILABLE = True
except ImportError:
    TOPOLOGICAL_ATTRACTORS_AVAILABLE = False
    # print("✗ GTMØ Topological Attractors not available")

# Import enhanced quantum metrics module
try:
    from gtmo_quantum_enhanced import (
        EnhancedQuantumAnalyzer,
        analyze_quantum_enhanced
    )
    QUANTUM_ENHANCED_AVAILABLE = True
except ImportError:
    QUANTUM_ENHANCED_AVAILABLE = False
    # print("✗ GTMØ Enhanced Quantum Metrics not available")

# Import HerBERT for semantic embeddings and sentence-BERT for coherence
try:
    from transformers import AutoTokenizer, AutoModel
    import torch
    from sentence_transformers import SentenceTransformer
    HERBERT_MODEL_NAME = "allegro/herbert-base-cased"
    herbert_tokenizer = AutoTokenizer.from_pretrained(HERBERT_MODEL_NAME)
    herbert_model = AutoModel.from_pretrained(HERBERT_MODEL_NAME)
    herbert_model.eval()
    HERBERT_AVAILABLE = True
    TRANSFORMERS_AVAILABLE = True
    # Global HerBERT instances to avoid reloading
    GLOBAL_HERBERT_TOKENIZER = herbert_tokenizer
    GLOBAL_HERBERT_MODEL = herbert_model
    print("✔ HerBERT loaded (global instance)")
except ImportError:
    herbert_tokenizer = None
    herbert_model = None
    HERBERT_AVAILABLE = False
    TRANSFORMERS_AVAILABLE = False
    GLOBAL_HERBERT_TOKENIZER = None
    GLOBAL_HERBERT_MODEL = None
    print("✗ HerBERT/sentence-transformers not available: pip install transformers torch sentence-transformers")

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
    'nom': np.array([0.849, 0.271, 0.455]),  # mianownik
    'gen': np.array([0.787, 0.270, 0.456]),  # dopełniacz
    'dat': np.array([0.773, 0.357, 0.456]),  # celownik
    'acc': np.array([0.836, 0.336, 0.450]),  # biernik
    'ins': np.array([0.708, 0.354, 0.468]),  # narzędnik
    'loc': np.array([0.728, 0.282, 0.456]),  # miejscownik
    'voc': np.array([0.683, 0.368, 0.458])   # wołacz
}

# GTMØ coordinates for Polish POS tags
POS_COORDS = {
    'subst': np.array([0.804, 0.477, 0.483]),  # rzeczownik
    'adj': np.array([0.747, 0.342, 0.477]),    # przymiotnik
    'verb': np.array([0.763, 0.351, 0.478]),   # czasownik
    'adv': np.array([0.732, 0.383, 0.481]),    # przysłówek
    'num': np.array([0.835, 0.422, 0.486]),    # liczebnik
    'pron': np.array([0.712, 0.453, 0.484]),   # zaimek
    'prep': np.array([0.76, 0.75, 0.24]),      # przyimek
    'conj': np.array([0.65, 0.85, 0.20]),      # spójnik
    'part': np.array([0.40, 0.26, 0.84]),      # partykuła
    'interp': np.array([0.95, 0.95, 0.05])     # interpunkcja
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
        try:
            if state is None or 'coordinates' not in state:
                raise ValueError("State is None or does not contain 'coordinates' key")
            
            coords = self._extract_coordinates(state)
            if coords is None:
                raise ValueError("State does not contain valid coordinates")
                
            if 'operation' in state:
                distance = np.linalg.norm(coords - SINGULARITY_COORDS)
                if distance < 0.05:
                    blocked_ops = ['add', 'subtract', 'multiply', 'divide']
                    if any(op in str(state['operation']).lower() for op in blocked_ops):
                        self._activate_axiom("AX9", f"Blocking operator near Ø")
                        state['operation_result'] = {'type': 'irreducible'}
                        raise SingularityError(f"AX9: Operator cannot act on Ø")
        except Exception as e:
            print(f"AX9: Unhandled exception: {str(e)}")
        
        return state
    
    def _ax10_meta_operator_definition(self, state: Dict) -> Dict:
        """Allow only meta-operators near Singularity."""
        if state is None or 'coordinates' not in state:
            raise ValueError("State is None or does not contain 'coordinates' key")
        
        coords = self._extract_coordinates(state)
        if coords is None:
            raise ValueError("State does not contain valid coordinates")
        
        distance = np.linalg.norm(coords - SINGULARITY_COORDS)
        if distance < 0.05 and 'meta_operation' in state:
            allowed_meta_ops = ['psi_gtmo', 'e_gtmo', 'topology_classification']
            if any(allowed in str(state['meta_operation']).lower() for allowed in allowed_meta_ops):
                self._activate_axiom("AX10", f"Allowing meta-operator near Ø")
            else:
                raise ValueError(f"AX10: Only meta-operators can act near Ø, but {state['meta_operation']} was found")
        
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

    def __init__(self, domain_dictionary: Optional['DomainDictionary'] = None,
                 herbert_tokenizer=None, herbert_model=None):
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

        # Initialize constitutional duality calculator with shared HerBERT
        if CONSTITUTIONAL_DUALITY_AVAILABLE:
            self.constitutional_calculator = ConstitutionalDualityCalculator(
                use_sa_v3=True,
                herbert_tokenizer=herbert_tokenizer,
                herbert_model=herbert_model
            )
        else:
            self.constitutional_calculator = None

        # Initialize quantum ambiguity analyzer
        self.quantum_ambiguity_analyzer = QuantumAmbiguityAnalyzer()

        # Initialize topological attractors analyzer
        if TOPOLOGICAL_ATTRACTORS_AVAILABLE:
            self.topological_analyzer = TopologicalAttractorAnalyzer()
            self.temporal_evolution = TemporalEvolutionAnalyzer(history_size=100)
        else:
            self.topological_analyzer = None
            self.temporal_evolution = None

        # Initialize enhanced quantum analyzer
        if QUANTUM_ENHANCED_AVAILABLE:
            self.quantum_enhanced = EnhancedQuantumAnalyzer()
        else:
            self.quantum_enhanced = None

        # Use shared HerBERT models instead of lazy loading
        self._herbert_tokenizer = herbert_tokenizer
        self._herbert_model = herbert_model
        self._sentence_model = None

    # =========================================================================
    # FIXED: HELPER METHODS FOR ENTROPY MEASUREMENT
    # =========================================================================

    def _load_herbert_model(self):
        """Get HerBERT model (uses shared instance if available)."""
        if not TRANSFORMERS_AVAILABLE:
            return None, None

        # Use already loaded models (passed in constructor or global)
        if self._herbert_tokenizer is not None and self._herbert_model is not None:
            return self._herbert_tokenizer, self._herbert_model

        # Fallback: try global instances
        if GLOBAL_HERBERT_TOKENIZER is not None and GLOBAL_HERBERT_MODEL is not None:
            self._herbert_tokenizer = GLOBAL_HERBERT_TOKENIZER
            self._herbert_model = GLOBAL_HERBERT_MODEL
            return self._herbert_tokenizer, self._herbert_model

        # Last resort: load new (should not happen if global loading succeeded)
        from transformers import AutoTokenizer, AutoModel
        print("WARNING: Loading new HerBERT instance (global instance not found)")
        self._herbert_tokenizer = AutoTokenizer.from_pretrained("allegro/herbert-base-cased")
        self._herbert_model = AutoModel.from_pretrained("allegro/herbert-base-cased")
        self._herbert_model.eval()

        return self._herbert_tokenizer, self._herbert_model

    def _load_sentence_model(self):
        """Lazy load sentence-BERT model for coherence measurement."""
        if not TRANSFORMERS_AVAILABLE:
            return None

        if self._sentence_model is None:
            from sentence_transformers import SentenceTransformer
            # Use multilingual sentence-BERT that supports Polish
            self._sentence_model = SentenceTransformer('paraphrase-multilingual-mpnet-base-v2')

        return self._sentence_model

    def _estimate_disambiguation_confidence(self, spacy_token, num_analyses: int) -> float:
        """
        Estimate confidence of disambiguation based on syntactic context.

        High confidence (0.9+): Content words with clear role (nsubj, obj, ROOT)
        Medium confidence (0.7-0.9): Modifiers, adverbs
        Low confidence (0.5-0.7): Function words, ambiguous attachments

        Returns:
            float [0,1]: disambiguation confidence
        """
        # Start with base confidence inversely proportional to ambiguity
        if num_analyses == 1:
            return 1.0
        elif num_analyses == 2:
            base_conf = 0.8
        elif num_analyses <= 4:
            base_conf = 0.6
        else:
            base_conf = 0.4

        # Adjust based on dependency role
        high_confidence_roles = ['ROOT', 'nsubj', 'obj', 'iobj', 'obl', 'aux']
        medium_confidence_roles = ['amod', 'advmod', 'nmod', 'acl']

        if spacy_token.dep_ in high_confidence_roles:
            role_boost = 0.2
        elif spacy_token.dep_ in medium_confidence_roles:
            role_boost = 0.1
        else:
            role_boost = 0.0

        # Adjust based on POS
        if spacy_token.pos_ in ['NOUN', 'VERB', 'PROPN']:
            pos_boost = 0.1
        else:
            pos_boost = 0.0

        confidence = base_conf + role_boost + pos_boost
        return np.clip(confidence, 0.0, 1.0)

    def disambiguate_morfeusz(self, word: str, analyses_list: List, sentence_context: str) -> Tuple:
        """
        Select SINGLE best interpretation using context-aware disambiguation.

        Args:
            word: Word form to disambiguate
            analyses_list: All possible Morfeusz2 interpretations
            sentence_context: Full sentence for context

        Returns:
            Single best analysis tuple
        """
        if len(analyses_list) == 1:
            return analyses_list[0]

        # Strategy 1: Use spaCy POS tagging for disambiguation
        if nlp:
            try:
                doc = nlp(sentence_context)
                for token in doc:
                    if token.text == word:
                        spacy_pos = token.pos_

                        # Map spaCy POS to Morfeusz tags
                        pos_mapping = {
                            'NOUN': 'subst', 'ADJ': 'adj', 'VERB': 'verb',
                            'ADV': 'adv', 'NUM': 'num', 'PRON': 'pron',
                            'ADP': 'prep', 'CONJ': 'conj', 'CCONJ': 'conj',
                            'PART': 'part', 'PUNCT': 'interp'
                        }

                        target_pos = pos_mapping.get(spacy_pos)
                        if target_pos:
                            for analysis in analyses_list:
                                _, _, _, lemma, tag, _, _ = analysis
                                if tag.startswith(target_pos):
                                    return analysis
                        break
            except:
                pass

        # Strategy 2: Use domain dictionary if available
        if self.use_domain_filtering and self.domain_dictionary:
            domain_scores = []
            for analysis in analyses_list:
                _, _, _, lemma, tag, _, _ = analysis
                score = 0

                if self.domain_dictionary.is_domain_term(lemma):
                    score += 10

                domain_tags = self.domain_dictionary.get_domain_tags_for_word(lemma)
                if tag in domain_tags:
                    score += 20

                domain_scores.append(score)

            if max(domain_scores) > 0:
                best_idx = domain_scores.index(max(domain_scores))
                return analyses_list[best_idx]

        # Strategy 3: Frequency heuristic - prefer most common POS
        # In Polish: subst > verb > adj > adv > others
        pos_priority = ['subst', 'verb', 'adj', 'adv', 'num', 'pron', 'prep', 'conj', 'part']

        for preferred_pos in pos_priority:
            for analysis in analyses_list:
                _, _, _, lemma, tag, _, _ = analysis
                if tag.startswith(preferred_pos):
                    return analysis

        # Fallback: return first analysis
        return analyses_list[0]

    # =========================================================================
    # FIXED: ENTROPY MEASUREMENT METHODS (from gtmo_morphosyntax_FIXED.py)
    # =========================================================================

    def calculate_polysemy_score(self, doc) -> float:
        """
        Measure polysemy (words with multiple meanings) using contextual embeddings.

        FIX: Context-aware polysemy measurement.

        For legal/technical text, terms are monosemous WITHIN their domain context.
        "powód" has multiple meanings in general Polish, but ONE meaning in legal context.

        Strategy:
        1. For each content word, check if it's domain-specific (legal, technical)
        2. Domain-specific words get low polysemy score (monosemous in context)
        3. General words measured by embedding variance across sentences
        4. Legal text should score ~0.3-0.5, not 0.8

        Returns:
            float [0,1]: 0 = monosemous in context, 1 = highly polysemous
        """
        tokenizer, model = self._load_herbert_model()

        if model is None:
            return self._calculate_polysemy_heuristic(doc)

        # Identify content words
        content_words = [t for t in doc if t.pos_ in ['NOUN', 'VERB', 'ADJ', 'ADV']]

        if not content_words:
            return 0.0

        # Legal/technical term markers (monosemous in domain)
        legal_markers = {
            'wyrok', 'sad', 'pozwany', 'powod', 'zasadza', 'orzeka',
            'postanowienie', 'sprawa', 'sygn', 'akt', 'rozpoznanie'
        }

        polysemy_scores = []

        # Sample max 15 content words for performance
        sampled_words = content_words[:15]

        for token in sampled_words:
            word_lemma = token.lemma_.lower()

            # CRITICAL FIX: Domain-specific terms are monosemous in context
            if word_lemma in legal_markers:
                polysemy_scores.append(0.1)  # Low polysemy for legal terms
                continue

            # For general words, measure contextual variance
            try:
                sent_text = token.sent.text
                with torch.no_grad():
                    inputs = tokenizer(sent_text, return_tensors="pt", truncation=True, max_length=128)
                    outputs = model(**inputs)
                    emb_in_context = outputs.last_hidden_state.mean(dim=1).squeeze().numpy()

                    # Compare with embedding of word alone (general meaning)
                    inputs_general = tokenizer(token.text, return_tensors="pt", truncation=True, max_length=128)
                    outputs_general = model(**inputs_general)
                    emb_general = outputs_general.last_hidden_state.mean(dim=1).squeeze().numpy()

                # Cosine distance
                norm1 = np.linalg.norm(emb_in_context)
                norm2 = np.linalg.norm(emb_general)

                if norm1 > 1e-10 and norm2 > 1e-10:
                    cos_sim = np.dot(emb_in_context, emb_general) / (norm1 * norm2)
                    contextual_shift = 1.0 - cos_sim
                    polysemy_score = np.clip(contextual_shift / 0.5, 0, 1)
                    polysemy_scores.append(polysemy_score)
                else:
                    polysemy_scores.append(0.3)

            except Exception:
                polysemy_scores.append(0.3)  # Default moderate

        if not polysemy_scores:
            return 0.3

        mean_polysemy = np.mean(polysemy_scores)
        return np.clip(mean_polysemy, 0, 1)

    def _calculate_polysemy_heuristic(self, doc) -> float:
        """
        Fallback polysemy estimation without HerBERT.

        Uses linguistic heuristics:
        - Short common words tend to be polysemous (e.g., "rzecz", "sprawa")
        - Long technical words tend to be monosemous
        """
        content_words = [t for t in doc if t.pos_ in ['NOUN', 'VERB', 'ADJ', 'ADV']]

        if not content_words:
            return 0.3

        polysemy_indicators = []

        for token in content_words:
            length_score = max(0, 1.0 - len(token.text) / 15.0)

            if token.pos_ == 'VERB':
                pos_score = 0.7
            elif token.pos_ == 'NOUN':
                pos_score = 0.5
            else:
                pos_score = 0.3

            polysemy_indicators.append(0.6 * length_score + 0.4 * pos_score)

        return np.clip(np.mean(polysemy_indicators), 0, 1)

    def calculate_syntactic_ambiguity(self, doc) -> float:
        """
        Measure syntactic ambiguity from parse tree.

        Indicators of syntactic ambiguity:
        1. Multiple valid PP-attachment sites
        2. Coordination ambiguity (what does "and" connect?)
        3. Long-distance dependencies
        4. Complex subordination

        Returns:
            float [0,1]: 0 = unambiguous syntax, 1 = highly ambiguous
        """
        ambiguity_scores = []

        # 1. PP-attachment ambiguity
        prep_phrases = [t for t in doc if t.pos_ == 'ADP']
        for prep in prep_phrases:
            distance = abs(prep.i - prep.head.i)
            ambig_score = min(distance / 10.0, 1.0)
            ambiguity_scores.append(ambig_score)

        # 2. Coordination ambiguity
        coord_conj = [t for t in doc if t.dep_ == 'conj']
        for conj in coord_conj:
            children = list(conj.children)
            ambig_score = min(len(children) / 5.0, 1.0)
            ambiguity_scores.append(ambig_score)

        # 3. Long-distance dependencies
        for token in doc:
            if token.dep_ in ['csubj', 'ccomp', 'xcomp', 'advcl']:
                distance = abs(token.i - token.head.i)
                ambig_score = min(distance / 15.0, 1.0)
                ambiguity_scores.append(ambig_score)

        # 4. Clause complexity
        sentences = list(doc.sents)
        for sent in sentences:
            subordinates = [t for t in sent if t.dep_ in ['csubj', 'ccomp', 'advcl']]
            clause_ambig = min(len(subordinates) / 3.0, 1.0)
            ambiguity_scores.append(clause_ambig)

        if not ambiguity_scores:
            return 0.2  # Default low ambiguity

        return np.clip(np.mean(ambiguity_scores), 0, 1)

    def calculate_coherence_score(self, doc) -> float:
        """
        Measure inter-sentence coherence using LOGICAL FLOW indicators.

        FIX: Legal text has structural coherence, not just thematic similarity.

        Indicators of logical coherence:
        1. Logical connectors (dlatego, wobec tego, jednak)
        2. Anaphoric references (pozwana, ona, strona → refers back)
        3. Information progression (new info builds on old)
        4. Semantic similarity (sentence-BERT as secondary measure)

        Returns:
            float [0,1]: 0 = random sentences, 1 = logically coherent
        """
        sentences = list(doc.sents)

        if len(sentences) < 2:
            return 1.0  # Single sentence is maximally coherent

        # Component 1: Logical connectors (30% weight)
        logical_connector_score = self._measure_logical_connectors(sentences)

        # Component 2: Anaphoric coherence (30% weight)
        anaphoric_score = self._measure_anaphoric_coherence(sentences)

        # Component 3: Lexical cohesion (20% weight)
        lexical_score = self._measure_lexical_cohesion(doc, sentences)

        # Component 4: Semantic similarity via sentence-BERT (20% weight)
        semantic_score = self._measure_semantic_coherence(sentences)

        # ADAPTIVE: For short texts (<10 sentences), use formulaic structure detection
        num_sentences = len(sentences)
        if num_sentences < 10:
            coherence = (0.10 * logical_connector_score +
                        0.15 * anaphoric_score +
                        0.25 * lexical_score +
                        0.50 * semantic_score)
        else:
            coherence = (0.30 * logical_connector_score +
                        0.30 * anaphoric_score +
                        0.20 * lexical_score +
                        0.20 * semantic_score)

        return np.clip(coherence, 0, 1)

    def _measure_logical_connectors(self, sentences) -> float:
        """Measure presence of logical connectors between sentences."""
        logical_connectors = {
            'dlatego', 'wobec', 'zatem', 'wiec', 'stad',
            'jednak', 'natomiast', 'ale', 'lecz', 'mimo',
            'nastepnie', 'ponadto', 'dodatkowo', 'rowniez',
            'ostatecznie', 'podsumowujac', 'konczac'
        }

        connector_count = 0
        for sent in sentences:
            sent_text_lower = sent.text.lower()
            if any(conn in sent_text_lower for conn in logical_connectors):
                connector_count += 1

        score = min(connector_count / (len(sentences) * 0.4), 1.0)
        return score

    def _measure_anaphoric_coherence(self, sentences) -> float:
        """Measure anaphoric references (pronouns/definite NPs referring back)."""
        anaphora_indicators = 0
        total_sentences = len(sentences)

        for i, sent in enumerate(sentences):
            if i == 0:
                continue  # First sentence can't have anaphora

            pronouns = [t for t in sent if t.pos_ == 'PRON']
            anaphora_indicators += len(pronouns)

            definite_markers = ['pozwany', 'pozwana', 'powod', 'sad', 'strona']
            sent_lemmas = [t.lemma_.lower() for t in sent]
            if any(marker in sent_lemmas for marker in definite_markers):
                anaphora_indicators += 1

        expected_anaphora = (total_sentences - 1) * 1.5
        score = min(anaphora_indicators / expected_anaphora, 1.0) if expected_anaphora > 0 else 0.5

        return score

    def _measure_lexical_cohesion(self, doc, sentences) -> float:
        """Measure lexical overlap between consecutive sentences."""
        if len(sentences) < 2:
            return 1.0

        overlaps = []

        for i in range(len(sentences) - 1):
            sent1_lemmas = set(t.lemma_.lower() for t in sentences[i]
                             if t.pos_ in ['NOUN', 'VERB', 'ADJ'])
            sent2_lemmas = set(t.lemma_.lower() for t in sentences[i + 1]
                             if t.pos_ in ['NOUN', 'VERB', 'ADJ'])

            if sent1_lemmas and sent2_lemmas:
                overlap = len(sent1_lemmas & sent2_lemmas) / min(len(sent1_lemmas), len(sent2_lemmas))
                overlaps.append(overlap)

        if not overlaps:
            return 0.5

        mean_overlap = np.mean(overlaps)
        return np.clip(mean_overlap, 0, 1)

    def _measure_semantic_coherence(self, sentences) -> float:
        """Measure semantic similarity using sentence-BERT (if available)."""
        sentence_model = self._load_sentence_model()

        if sentence_model is None:
            return 0.5  # Neutral fallback

        sentence_texts = [sent.text.strip() for sent in sentences]
        try:
            embeddings = sentence_model.encode(sentence_texts, convert_to_numpy=True)

            similarities = []
            for i in range(len(embeddings) - 1):
                emb1 = embeddings[i]
                emb2 = embeddings[i + 1]

                norm1 = np.linalg.norm(emb1)
                norm2 = np.linalg.norm(emb2)

                if norm1 > 1e-10 and norm2 > 1e-10:
                    cos_sim = np.dot(emb1, emb2) / (norm1 * norm2)
                    similarities.append(cos_sim)

            if similarities:
                return np.clip(np.mean(similarities), 0, 1)
            else:
                return 0.5

        except Exception:
            return 0.5

    def _calculate_E_FIXED(self, text: str) -> float:
        """
        FIXED Entropy calculation using proper semantic measures.

        Uses:
        - Polysemy: words with multiple meanings (40% weight)
        - Syntactic ambiguity: parse tree ambiguity (40% weight)
        - Incoherence: 1 - coherence between sentences (20% weight)

        Formula: E = 0.4×polysemy + 0.4×syntactic_amb + 0.2×(1 - coherence)

        Returns:
            float [0,1]: 0 = perfectly ordered, 1 = maximum chaos
        """
        if not nlp:
            # Fallback if spaCy not available
            return 0.5

        doc = nlp(text)

        # Component 1: Polysemy (semantic ambiguity)
        polysemy = self.calculate_polysemy_score(doc)

        # Component 2: Syntactic ambiguity
        synt_ambig = self.calculate_syntactic_ambiguity(doc)

        # Component 3: Incoherence (inverse of coherence)
        coherence = self.calculate_coherence_score(doc)
        incoherence = 1.0 - coherence

        # Weighted combination
        E_fixed = 0.4 * polysemy + 0.4 * synt_ambig + 0.2 * incoherence

        return np.clip(E_fixed, 0.0, 1.0)

    def extract_true_morphemes(self, analysis: Tuple) -> Dict[str, np.ndarray]:
        """
        Extract TRUE morphological segments (root, suffix, prefix).

        This is real morpheme decomposition, not interpretation variants!

        Args:
            analysis: Selected Morfeusz analysis

        Returns:
            Dictionary with morpheme segments and their D-S-E coordinates
        """
        _, _, form, lemma, _, _, _ = analysis

        morphemes = {}

        # Extract root (lemma is approximation of root)
        root = lemma
        morphemes['root'] = root

        # Extract suffix by comparing form to lemma
        if len(form) > len(lemma) and form.startswith(lemma):
            suffix = form[len(lemma):]
            if suffix:
                morphemes['suffix'] = suffix
        elif form != lemma:
            # Complex morphology - just note the difference
            morphemes['inflection'] = form

        # Assign D-S-E to morphemes (morphology ADDS PRECISION!)
        # CRITICAL: High D MUST correlate with High S (precise → stable)
        morpheme_coords = {}

        # Root: Usually polysemous (moderate D) but lexically stable (moderate S)
        # E is moderate because roots can have multiple senses
        morpheme_coords['root'] = np.array([0.6, 0.65, 0.4])  # D-S correlation maintained

        # Suffix: Grammatical endings are HIGHLY PRECISE AND STABLE
        # Polish inflection is unambiguous once parsed correctly
        # High D + High S + Low E = morphological precision
        if 'suffix' in morphemes:
            morpheme_coords['suffix'] = np.array([0.90, 0.92, 0.10])  # Perfect D-S correlation!

        if 'inflection' in morphemes:
            # Complete inflected form: very precise and stable
            morpheme_coords['inflection'] = np.array([0.88, 0.90, 0.12])  # D-S correlation

        return morpheme_coords

    def tensor_product_composition(self, morpheme_coords: List[np.ndarray]) -> np.ndarray:
        """
        Compose morpheme D-S-E using tensor product, NOT averaging.

        Averaging destroys information. Tensor product preserves compositionality.

        Args:
            morpheme_coords: List of [D, S, E] for each morpheme

        Returns:
            Composed [D, S, E] coordinates
        """
        if not morpheme_coords:
            return np.array([0.5, 0.5, 0.5])

        if len(morpheme_coords) == 1:
            return morpheme_coords[0]

        # Geometric mean preserves proportions without extreme sensitivity
        # This is closer to tensor product composition than arithmetic mean
        composed = np.ones(3)
        for coords in morpheme_coords:
            composed *= coords

        # Take nth root where n = number of morphemes
        composed = np.power(composed, 1.0 / len(morpheme_coords))

        # Ensure constraints: High D MUST correlate with High S
        # Morphological composition increases precision (D) and stability (S)
        # while decreasing entropy (E)

        # Apply composition boost: morphology makes meaning more precise
        precision_boost = 0.05 * (len(morpheme_coords) - 1)
        composed[0] = min(1.0, composed[0] + precision_boost)  # Increase D
        composed[1] = min(1.0, composed[1] + precision_boost)  # Increase S
        composed[2] = max(0.0, composed[2] - precision_boost)  # Decrease E

        # CRITICAL FIX: Enforce D-S correlation
        # If D is high, S must also be high (precise → stable)
        # If D is low, S can be variable
        if composed[0] > 0.7 and composed[1] < composed[0] - 0.1:
            # High D but low S - CONTRADICTION!
            # Boost S to match D (precise things are stable)
            composed[1] = min(1.0, composed[0] - 0.05)

        # Ensure E consistency: High D+S → Low E
        if composed[0] > 0.7 and composed[1] > 0.7 and composed[2] > 0.3:
            # High precision + high stability but high entropy - CONTRADICTION!
            composed[2] = max(0.0, 1.0 - (composed[0] + composed[1]) / 2)

        return composed

    def analyze_morphology_quantum(self, text: str) -> Tuple[np.ndarray, Dict, Dict[str, QuantumSemanticState]]:
        """
        Morphological analysis with PROPER disambiguation.

        CRITICAL FIX:
        - Each word gets ONE disambiguated interpretation, not all variants
        - True morpheme decomposition when needed
        - Tensor product composition, not averaging
        """
        if not morfeusz:
            raise Exception("Morfeusz2 not available")

        word_coords_list = []
        case_counts = {}
        pos_counts = {}
        debug_info = []
        word_quantum_states = {}
        disambiguation_log = []

        try:
            # Get sentence for context
            sentence = text

            analyses = morfeusz.analyse(text)

            # Group analyses by word form
            word_analyses = {}
            for start, end, (form, lemma, tag, labels, qualifiers) in analyses:
                if not form or form.isspace():
                    continue

                if form not in word_analyses:
                    word_analyses[form] = []
                word_analyses[form].append((start, end, form, lemma, tag, labels, qualifiers))

            # Process each word with DISAMBIGUATION (not aggregation!)
            for form, analyses_list in word_analyses.items():
                # CRITICAL: Select ONE best interpretation
                best_analysis = self.disambiguate_morfeusz(form, analyses_list, sentence)

                start, end, form, lemma, tag, labels, qualifiers = best_analysis

                disambiguation_log.append(f"{form} → {lemma}:{tag} (from {len(analyses_list)} options)")
                debug_info.append(f"{lemma}:{tag}")

                tag_parts = tag.split(':')
                main_pos = tag_parts[0]

                # Extract case from chosen interpretation
                word_case = None
                for part in tag_parts:
                    if part in CASE_COORDS:
                        word_case = part
                        case_counts[part] = case_counts.get(part, 0) + 1
                        break
                    elif '.' in part:
                        sub_parts = part.split('.')
                        for sub_part in sub_parts:
                            if sub_part in CASE_COORDS:
                                word_case = sub_part
                                case_counts[sub_part] = case_counts.get(sub_part, 0) + 1
                                break
                        if word_case:
                            break

                # Extract POS
                if main_pos in POS_COORDS:
                    pos_counts[main_pos] = pos_counts.get(main_pos, 0) + 1

                # TRUE morpheme decomposition for this ONE interpretation
                morpheme_coords_dict = self.extract_true_morphemes(best_analysis)
                morpheme_coords_list = list(morpheme_coords_dict.values())

                # Tensor product composition (not averaging!)
                if morpheme_coords_list:
                    word_coords = self.tensor_product_composition(morpheme_coords_list)
                else:
                    # Fallback to case/POS coords
                    if word_case and word_case in CASE_COORDS:
                        word_coords = CASE_COORDS[word_case].copy()
                    elif main_pos in POS_COORDS:
                        word_coords = POS_COORDS[main_pos].copy()
                    else:
                        word_coords = np.array([0.5, 0.5, 0.5])

                word_coords_list.append(word_coords)

                # Create quantum state (NOW based on single interpretation)
                if word_case:
                    quantum_state = self.create_superposition_state(form, {word_case: 1})
                    word_quantum_states[form] = quantum_state

            # Final composition: geometric mean (not arithmetic!)
            if word_coords_list:
                final_coords = self.tensor_product_composition(word_coords_list)
            else:
                final_coords = np.array([0.5, 0.5, 0.5])

            # Calculate morphological ambiguity (average interpretations per word)
            avg_ambiguity = len(analyses) / len(word_analyses) if word_analyses else 1.0

            metadata = {
                'total_analyses': len(analyses),
                'unique_words': len(word_analyses),
                'disambiguations': len(disambiguation_log),
                'cases': case_counts,
                'pos': pos_counts,
                'ambiguity': avg_ambiguity,  # CRITICAL: Required by constitutional calculator
                'ambiguity_ratio': avg_ambiguity,  # Kept for backwards compatibility
                'debug_tags': debug_info[:10],
                'disambiguation_log': disambiguation_log[:5],
                'coords_count': len(word_coords_list),
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
        """
        Create quantum superposition state for word.

        CRITICAL FIX: Phase coherence interpretation
        - coherence = 1.0 means FULLY COHERENT (all phases aligned)
        - coherence = 0.0 means FULLY DECOHERENT (random phases)

        After disambiguation, we have ONE interpretation, so coherence should be HIGH!
        """
        total_observations = sum(case_frequencies.values())
        if total_observations == 0:
            amplitudes = {case: SQRT_2_INV for case in self.case_coords.keys()}
        else:
            amplitudes = {}
            for case, freq in case_frequencies.items():
                probability = freq / total_observations
                # FIXED: After disambiguation, single interpretation → aligned phases
                # Use small phase variation, not random [0, 2π]
                phase_variation = np.random.uniform(-0.1, 0.1)  # Small coherent phase
                amplitude = np.sqrt(probability) * cmath.exp(1j * phase_variation)
                amplitudes[case] = amplitude

        phase = np.angle(sum(amplitudes.values()))

        # Calculate coherence: 1.0 = fully coherent, 0.0 = fully decoherent
        # After disambiguation, we expect HIGH coherence!
        total_amplitude = sum(amplitudes.values())
        total_intensity = sum(abs(amp)**2 for amp in amplitudes.values())

        if total_intensity > 0:
            coherence = abs(total_amplitude)**2 / total_intensity
        else:
            coherence = 0.0

        # CORRECTION: After disambiguation, boost coherence
        # Single interpretation = high coherence (phases aligned)
        if len(case_frequencies) == 1:
            coherence = max(coherence, 0.95)  # Single case → highly coherent

        quantum_state = QuantumSemanticState(
            amplitudes=amplitudes,
            phase=phase,
            coherence=coherence,  # NOW: 1.0 = coherent, 0.0 = decoherent
            entangled_with=[],
            measurement_count=0
        )

        self.quantum_states[word] = quantum_state
        return quantum_state
    
    def collapse_superposition(self, word: str, observed_cases: List[str]) -> np.ndarray:
        """
        Collapse quantum superposition to classical coordinates.

        FIXED: Use geometric composition instead of arithmetic weighted average.
        """
        if word not in self.quantum_states:
            coords_list = [self.case_coords[case] for case in observed_cases if case in self.case_coords]
            # Use tensor product composition, not mean!
            return self.tensor_product_composition(coords_list) if coords_list else np.array([0.5, 0.5, 0.5])

        quantum_state = self.quantum_states[word]

        # Collect coordinates and their quantum probabilities
        case_coords_with_probs = []

        for case in observed_cases:
            if case in quantum_state.amplitudes and case in self.case_coords:
                probability = abs(quantum_state.amplitudes[case])**2
                case_coords_with_probs.append((self.case_coords[case], probability))

        if case_coords_with_probs:
            # Geometric weighted composition
            # More probable states contribute more, but geometrically not arithmetically
            composed = np.ones(3)
            total_weight = 0

            for coords, prob in case_coords_with_probs:
                # Weight by probability in geometric space
                composed *= np.power(coords, prob)
                total_weight += prob

            # Normalize by total weight
            if total_weight > 0:
                collapsed_coords = np.power(composed, 1.0 / total_weight)
            else:
                collapsed_coords = composed
        else:
            coords_list = [self.case_coords[case] for case in observed_cases if case in self.case_coords]
            collapsed_coords = np.mean(coords_list, axis=0) if coords_list else np.array([0.5, 0.5, 0.5])
        
        quantum_state.measurement_count += 1
        self._apply_decoherence(word)
        
        return collapsed_coords
    
    def _apply_decoherence(self, word: str):
        """
        Apply quantum decoherence over time.

        FIXED: Decoherence REDUCES coherence (toward 0), not increases it!
        - coherence *= (1 - rate) → CORRECT (reduces coherence)
        - But after disambiguation, we should have STABLE high coherence
        """
        if word not in self.quantum_states:
            return

        quantum_state = self.quantum_states[word]

        # Decoherence reduces coherence over measurements
        # BUT: After disambiguation, meaning is stable → minimal decoherence
        if quantum_state.measurement_count < 2:
            # First few measurements: minimal decoherence (meaning is stable after disambiguation)
            decoherence_rate = DECOHERENCE_RATE * 0.1  # 10% of normal rate
        else:
            # Multiple measurements: normal decoherence (semantic drift over time)
            decoherence_rate = DECOHERENCE_RATE

        quantum_state.coherence *= (1 - decoherence_rate)
        quantum_state.coherence = max(quantum_state.coherence, 0.5)  # Minimum coherence after disambiguation

        for case in quantum_state.amplitudes:
            phase_noise = np.random.normal(0, decoherence_rate)
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
        """
        Classify type of quantum superposition in the text.

        FIXED: Coherence interpretation
        - HIGH coherence (>0.8) = COHERENT state (phases aligned, stable meaning)
        - LOW coherence (<0.5) = DECOHERENT state (phases random, unstable meaning)
        """
        if not word_quantum_states:
            return "VACUUM_STATE"

        total_coherence = sum(qs.coherence for qs in word_quantum_states.values())
        avg_coherence = total_coherence / len(word_quantum_states)

        entangled_count = sum(len(qs.entangled_with) for qs in word_quantum_states.values()) / 2

        # CORRECTED: High coherence = COHERENT (not decoherent!)
        if avg_coherence > 0.8:
            if entangled_count > len(word_quantum_states) / 2:
                return "HIGHLY_ENTANGLED_COHERENT"
            else:
                return "COHERENT_SUPERPOSITION"  # Stable, well-defined meaning
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
        print(f"  🔄 Starting morphological analysis...")

        # Run morphological analysis with quantum superposition
        morph_coords, morph_meta, word_quantum_states = self.analyze_morphology_quantum(text)
        print(f"  ✅ Morphological analysis completed")

        print(f"  🔄 Starting syntactic analysis...")
        # Run syntactic analysis with entanglement detection
        synt_coords, synt_meta, entanglements = self.analyze_syntax_quantum(text, word_quantum_states)
        print(f"  ✅ Syntactic analysis completed")

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
        
        # Compute Φ⁹ and attach to result
        try:
            _phi9_arr = F3_to_Phi9(float(final_coords[0]), float(final_coords[1]), float(final_coords[2]))
            result["phi9"] = [float(x) for x in np.asarray(_phi9_arr).flatten().tolist()]
        except Exception:
            pass

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

        # Add HerBERT embedding
        if HERBERT_AVAILABLE:
            try:
                with torch.no_grad():
                    inputs = herbert_tokenizer(text, return_tensors="pt", 
                                              truncation=True, max_length=512, 
                                              padding=True)
                    outputs = herbert_model(**inputs)
                    embedding = outputs.last_hidden_state[:, 0, :].squeeze().numpy()
                result["herbert_embedding"] = embedding.tolist()
                print(f"  🤖 HerBERT embedding: {embedding.shape}")
            except Exception as e:
                print(f"  ⚠️ HerBERT embedding failed: {e}")

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

                # Formal Register Violation Detection (dodane 22.11.2025)
                try:
                    irony_score = rhetorical_metadata.get('irony_score', 0.0)
                    register_violation = self.rhetorical_analyzer.detect_formal_register_violation(
                        text=text,
                        irony_score=irony_score,
                        context_type='legal'
                    )

                    # Add to result
                    result["rhetorical_analysis"]["register_violation"] = register_violation

                    # Override mode if IRRATIONAL_ANOMALY detected
                    if register_violation.get('classification') == 'IRRATIONAL_ANOMALY':
                        result["rhetorical_analysis"]["mode"] = 'IRRATIONAL_ANOMALY'
                        severity = register_violation.get('severity', 'UNKNOWN')
                        anomaly_type = register_violation.get('anomaly_type', 'UNKNOWN')
                        print(f"  🚨 ANOMALY DETECTED: {anomaly_type} (severity: {severity})")

                        if register_violation.get('vulgar_words_found'):
                            print(f"     Vulgar words: {', '.join(register_violation['vulgar_words_found'])}")
                        if register_violation.get('irony_triggered'):
                            print(f"     High irony in legal context: {irony_score:.2f}")

                except Exception as reg_err:
                    print(f"  ⚠️ Register violation detection failed: {reg_err}")

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
                inflectional_forms_count=morph_meta.get('total_analyses'),
                text=text,  # Pass text for SA v3.0 calculation
                kinetic_power_est=0.85  # High kinetic power for legal/formal texts
            )

            # Print summary
            print(f"  CONSTITUTIONAL_DEFINITENESS: {const_metrics.CD:.4f}")
            print(f"  CONSTITUTIONAL_INDEFINITENESS: {const_metrics.CI:.4f}")
            print(f"  DUALITY_CHECK: CI × CD = {const_metrics.duality_product:.4f} ≈ Depth² = {const_metrics.duality_theoretical} (error: {const_metrics.duality_error:.2%})")
            print(f"  SEMANTIC_ACCESSIBILITY v2.0: {const_metrics.SA:.4f} ({const_metrics.SA*100:.1f}% - {const_metrics.sa_category.value})")
            if const_metrics.SA_v3 is not None:
                delta = const_metrics.SA_v3 - const_metrics.SA
                print(f"  SEMANTIC_ACCESSIBILITY v3.0: {const_metrics.SA_v3:.4f} ({const_metrics.SA_v3*100:.1f}%) [Δ={delta:+.4f}]")
            print(f"  CI_DECOMPOSITION: Morphological={const_metrics.CI_morphological:.2f}, Syntactic={const_metrics.CI_syntactic:.2f}, Semantic={const_metrics.CI_semantic:.2f}")
            sys.stdout.flush()

            # Safe print of classification (can hang for very complex sentences)
            try:
                classification_str = str(const_metrics.structure_classification.value)
                cd_ci_ratio = float(const_metrics.cd_ci_ratio)
                print(f"  CLASSIFICATION: {classification_str} (CD/CI = {cd_ci_ratio:.4f})")
                sys.stdout.flush()
            except Exception as e:
                print(f"  CLASSIFICATION: <error computing: {e}>")
                sys.stdout.flush()

            # Dodaj tensor do JSON
            result["quantum_tensor"] = {
                "value": round(T_quantum, 6),
                "formula": f"{D:.3f} × {S:.3f} × {1-E:.3f} = {T_quantum:.3f}"
            }

            # Dodaj Constitutional Metrics do JSON using dedicated calculator
            try:
                result["constitutional_metrics"] = const_metrics.to_dict()
            except Exception as e:
                print(f"  ⚠️ Error serializing constitutional metrics: {e}")
                result["constitutional_metrics"] = {"error": str(e)}

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

            # OVERRIDE SA dla IRRATIONAL_ANOMALY (dodane 22.11.2025)
            try:
                register_violation = result.get("rhetorical_analysis", {}).get("register_violation", {})
                if register_violation.get('classification') == 'IRRATIONAL_ANOMALY':
                    severity = register_violation.get('severity', 'UNKNOWN')
                    violation_score = register_violation.get('violation_score', 0)

                    # Drastyczna penalty dla SA based on severity
                    if severity == 'CRITICAL':
                        # CRITICAL: SA max 15% (wulgaryzmy w kontekście formalnym)
                        sa_override = min(const_metrics.SA_v3, 0.15)
                        penalty_reason = "VULGAR_IN_FORMAL_CONTEXT"
                    elif severity == 'HIGH':
                        # HIGH: SA max 20% (wysoka ironia w tekście prawnym)
                        sa_override = min(const_metrics.SA_v3, 0.20)
                        penalty_reason = "HIGH_IRONY_IN_LEGAL"
                    else:
                        # MODERATE/LOW: SA max 25%
                        sa_override = min(const_metrics.SA_v3, 0.25)
                        penalty_reason = "INFORMAL_LANGUAGE"

                    original_sa = const_metrics.SA_v3

                    # Nadpisz SA w result["constitutional_metrics"]
                    if "constitutional_metrics" in result and "semantic_accessibility" in result["constitutional_metrics"]:
                        result["constitutional_metrics"]["semantic_accessibility"]["v3"]["value"] = sa_override
                        result["constitutional_metrics"]["semantic_accessibility"]["v3"]["percentage"] = sa_override * 100
                        result["constitutional_metrics"]["semantic_accessibility"]["v3"]["anomaly_override"] = True
                        result["constitutional_metrics"]["semantic_accessibility"]["v3"]["original_value"] = original_sa
                        result["constitutional_metrics"]["semantic_accessibility"]["v3"]["penalty_reason"] = penalty_reason

                    # Oznacz jako krytyczny blok
                    result["critical_block"] = True
                    result["critical_reason"] = "IRRATIONAL_ANOMALY"

                    print(f"  ⚠️ SA OVERRIDDEN: {original_sa:.4f} → {sa_override:.4f} (reason: {penalty_reason})")
            except Exception as override_err:
                print(f"  ⚠️ SA override failed: {override_err}")

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

        # ========================================================================
        # TOPOLOGICAL ATTRACTORS ANALYSIS
        # ========================================================================
        print("  🔄 Starting topological analysis...")
        sys.stdout.flush()

        if self.topological_analyzer:
            try:
                coords_array = np.array([D, S, E])

                # Find nearest attractor
                print("  🔄 Finding nearest attractor...")
                sys.stdout.flush()
                nearest_name, distance, attractor_meta = \
                    self.topological_analyzer.find_nearest_attractor(coords_array)

                # Basin analysis
                basin_analysis = self.topological_analyzer.analyze_basin_of_attraction(coords_array)

                # Add to temporal evolution if enabled
                if self.temporal_evolution:
                    import time
                    self.temporal_evolution.add_state(coords_array, timestamp=time.time())

                    # Get evolution summary if we have history
                    evolution_summary = self.temporal_evolution.get_evolution_summary()

                result["topological_attractors"] = {
                    "nearest_attractor": attractor_meta,
                    "basin_analysis": basin_analysis
                }

                if self.temporal_evolution and evolution_summary and 'error' not in evolution_summary:
                    result["temporal_evolution"] = evolution_summary

                print(f"  🌐 TOPOLOGICAL: Nearest attractor = {attractor_meta['attractor_name']}")
                print(f"     Distance: {distance:.3f}, In basin: {attractor_meta['in_basin']}")

            except Exception as e:
                print(f"  ⚠️ Topological analysis failed: {e}")

        # ========================================================================
        # ENHANCED QUANTUM METRICS
        # ========================================================================
        print("  🔄 Starting enhanced quantum analysis...")
        sys.stdout.flush()

        if self.quantum_enhanced:
            try:
                # Extract words - use final_coords for all words since per-word coords aren't stored
                words = list(word_quantum_states.keys())
                print(f"  🔄 Processing {len(words)} words for quantum analysis...")
                sys.stdout.flush()

                # Use the final D-S-E coordinates for all words
                # (This is a simplification - in future could modify morphology analysis to store per-word coords)
                coords_per_word = [np.array([D, S, E]) for _ in words]

                if len(words) > 0:
                    from gtmo_quantum_enhanced import analyze_quantum_enhanced as qe_analyze

                    print(f"  🔄 Calling analyze_quantum_enhanced with {len(words)} words...")
                    sys.stdout.flush()

                    quantum_enhanced_result = qe_analyze(
                        text=text,
                        words=words,
                        coords_per_word=coords_per_word,
                        base_coherence=total_quantum_coherence
                    )

                    print(f"  ✅ Quantum analysis completed")
                    sys.stdout.flush()

                    print(f"  🔄 Assigning results to dict...")
                    sys.stdout.flush()

                    result["quantum_enhanced"] = quantum_enhanced_result

                    print(f"  ✅ Results assigned")
                    sys.stdout.flush()

                    print(f"  🔄 Printing quantum enhanced results...")
                    sys.stdout.flush()

                    print(f"  ⚛️ QUANTUM ENHANCED: {quantum_enhanced_result['num_quantum_states']} states")
                    print(f"     Phase coherence: {quantum_enhanced_result['coherence_detailed']['phase_coherence']:.3f}")
                    print(f"     Entanglement: {quantum_enhanced_result['entanglement']['mean_entanglement']:.3f}")
                    print(f"     Classification: {quantum_enhanced_result['quantum_classification']}")
                    sys.stdout.flush()

            except Exception as e:
                print(f"  ⚠️ Enhanced quantum analysis failed: {e}")

        return result


# ==================================================
# UTILITY FUNCTIONS
# ==================================================
# Note: load_markdown_file() is now imported from gtmo_file_loader module

def F3_to_Phi9(D, S, E, *, max_iter=1000, R_escape=2.0, c_base=(-0.8, 0.156)):
    """
    KOMPLETNA transformacja F³ → Φ⁹

    Input:  (D, S, E) ∈ [0,1]³  (Phase Space)
    Output: (D, S, E, θ_D, φ_D, ρ_D, θ_S, φ_S, ρ_S) ∈ Φ⁹
    """
    import numpy as np

    D = float(D); S = float(S); E = float(E)
    D, S, E = (np.clip(D, 0.0, 1.0),
               np.clip(S, 0.0, 1.0),
               np.clip(E, 0.0, 1.0))

    z0_DS = complex(D, S)
    z0_DE = complex(D, E)

    c = complex(*c_base) * (1.0 + E)

    def julia_iterate(z0, c, max_iter=1000, R_escape=2.0):
        z = z0
        R2 = R_escape * R_escape
        for n in range(1, max_iter + 1):
            z = z*z + c
            if (z.real*z.real + z.imag*z.imag) > R2:
                return True, n, z
        return False, max_iter, z

    escaped_DS, n_DS, z_final_DS = julia_iterate(z0_DS, c, max_iter, R_escape)
    escaped_DE, n_DE, z_final_DE = julia_iterate(z0_DE, c, max_iter, R_escape)

    def extract_K6_from_escape(escaped, n_escape, z_final, z0):
        import numpy as np
        if not escaped:
            return 0.0, 0.0, 0.0

        r_final = abs(z_final)
        theta = float(np.mod(np.angle(z_final), 2.0 * np.pi))
        denom = r_final if r_final > 0.0 else 1.0
        val = np.clip(z_final.imag / denom, -1.0, 1.0)
        phi = float(np.arccos(val))
        r0 = abs(z0)
        ratio = (r_final / r0) if r0 > 0.0 else r_final
        ratio = max(ratio, 1e-12)
        rho = float(np.log(ratio) / max(n_escape, 1))
        return theta, phi, rho

    theta_D, phi_D, rho_D = extract_K6_from_escape(escaped_DS, n_DS, z_final_DS, z0_DS)
    theta_S, phi_S, rho_S = extract_K6_from_escape(escaped_DE, n_DE, z_final_DE, z0_DE)

    phi9 = np.array([
        D, S, E,
        theta_D, phi_D, rho_D,
        theta_S, phi_S, rho_S
    ], dtype=float)

    return phi9

# ==================================================
# MAIN ANALYSIS FUNCTIONS
# ==================================================

# Global engine instance (initialized lazily)
_GLOBAL_ENGINE = None

def _get_global_engine():
    """Get or create global QuantumMorphosyntaxEngine instance."""
    global _GLOBAL_ENGINE
    if _GLOBAL_ENGINE is None:
        _GLOBAL_ENGINE = QuantumMorphosyntaxEngine(
            herbert_tokenizer=GLOBAL_HERBERT_TOKENIZER,
            herbert_model=GLOBAL_HERBERT_MODEL
        )
    return _GLOBAL_ENGINE

def analyze_quantum_with_axioms(text: str, source_file: str = "unknown") -> Dict:
    """
    Główna funkcja analizy z aksjomatami GTMØ
    Uses shared global engine instance to avoid reloading models.
    """
    engine = _get_global_engine()
    return engine.gtmo_analyze_quantum(text, source_file)

def batch_analyze_quantum_with_axioms(texts: List[str], source_file: str = "batch") -> List[Dict]:
    """
    Analiza wsadowa z aksjomatami GTMØ
    Uses shared global engine instance to avoid reloading models.
    """
    engine = _get_global_engine()
    results = []

    for i, text in enumerate(texts):
        source_info = f"{source_file}_sentence_{i+1}"
        result = engine.gtmo_analyze_quantum(text, source_info)
        results.append(result)

    return results

# ==================================================
# STANZA INTEGRATION FOR LEGAL TEXT ANALYSIS
# ==================================================

try:
    import stanza
    stanza_nlp = stanza.Pipeline('pl',
                                 processors='tokenize,mwt,pos,lemma,depparse',
                                 verbose=False,
                                 use_gpu=False)
    STANZA_AVAILABLE = True
    logger.info("✔ Stanza Polish pipeline loaded")
except ImportError:
    stanza_nlp = None
    STANZA_AVAILABLE = False
    logger.warning("✗ Stanza not available")

class EnhancedGTMOProcessor:
    """Main processor with Stanza integration for legal text analysis."""

    def __init__(self):
        self.stanza = stanza_nlp
        # Use global engine instance
        self.engine = _get_global_engine() if morfeusz else None

    def analyze_legal_text(self, text: str) -> Dict[str, Any]:
        """
        Comprehensive legal text analysis combining GTMØ + Stanza.

        Returns dict with:
        - gtmo_coordinates: [D, S, E]
        - stanza_analysis: dependency structure + smoking guns
        - legal_assessment: quality scoring
        """
        analysis = {
            'metadata': {
                'timestamp': datetime.now().isoformat(),
                'text_hash': hashlib.sha256(text.encode()).hexdigest()[:16],
                'engines_used': {
                    'stanza': STANZA_AVAILABLE,
                    'morfeusz2': morfeusz is not None,
                    'gtmo_quantum': True
                }
            },
            'text': text
        }

        # GTMØ analysis
        if self.engine:
            gtmo_result = self.engine.gtmo_analyze_quantum(text)
            coords = gtmo_result.get('coordinates', {})
            analysis['gtmo_coordinates'] = {
                'determination': float(coords.get('determination', 0.5)),
                'stability': float(coords.get('stability', 0.5)),
                'entropy': float(coords.get('entropy', 0.5))
            }
        else:
            analysis['gtmo_coordinates'] = {
                'determination': 0.5,
                'stability': 0.5,
                'entropy': 0.5
            }

        # Stanza analysis
        if STANZA_AVAILABLE and self.stanza:
            stanza_result = self._analyze_with_stanza(text)
            analysis['stanza_analysis'] = stanza_result
        else:
            analysis['stanza_analysis'] = {'error': 'Stanza not available'}

        # Legal assessment
        analysis['legal_assessment'] = self._generate_assessment(analysis)

        return analysis

    def _analyze_with_stanza(self, text: str) -> Dict:
        """Analyze with Stanza - advanced smoking gun detection."""
        try:
            doc = self.stanza(text)

            sentences = []
            smoking_guns = []

            # Track key verbs and negations across sentences
            negated_verbs = []
            affirming_verbs = []

            for sent in doc.sentences:
                sent_info = {
                    'text': sent.text,
                    'words': len(sent.words),
                    'dependencies': []
                }

                # Extract dependencies and track semantic patterns
                has_negation = False
                negated_action = None

                for word in sent.words:
                    sent_info['dependencies'].append({
                        'text': word.text,
                        'lemma': word.lemma,
                        'upos': word.upos,
                        'deprel': word.deprel
                    })

                    # Track negations
                    if word.lemma == 'nie' and word.deprel == 'advmod:neg':
                        has_negation = True

                    # Track negated verbs (legal actions)
                    if word.upos == 'VERB' and has_negation:
                        negated_action = word.lemma
                        negated_verbs.append({
                            'lemma': word.lemma,
                            'text': word.text,
                            'sentence': sent.text
                        })

                    # Track affirming actions (consequences)
                    if word.lemma in ['skazywać', 'skazać', 'ukarać'] and word.upos == 'VERB':
                        affirming_verbs.append({
                            'lemma': word.lemma,
                            'text': word.text,
                            'sentence': sent.text
                        })

                sentences.append(sent_info)

            # Cross-sentence contradiction detection
            # "nie popełnił" → "skazuje"
            if negated_verbs and affirming_verbs:
                for neg in negated_verbs:
                    for aff in affirming_verbs:
                        if neg['sentence'] != aff['sentence']:
                            smoking_guns.append({
                                'type': 'negation_consequence_conflict',
                                'severity': 0.95,
                                'details': {
                                    'negation': f"'{neg['text']}' (nie {neg['lemma']})",
                                    'consequence': f"'{aff['text']}' ({aff['lemma']})",
                                    'conflict': f"'{neg['text']}' → '{aff['text']}'"
                                },
                                'sentences': [neg['sentence'], aff['sentence']]
                            })

            return {
                'sentences': sentences,
                'smoking_guns': smoking_guns,
                'sentence_count': len(sentences)
            }

        except Exception as e:
            return {'error': str(e)}

    def _generate_assessment(self, analysis: Dict) -> Dict:
        """Generate legal quality assessment."""
        coords = analysis['gtmo_coordinates']
        stanza = analysis.get('stanza_analysis', {})

        # Simple quality scoring
        coherence = (coords['stability'] + (1 - coords['entropy'])) / 2

        smoking_gun_count = len(stanza.get('smoking_guns', []))

        if smoking_gun_count > 2 or coherence < 0.3:
            quality = 'critical'
        elif smoking_gun_count > 0 or coherence < 0.5:
            quality = 'poor'
        elif coherence < 0.7:
            quality = 'fair'
        else:
            quality = 'good'

        return {
            'quality': quality,
            'legal_coherence_score': float(coherence),
            'smoking_gun_count': smoking_gun_count
        }

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
            # Load markdown file - now returns articles, not sentences
            articles = load_markdown_file(file_path)

            if not articles:
                print("❌ No articles found in file")
                sys.exit(1)

            print(f"📄 Found {len(articles)} articles/units")
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

            # Store all article analyses for full document
            article_analyses = []

            # Global sentence counter (to avoid overwriting sentence files)
            global_sentence_counter = 0

            try:
                # Analyze each article and save individually
                for i, article in enumerate(articles, 1):
                    print(f"\n🌌 Analyzing article {i}/{len(articles)}")
                    print(f"Text: {article[:100]}{'...' if len(article) > 100 else ''}")

                    try:
                        # Split article into paragraphs (§1, §2, §3, etc.)
                        import re
                        paragraph_pattern = r'§\s*\d+[^\n]*(?:\n(?!§\s*\d+)[^\n]*)*'
                        paragraph_matches = re.findall(paragraph_pattern, article, flags=re.MULTILINE | re.DOTALL)

                        # If no paragraphs found, treat entire article as sentences
                        if not paragraph_matches:
                            print(f"  ℹ️  No paragraph markers found - analyzing as individual sentences")

                            # Split into sentences
                            if nlp:
                                doc = nlp(article)
                                sentences = [sent.text.strip() for sent in doc.sents if sent.text.strip() and len(sent.text.strip()) >= 10]
                            else:
                                sentences = re.split(r'[.!?]+', article)
                                sentences = [s.strip() for s in sentences if s.strip() and len(s.strip()) >= 10]

                            print(f"  📝 Found {len(sentences)} sentences to analyze")

                            # Analyze and save each sentence individually
                            sentence_analyses = []
                            for s_idx, sentence in enumerate(sentences, 1):
                                try:
                                    global_sentence_counter += 1
                                    print(f"\n🔄 Processing sentence {global_sentence_counter} (art {i}, local {s_idx}/{len(sentences)})")
                                    print(f"   Text preview: {sentence[:100]}...")
                                    sent_result = analyze_quantum_with_axioms(sentence, os.path.basename(file_path))
                                    sent_result['sentence_number'] = s_idx
                                    sent_result['global_sentence_number'] = global_sentence_counter
                                    sent_result['article_number'] = i
                                    sentence_analyses.append(sent_result)

                                    # Save individual sentence using GLOBAL counter
                                    saved_file = saver.save_sentence_analysis(sent_result, sentence, global_sentence_counter)
                                    print(f"  ✅ Saved sentence {global_sentence_counter} (art {i}, local {s_idx}/{len(sentences)}): {saved_file}")

                                except Exception as e:
                                    print(f"  ⚠️  Error analyzing sentence {s_idx}: {e}")
                                    continue

                            # Create article analysis with sentences for full document
                            article_result = {
                                'article_number': i,
                                'total_articles': len(articles),
                                'sentence_count': len(sentence_analyses),
                                'sentences': sentence_analyses,
                                'article_text': article
                            }
                            article_analyses.append(article_result)

                            # Skip the rest of the article processing (paragraphs)
                            continue

                        # Analyze each paragraph
                        paragraph_analyses = []
                        total_sentences = 0

                        for p_idx, paragraph in enumerate(paragraph_matches, 1):
                            # Split paragraph into sentences using spaCy
                            if nlp:
                                doc = nlp(paragraph)
                                sentences = [sent.text.strip() for sent in doc.sents if sent.text.strip() and len(sent.text.strip()) >= 10]
                            else:
                                # Fallback: simple sentence splitting
                                sentences = re.split(r'[.!?]+', paragraph)
                                sentences = [s.strip() for s in sentences if s.strip() and len(s.strip()) >= 10]

                            # Analyze each sentence in paragraph
                            sentence_analyses = []
                            for s_idx, sentence in enumerate(sentences, 1):
                                try:
                                    global_sentence_counter += 1
                                    print(f"\n🔄 Processing sentence {global_sentence_counter} (§{p_idx}, local {s_idx}/{len(sentences)})")
                                    print(f"   Text preview: {sentence[:100]}...")
                                    sent_result = analyze_quantum_with_axioms(sentence, os.path.basename(file_path))
                                    sent_result['sentence_number'] = s_idx
                                    sent_result['global_sentence_number'] = global_sentence_counter
                                    sent_result['paragraph_number'] = p_idx
                                    sent_result['article_number'] = i
                                    sentence_analyses.append(sent_result)

                                    # Save individual sentence to separate JSON file
                                    saved_file = saver.save_sentence_analysis(sent_result, sentence, global_sentence_counter)
                                    print(f"  ✅ Saved sentence {global_sentence_counter} (art {i}, §{p_idx}, local {s_idx}/{len(sentences)}): {saved_file}")

                                except Exception as e:
                                    print(f"  ⚠️  Error analyzing sentence {s_idx} in §{p_idx}: {e}")
                                    continue

                            # Analyze entire paragraph (skip if too large to avoid hangs)
                            try:
                                # Skip paragraph-level analysis if too many sentences (>20) or too long (>5000 chars)
                                if len(sentences) > 20 or len(paragraph) > 5000:
                                    print(f"  ⏭️  Skipping paragraph-level analysis for §{p_idx} ({len(sentences)} sentences, {len(paragraph)} chars - too large)")
                                    # Create minimal paragraph result without full analysis
                                    para_result = {
                                        'paragraph_number': p_idx,
                                        'sentence_count': len(sentences),
                                        'sentences': sentence_analyses,
                                        'skipped': True,
                                        'skip_reason': f'Too large: {len(sentences)} sentences, {len(paragraph)} characters'
                                    }
                                else:
                                    para_result = analyze_quantum_with_axioms(paragraph, os.path.basename(file_path))
                                    para_result['paragraph_number'] = p_idx
                                    para_result['sentence_count'] = len(sentences)
                                    para_result['sentences'] = sentence_analyses
                                    print(f"  ✓ §{p_idx}: {len(sentences)} sentences analyzed")

                                paragraph_analyses.append(para_result)
                                total_sentences += len(sentences)
                            except Exception as e:
                                print(f"  ⚠️  Error analyzing paragraph {p_idx}: {e}")
                                continue

                        # Analyze complete article (all paragraphs together) - skip if too large
                        result = analyze_quantum_with_axioms(article, os.path.basename(file_path))
                        result["article_number"] = i
                        result["total_articles"] = len(articles)
                        result["paragraph_count"] = len(paragraph_matches)
                        result["sentence_count"] = total_sentences
                        result["paragraphs"] = paragraph_analyses

                        # Save individual article result
                        saved_file = saver.save_article_analysis(result, article, i)
                        print(f"✅ Saved article to: {saved_file}")
                        print(f"   Hierarchy: {len(paragraph_matches)} paragraphs, {total_sentences} sentences")

                        # Store for full document analysis
                        article_analyses.append(result)

                    except Exception as e:
                        print(f"❌ Error analyzing article {i}: {e}")
                        continue

                # Save full document analysis (after all articles processed)
                if article_analyses:
                    try:
                        full_doc_file = saver.save_full_document_analysis(
                            source_file=file_path,
                            articles=articles,
                            article_analyses=article_analyses
                        )
                        print(f"\n📄 Saved full document analysis to: {full_doc_file}")
                    except Exception as e:
                        print(f"❌ Error saving full document: {e}")

            except KeyboardInterrupt:
                print(f"\n⚠️  Analysis interrupted by user. Saving embeddings...")
            except Exception as e:
                print(f"\n❌ Error during analysis: {e}")
            finally:
                # ALWAYS finalize and save HerBERT embeddings, even if interrupted
                try:
                    embeddings_file = saver.finalize_embeddings()
                    if embeddings_file:
                        print(f"🤖 Saved {len(saver.embedding_storage.embeddings_cache)} HerBERT embeddings to: {embeddings_file}")
                except Exception as e:
                    print(f"⚠️  Error saving embeddings: {e}")

                # ALWAYS finalize and save numeric matrices, even if interrupted
                try:
                    matrices_file = saver.finalize_matrices()
                    if matrices_file:
                        print(f"🔢 Saved {len(saver.matrix_storage.matrices_cache)} numeric matrices to: {matrices_file}")
                except Exception as e:
                    print(f"⚠️  Error saving matrices: {e}")

                # Create HerBERT semantic flow analysis
                try:
                    herbert_analysis_file = saver.create_herbert_analysis()
                    if herbert_analysis_file:
                        print(f"📊 Saved HerBERT semantic analysis to: {herbert_analysis_file}")
                except Exception as e:
                    print(f"⚠️  Error creating HerBERT analysis: {e}")

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

            # Test Stanza integration
            print("\n" + "=" * 70)
            print("🧪 TESTING STANZA INTEGRATION (EnhancedGTMOProcessor)")
            print("=" * 70)
            print(f"Stanza available: {STANZA_AVAILABLE}")

            if STANZA_AVAILABLE:
                print("✔ Testing smoking gun detection...")
                processor = EnhancedGTMOProcessor()
                test_legal = "Sąd uznał, że oskarżony nie popełnił czynu. Jednak go skazuje."
                result = processor.analyze_legal_text(test_legal)

                print(f"\n📄 Text: {test_legal}")
                print(f"🎯 GTMØ Coordinates: D={result['gtmo_coordinates']['determination']:.3f}, "
                      f"S={result['gtmo_coordinates']['stability']:.3f}, "
                      f"E={result['gtmo_coordinates']['entropy']:.3f}")

                if result['stanza_analysis']['smoking_guns']:
                    print(f"\n🔫 SMOKING GUNS DETECTED: {len(result['stanza_analysis']['smoking_guns'])}")
                    for gun in result['stanza_analysis']['smoking_guns']:
                        print(f"   Type: {gun['type']}")
                        print(f"   Severity: {gun['severity']}")
                        if 'details' in gun:
                            print(f"   Conflict: {gun['details'].get('conflict', 'N/A')}")
                else:
                    print("\n✓ No contradictions detected")

                print(f"\n⚖️ Legal Assessment:")
                print(f"   Quality: {result['legal_assessment']['quality']}")
                print(f"   Coherence: {result['legal_assessment']['legal_coherence_score']:.3f}")
            else:
                print("✗ Stanza not available - install with:")
                print("  pip install stanza")
                print("  python -c \"import stanza; stanza.download('pl')\"")

            print("=" * 70)
