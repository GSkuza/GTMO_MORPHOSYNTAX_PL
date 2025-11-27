#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
GTMØ Pure Rhetorical Analysis Module
=====================================
Rhetorical mode detection based on morphosyntactic anomalies,
not pattern matching. True to GTMØ philosophy.
"""

import numpy as np
from typing import Dict, Tuple, Optional, List
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class GTMORhetoricalAnalyzer:
    """
    Detects rhetorical modes through structural anomalies,
    not through keyword patterns.
    """
    
    def __init__(self):
        # Normalne wartości dla języka polskiego (baseline)
        self.baseline = {
            'adj_ratio': 0.15,        # Normalnie ~15% przymiotników
            'case_entropy': 0.6,       # Normalny rozkład przypadków
            'divergence_threshold': 0.12,  # Normalna rozbieżność morph/syntax
            'ambiguity_normal': 1.2,   # Normalna wieloznaczność morfologiczna
            'depth_normal': 3.5,       # Normalna głębokość składniowa
        }
    
    def calculate_morphological_entropy(self, cases: Dict[str, int]) -> float:
        """
        Oblicz entropię rozkładu przypadków.
        Wysoka entropia = równomierny rozkład (nienaturalny)
        Niska entropia = dominacja jednego przypadka (naturalny)
        """
        if not cases:
            return 0.5
        
        total = sum(cases.values())
        if total == 0:
            return 0.5
        
        # Oblicz rozkład prawdopodobieństw
        probs = np.array([count/total for count in cases.values()])
        
        # Entropia Shannona
        entropy = -np.sum(probs * np.log(probs + 1e-10))
        
        # Normalizuj do [0, 1]
        max_entropy = -np.log(1/7)  # 7 przypadków w polskim
        return min(entropy / max_entropy, 1.0)
    
    def calculate_pos_anomaly(self, pos_counts: Dict[str, int]) -> Dict[str, float]:
        """
        Wykryj anomalie w rozkładzie części mowy.
        """
        if not pos_counts:
            return {'anomaly_score': 0.0}
        
        total = sum(pos_counts.values())
        if total == 0:
            return {'anomaly_score': 0.0}
        
        # Oblicz współczynniki
        adj_ratio = pos_counts.get('adj', 0) / total
        subst_ratio = pos_counts.get('subst', 0) / total
        verb_ratio = pos_counts.get('verb', 0) / total
        interp_ratio = pos_counts.get('interp', 0) / total
        
        anomalies = {
            'adj_excess': max(0, (adj_ratio - self.baseline['adj_ratio']) * 3),
            'verb_deficit': max(0, (0.2 - verb_ratio) * 2),
            'punct_excess': max(0, (interp_ratio - 0.15) * 2),
            'adj_ratio': adj_ratio,
            'subst_ratio': subst_ratio,
            'verb_ratio': verb_ratio
        }
        
        # Zagregowana anomalia
        anomalies['anomaly_score'] = (
            anomalies['adj_excess'] + 
            anomalies['verb_deficit'] + 
            anomalies['punct_excess']
        )
        
        return anomalies
    
    def calculate_structural_divergence(self, 
                                      morph_coords: np.ndarray,
                                      syntax_coords: np.ndarray) -> float:
        """
        Oblicz rozbieżność między analizą morfologiczną a składniową.
        Wysoka rozbieżność sugeruje niespójność intencji.
        """
        # Euklidesowa odległość
        divergence = np.linalg.norm(morph_coords - syntax_coords)
        
        # Szczególnie ważna jest rozbieżność w determinacji i stabilności
        d_diff = abs(morph_coords[0] - syntax_coords[0])
        s_diff = abs(morph_coords[1] - syntax_coords[1])
        
        # Ważona rozbieżność
        weighted_divergence = divergence + 0.5 * (d_diff + s_diff)
        
        return weighted_divergence
    
    def detect_ironic_structure(self, 
                                morph_coords: np.ndarray,
                                syntax_coords: np.ndarray,
                                morph_metadata: Dict,
                                syntax_metadata: Dict) -> Tuple[bool, float, Dict]:
        """
        Wykryj strukturę ironiczną przez anomalie, nie wzorce.
        
        Ironia charakteryzuje się:
        1. Wysoką determinacją morfologiczną przy niskiej składniowej
        2. Nadmiarem przymiotników wartościujących
        3. Nienaturalnym rozkładem przypadków
        4. Rozbieżnością między formą a treścią
        """
        
        analysis = {
            'structural_divergence': 0.0,
            'morphological_entropy': 0.0,
            'pos_anomalies': {},
            'ambiguity_factor': 0.0,
            'irony_indicators': []
        }
        
        # 1. Rozbieżność strukturalna
        divergence = self.calculate_structural_divergence(morph_coords, syntax_coords)
        analysis['structural_divergence'] = divergence
        
        # 2. Anomalie części mowy
        pos_anomalies = self.calculate_pos_anomaly(morph_metadata.get('pos', {}))
        analysis['pos_anomalies'] = pos_anomalies
        
        # 3. Entropia morfologiczna
        case_entropy = self.calculate_morphological_entropy(morph_metadata.get('cases', {}))
        analysis['morphological_entropy'] = case_entropy
        
        # 4. Współczynnik wieloznaczności
        ambiguity = morph_metadata.get('ambiguity', 1.0)
        analysis['ambiguity_factor'] = ambiguity
        
        # Wskaźniki ironii (bez pattern matching!)
        irony_score = 0.0
        
        # Wysoka determinacja morfologiczna z rozbieżnością
        # Było: if morph_coords[0] > 0.65 and divergence > 0.15:
        # Potem: if morph_coords[0] > 0.65 and divergence > 0.30:
        # Teraz:
        if morph_coords[0] > 0.75 and divergence > 0.40:  # Drastycznie wyższe progi
            irony_score += 0.3
            analysis['irony_indicators'].append('high_determination_with_divergence')
        
        # Nadmiar przymiotników (struktura przesadzona)
        # Było: if pos_anomalies['adj_ratio'] > 0.35:
        # Potem: if pos_anomalies['adj_ratio'] > 0.45:
        # Teraz:
        if pos_anomalies.get('adj_ratio', 0.0) > 0.60:  # Drastycznie wyższy próg
            irony_score += 0.4
            analysis['irony_indicators'].append('adjective_overload')
        elif pos_anomalies.get('adj_ratio', 0.0) > 0.25:
            irony_score += 0.2
            analysis['irony_indicators'].append('adjective_excess')
        
        # Nienaturalna entropia przypadków
        if case_entropy > 0.75 or case_entropy < 0.25:
            irony_score += 0.2
            analysis['irony_indicators'].append('unnatural_case_distribution')
        
        # Deficyt czasowników przy nadmiarze przymiotników
        if pos_anomalies.get('verb_ratio', 0.2) < 0.1 and pos_anomalies['adj_ratio'] > 0.2:
            irony_score += 0.3
            analysis['irony_indicators'].append('verb_deficit_with_adj_excess')
        
        # Wysoka stabilność morfologiczna przy niskiej składniowej
        morph_stability = morph_coords[1]
        syntax_stability = syntax_coords[1]
        if morph_stability > 0.65 and syntax_stability < 0.55:
            irony_score += 0.2
            analysis['irony_indicators'].append('stability_mismatch')
        
        # Anomalna kombinacja: wysoka determinacja + niska entropia + rozbieżność
        if (morph_coords[0] > 0.6 and 
            morph_coords[2] < 0.35 and 
            divergence > 0.12):
            irony_score += 0.25
            analysis['irony_indicators'].append('forced_certainty_pattern')
        
        analysis['irony_score'] = irony_score

        # Decyzja o ironii (próg adaptacyjny)
        # Było: is_ironic = irony_score > 0.5
        # Potem: is_ironic = irony_score > 1.0
        # Teraz:
        is_ironic = irony_score > 1.75  # Potrzeba jeszcze więcej dowodów

        return is_ironic, irony_score, analysis
    
    def detect_paradox_structure(self,
                                 morph_coords: np.ndarray,
                                 syntax_coords: np.ndarray,
                                 morph_metadata: Dict) -> Tuple[bool, float, Dict]:
        """
        Wykryj strukturę paradoksu.
        
        Paradoks charakteryzuje się:
        1. Wysoką entropią przy wysokiej stabilności
        2. Równowagą przeciwstawnych elementów
        3. Symetrią strukturalną
        """
        
        analysis = {
            'paradox_indicators': [],
            'symmetry_score': 0.0
        }
        
        paradox_score = 0.0
        
        # Wysoka entropia + wysoka stabilność = paradoks
        if morph_coords[2] > 0.6 and morph_coords[1] > 0.6:
            paradox_score += 0.5
            analysis['paradox_indicators'].append('high_entropy_high_stability')
        
        # Równowaga przypadków (symetria)
        cases = morph_metadata.get('cases', {})
        if cases:
            case_values = list(cases.values())
            if len(case_values) > 1:
                case_variance = np.var(case_values)
                # Niska wariancja = równowaga
                if case_variance < 2.0:
                    paradox_score += 0.3
                    analysis['paradox_indicators'].append('case_balance')
                    analysis['symmetry_score'] = 1.0 - (case_variance / 10.0)
        
        # Stabilność mimo sprzeczności (D średnie, S wysokie, E wysokie)
        if (0.4 < morph_coords[0] < 0.6 and 
            morph_coords[1] > 0.65 and 
            morph_coords[2] > 0.55):
            paradox_score += 0.3
            analysis['paradox_indicators'].append('stable_contradiction')
        
        analysis['paradox_score'] = paradox_score
        is_paradox = paradox_score > 0.5
        
        return is_paradox, paradox_score, analysis
    
    def analyze_rhetorical_mode(self,
                                text: str,
                                morph_coords: np.ndarray,
                                syntax_coords: np.ndarray,
                                morph_metadata: Dict,
                                syntax_metadata: Dict) -> Tuple[np.ndarray, str, Dict]:
        """
        Główna funkcja analizy retorycznej.
        Zwraca przekształcone współrzędne i tryb.
        """
        
        # Sprawdź ironię
        is_ironic, irony_score, irony_analysis = self.detect_ironic_structure(
            morph_coords, syntax_coords, morph_metadata, syntax_metadata
        )
        
        # Sprawdź paradoks
        is_paradox, paradox_score, paradox_analysis = self.detect_paradox_structure(
            morph_coords, syntax_coords, morph_metadata
        )
        
        # Metadane
        metadata = {
            'irony_score': irony_score,
            'paradox_score': paradox_score,
            'irony_analysis': irony_analysis,
            'paradox_analysis': paradox_analysis,
            'structural_divergence': irony_analysis['structural_divergence'],
            'pos_anomalies': irony_analysis['pos_anomalies']
        }
        
        # Decyzja o trybie i transformacja współrzędnych
        if is_ironic and irony_score > paradox_score:
            # INWERSJA dla ironii
            transformed_coords = np.array([
                1.0 - morph_coords[0],  # Odwróć determinację
                1.0 - morph_coords[1],  # Odwróć stabilność
                1.0 - morph_coords[2]   # Odwróć entropię
            ])
            mode = 'irony'
            logger.info(f"Detected IRONY (score: {irony_score:.2f})")
            
        elif is_paradox:
            # Boost entropii dla paradoksu
            transformed_coords = morph_coords.copy()
            transformed_coords[2] = min(transformed_coords[2] * 1.5, 1.0)
            mode = 'paradox'
            logger.info(f"Detected PARADOX (score: {paradox_score:.2f})")
            
        else:
            # Brak transformacji
            transformed_coords = morph_coords
            mode = 'literal'
        
        return transformed_coords, mode, metadata

    def detect_legal_formal_context(self, text: str) -> str:
        """
        Auto-detekcja kontekstu prawnego/formalnego.

        Rozróżnia:
        - 'legal_formal': Formalny tekst prawny (konstytucje, ustawy, umowy)
          → Wyłączona detekcja ironii strukturalnej (false positives)
        - 'legal': Ogólny kontekst prawny, ale potencjalnie z ironią
        - 'formal': Tekst formalny nie-prawny
        - 'informal': Tekst nieformalny

        Returns:
            Typ kontekstu jako string
        """
        text_lower = text.lower()

        # Markery formalnego tekstu prawnego (konstytucje, ustawy)
        LEGAL_FORMAL_MARKERS = {
            # Struktura ustaw/konstytucji
            'art.', 'ust.', '§', 'rozdział', 'dział', 'tytuł',
            # Terminologia konstytucyjna
            'konstytucj', 'rzeczpospolit', 'sejm', 'senat', 'prezydent',
            'trybunał', 'sąd najwyższy', 'rada ministrów',
            # Terminologia ustawowa
            'ustaw', 'rozporządzen', 'dekret', 'uchwał',
            # Frazy prawne
            'wchodzi w życie', 'traci moc', 'stosuje się', 'podlega',
            'na podstawie', 'w drodze', 'zgodnie z', 'w trybie'
        }

        # Markery ogólnego kontekstu prawnego (mniej formalnego)
        LEGAL_GENERAL_MARKERS = {
            'umow', 'regulamin', 'statut', 'kodeks',
            'pozwan', 'powód', 'oskarżon', 'wyrok'
        }

        # Zlicz markery
        formal_count = sum(1 for m in LEGAL_FORMAL_MARKERS if m in text_lower)
        general_count = sum(1 for m in LEGAL_GENERAL_MARKERS if m in text_lower)

        # Decyzja
        if formal_count >= 2:
            return 'legal_formal'  # Formalny tekst prawny - BEZ detekcji ironii
        elif formal_count >= 1 or general_count >= 2:
            return 'legal'  # Ogólny kontekst prawny
        elif general_count >= 1:
            return 'formal'  # Formalny ale nie prawny
        else:
            return 'informal'  # Nieformalny - pełna detekcja ironii

    def detect_formal_register_violation(
        self,
        text: str,
        irony_score: float,
        context_type: str = 'auto'
    ) -> Dict:
        """
        Wykrywa naruszenia rejestru formalnego w tekstach prawnych.

        Sprawdza:
        1. Wulgaryzmy i język potoczny
        2. Slang i bełkot
        3. Wysoką ironię (> 0.7) jako anomalię (TYLKO dla context_type != 'legal_formal')
        4. Niespójność stylistyczną

        Args:
            text: Tekst do analizy
            irony_score: Wynik analizy ironii (0-1)
            context_type: Typ kontekstu ('auto', 'legal_formal', 'legal', 'formal', 'informal')
                         'auto' = automatyczna detekcja
                         'legal_formal' = formalny tekst prawny (wyłączona detekcja ironii)

        Returns:
            Dict ze szczegółami naruszenia rejestru
        """
        # Auto-detekcja kontekstu jeśli 'auto'
        if context_type == 'auto':
            context_type = self.detect_legal_formal_context(text)

        # Lista wulgaryzmów i języka nieformalnego
        VULGAR_WORDS = {
            # Wulgaryzmy
            'jebany', 'kurwa', 'pierdol', 'chuj', 'dupa', 'gówno',
            'srać', 'pierdolić', 'kurewsk', 'zajebist',
            # Slang i potoczyzmy
            'koleś', 'ziom', 'fajn', 'spoko', 'git', 'ogarn',
            'mega', 'super', 'ekstra', 'zajebiście',
            # Bełkot / nonsens
            'bzdura', 'bullshit', 'gówniany', 'sratwy'
        }

        # Markery języka nieformalnego
        INFORMAL_MARKERS = {
            'no', 'jak', 'jakby', 'typu', 'znaczy', 'wiesz',
            'elo', 'hej', 'cześć', 'siema'
        }

        text_lower = text.lower()
        words = text_lower.split()

        # Wykryj wulgaryzmy
        vulgar_found = []
        for word in VULGAR_WORDS:
            if word in text_lower:
                vulgar_found.append(word)

        # Wykryj markery nieformalne
        informal_found = []
        for marker in INFORMAL_MARKERS:
            if f' {marker} ' in f' {text_lower} ':
                informal_found.append(marker)

        # Oblicz severity
        register_violation_score = 0.0
        anomaly_type = None
        severity = 'NONE'

        # Wulgaryzmy w kontekście formalnym = CRITICAL
        if vulgar_found and context_type in ['legal', 'legal_formal', 'formal']:
            register_violation_score = 1.0
            anomaly_type = 'VULGAR_IN_FORMAL_CONTEXT'
            severity = 'CRITICAL'

        # Wysoka ironia (> 0.7) w tekście prawnym = ANOMALY
        # UWAGA: Wyłączone dla 'legal_formal' (formalnych tekstów prawnych)
        # ponieważ struktura prawna generuje FALSE POSITIVES
        elif irony_score > 0.7 and context_type == 'legal':
            # Tylko dla 'legal' (nie 'legal_formal')
            register_violation_score = min(irony_score, 1.0)
            anomaly_type = 'HIGH_IRONY_IN_LEGAL'
            severity = 'HIGH'

        # Nadmiar markerów nieformalnych
        elif len(informal_found) >= 3:
            register_violation_score = min(len(informal_found) / 5.0, 1.0)
            anomaly_type = 'EXCESSIVE_INFORMAL_MARKERS'
            severity = 'MODERATE'

        # Pojedyncze markery nieformalne
        elif informal_found:
            register_violation_score = 0.3
            anomaly_type = 'INFORMAL_MARKERS_PRESENT'
            severity = 'LOW'

        return {
            'has_violation': register_violation_score > 0.0,
            'violation_score': register_violation_score,
            'anomaly_type': anomaly_type,
            'severity': severity,
            'vulgar_words_found': vulgar_found,
            'informal_markers_found': informal_found,
            'irony_triggered': irony_score > 0.7 and context_type != 'legal_formal',
            'classification': 'IRRATIONAL_ANOMALY' if severity in ['CRITICAL', 'HIGH'] else None,
            'detected_context': context_type,  # Dodane: wykryty/użyty kontekst
            'irony_suppressed': context_type == 'legal_formal' and irony_score > 0.7  # Czy ironia została wyciszona
        }


def integrate_with_gtmo(text: str, base_result: Dict) -> Dict:
    """
    Integracja z istniejącym systemem GTMØ.
    """
    
    # Wyciągnij dane z bazowej analizy
    morph_coords = np.array([
        base_result['coordinates']['determination'],
        base_result['coordinates']['stability'],
        base_result['coordinates']['entropy']
    ])
    
    # Oblicz syntax_coords z metadanych (jeśli dostępne)
    if 'syntax' in base_result and 'morphology' in base_result:
        # Mamy pełne dane
        morph_meta = base_result['morphology']
        syntax_meta = base_result['syntax']
        
        # Przybliżone współrzędne składniowe
        # (w prawdziwej implementacji byłyby dokładniejsze)
        syntax_coords = morph_coords * [0.8, 0.9, 1.2]  # Przykład
    else:
        # Fallback
        syntax_coords = morph_coords * [0.9, 0.95, 1.1]
        morph_meta = {}
        syntax_meta = {}
    
    # Analiza retoryczna
    analyzer = GTMORhetoricalAnalyzer()
    transformed_coords, mode, metadata = analyzer.analyze_rhetorical_mode(
        text, morph_coords, syntax_coords, morph_meta, syntax_meta
    )
    
    # Zaktualizuj wynik
    result = base_result.copy()
    result['coordinates'] = {
        'determination': float(transformed_coords[0]),
        'stability': float(transformed_coords[1]),
        'entropy': float(transformed_coords[2])
    }
    result['rhetorical_mode'] = mode
    result['rhetorical_metadata'] = metadata
    
    return result


# Testy
if __name__ == "__main__":
    print("="*60)
    print("GTMØ Pure Rhetorical Analysis")
    print("="*60)
    
    # Symulacja danych z gtmo_morphosyntax
    test_cases = [
        {
            'text': "Super, kolejna awaria",
            'morph_coords': np.array([0.733, 0.719, 0.298]),
            'syntax_coords': np.array([0.520, 0.530, 0.475]),
            'morph_meta': {
                'pos': {'adj': 24, 'subst': 2, 'adv': 1, 'interp': 1},
                'cases': {'acc': 6, 'dat': 3, 'gen': 4, 'loc': 3, 'nom': 7},
                'ambiguity': 1.8
            },
            'expected': 'irony'
        },
        {
            'text': "Kocham i nienawidzę jednocześnie",
            'morph_coords': np.array([0.5, 0.7, 0.75]),
            'syntax_coords': np.array([0.48, 0.68, 0.73]),
            'morph_meta': {
                'pos': {'verb': 2, 'conj': 1, 'adv': 1},
                'cases': {'acc': 1},
                'ambiguity': 1.2
            },
            'expected': 'paradox'
        },
        {
            'text': "Pada deszcz",
            'morph_coords': np.array([0.8, 0.75, 0.25]),
            'syntax_coords': np.array([0.78, 0.73, 0.27]),
            'morph_meta': {
                'pos': {'verb': 1, 'subst': 1},
                'cases': {'nom': 1},
                'ambiguity': 1.0
            },
            'expected': 'literal'
        }
    ]
    
    analyzer = GTMORhetoricalAnalyzer()
    
    for case in test_cases:
        print(f"\nAnalyzing: '{case['text']}'")
        print(f"Expected: {case['expected']}")
        
        coords, mode, meta = analyzer.analyze_rhetorical_mode(
            case['text'],
            case['morph_coords'],
            case['syntax_coords'],
            case['morph_meta'],
            {}
        )
        
        print(f"Detected: {mode}")
        print(f"  Irony score: {meta['irony_score']:.2f}")
        print(f"  Paradox score: {meta['paradox_score']:.2f}")
        
        if mode == 'irony':
            print(f"  Original coords: {case['morph_coords']}")
            print(f"  Inverted coords: {coords}")
            print(f"  Indicators: {meta['irony_analysis']['irony_indicators']}")
        
        status = "✓" if mode == case['expected'] else "✗"
        print(f"Result: {status}")
    
    print("\n" + "="*60)

    from gtmo_pure_rhetoric import integrate_with_gtmo

    # Przykładowy wynik z bazowej analizy GTMØ
    base_result = {
        'coordinates': {
            'determination': 0.733,
            'stability': 0.719,
            'entropy': 0.298
        },
        'morphology': {
            'pos': {'adj': 24, 'subst': 2, 'adv': 1, 'interp': 1},
            'cases': {'acc': 6, 'dat': 3, 'gen': 4, 'loc': 3, 'nom': 7},
            'ambiguity': 1.8
        },
        'syntax': {
            # ...opcjonalne metadane składniowe...
        }
    }

    text = "Super, kolejna awaria"

    # Integracja z analizą retoryczną
    result = integrate_with_gtmo(text, base_result)

    print(result['rhetorical_mode'])           # np. 'irony'
    print(result['rhetorical_metadata'])       # szczegóły analizy
    print(result['coordinates'])               # przekształcone współrzędne