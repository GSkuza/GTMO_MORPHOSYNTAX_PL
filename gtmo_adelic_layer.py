#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
GTMØ Adelic Layer - P-adyczna warstwa emergencji semantycznej
============================================================

Implementacja teoretycznej struktury adelicznej dla modelowania emergencji
globalnego znaczenia z lokalnych interpretacji obserwatorów.

Matematyczne podstawy:
- Obserwatorzy jako "semantyczne liczby pierwsze"
- Pierścień Adeli: 𝔸_sem(w) = ℝ_sem × ∏'_O 𝕂_O
- Emergencja przy n ≥ 2: konsensus lokalnych kół → wartość globalna

Autor: GTMØ Development Team
Data: 2024-11-24
"""

import numpy as np
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass, field
import time
from enum import Enum
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# =============================================================================
# OBSERVER - Semantyczna liczba pierwsza
# =============================================================================

@dataclass
class Observer:
    """
    Obserwator jako niepodzielna jednostka interpretacji.
    Każdy obserwator definiuje własne lokalne koło semantyczne 𝕂_O.

    Analogia matematyczna:
    - Liczba pierwsza p → Observer O
    - Koło p-adyczne ℚₚ → Koło semantyczne 𝕂_O
    - Norma p-adyczna → Metryka interpretacji obserwatora

    Attributes:
        id: Unikalny identyfikator obserwatora (np. "O_formal", "O_poetic")
        interpretation_bias: Systematyczne przesunięcie interpretacji [D, S, E]
        coherence_threshold: Próg spójności dla tego obserwatora (0.0-1.0)
        topology_metric: Metryka używana przez obserwatora
        register: Rejestr językowy obserwatora
        temperature: "Temperatura" interpretacji (rozrzut/niepewność)
        history: Historia obserwacji (dla adaptacyjnego uczenia)
    """
    id: str
    interpretation_bias: np.ndarray  # [3] - przesunięcie w przestrzeni D-S-E
    coherence_threshold: float       # 0.0-1.0
    topology_metric: str             # 'euclidean' | 'phi9' | 'taxicab'
    register: str                    # 'formal' | 'casual' | 'poetic' | 'legal' | ...

    # Parametry dynamiczne
    temperature: float = 1.0         # Domyślnie: standardowa interpretacja
    history: List[Dict] = field(default_factory=list)

    def __post_init__(self):
        """Walidacja parametrów obserwatora."""
        if not isinstance(self.interpretation_bias, np.ndarray):
            self.interpretation_bias = np.array(self.interpretation_bias)

        if len(self.interpretation_bias) != 3:
            raise ValueError(f"interpretation_bias must have 3 components (D,S,E), got {len(self.interpretation_bias)}")

        if not 0.0 <= self.coherence_threshold <= 1.0:
            raise ValueError(f"coherence_threshold must be in [0,1], got {self.coherence_threshold}")

        if self.topology_metric not in ['euclidean', 'phi9', 'taxicab']:
            logger.warning(f"Unknown topology_metric '{self.topology_metric}', using 'euclidean'")
            self.topology_metric = 'euclidean'

    def apply_interpretation(self, base_coords: np.ndarray) -> np.ndarray:
        """
        Aplikuje bias obserwatora do bazowych współrzędnych.
        Wynik może wykroczyć poza [0,1]³ (AlienatedNumber).

        Args:
            base_coords: Bazowe współrzędne [D, S, E] z analizy GTMØ

        Returns:
            Lokalne współrzędne φ_O (mogą być poza [0,1]³)
        """
        # Zastosuj bias z uwzględnieniem temperatury
        local_coords = base_coords + (self.temperature * self.interpretation_bias)

        # Dodaj losowy szum jeśli temperatura > 1
        if self.temperature > 1.0:
            noise_scale = (self.temperature - 1.0) * 0.1
            noise = np.random.normal(0, noise_scale, size=3)
            local_coords += noise

        return local_coords

    def record_observation(self, text: str, local_coords: np.ndarray, metadata: Optional[Dict] = None):
        """Zapisuje obserwację w historii obserwatora."""
        observation = {
            'timestamp': time.time(),
            'text': text,
            'local_coords': local_coords.tolist(),
            'metadata': metadata or {}
        }
        self.history.append(observation)

        # Ogranicz rozmiar historii
        if len(self.history) > 1000:
            self.history.pop(0)

    def __repr__(self) -> str:
        return (f"Observer(id={self.id}, register={self.register}, "
                f"metric={self.topology_metric}, threshold={self.coherence_threshold:.2f})")


# =============================================================================
# ALIENATED NUMBER - Wartość przedemergentna
# =============================================================================

@dataclass
class AlienatedNumber:
    """
    Wartość semantyczna PRZED emergencją globalnej.
    Istnieje tylko w lokalnym kole semantycznym 𝕂_O obserwatora.

    Kluczowa własność: MOŻE wykraczać poza [0,1]³

    Interpretacja wartości poza [0,1]³:
    - D < 0: brak ustalonej denotacji (słowo jeszcze nie "istnieje")
    - D > 1: nadmiarowa determinacja (wiele sprzecznych znaczeń)
    - S < 0: niestabilność semantyczna (znaczenie płynie)
    - S > 1: nadmierna stabilność (skostniałe znaczenie)
    - E < 0: nadmiar struktury (zbyt określone)
    - E > 1: semantyczny chaos (ponad maksimum entropii)

    Attributes:
        local_value: Współrzędne [D, S, E] w lokalnym kole 𝕂_O
        observer_id: ID obserwatora, którego to interpretacja
        n_observers: Liczba obserwatorów, którzy widzieli to słowo
        synchronization_energy: Energia desynchronizacji V_Comm
        can_collapse: Czy spełnia warunek emergencji (n≥2 i energia<ε)
        collapse_direction: Gradient ∇V (kierunek w którym ma kolapsować)
        timestamp: Kiedy utworzono tę wartość
    """
    local_value: np.ndarray
    observer_id: str
    n_observers: int
    synchronization_energy: float
    can_collapse: bool
    collapse_direction: Optional[np.ndarray] = None
    timestamp: float = field(default_factory=time.time)

    def __post_init__(self):
        """Walidacja i konwersja typów."""
        if not isinstance(self.local_value, np.ndarray):
            self.local_value = np.array(self.local_value)

        if len(self.local_value) != 3:
            raise ValueError(f"local_value must have 3 components, got {len(self.local_value)}")

    def is_emerged(self) -> bool:
        """
        Czy wartość zemergowała do globalnej φ_∞?
        Wymaga: n≥2 i can_collapse=True
        """
        return self.n_observers >= 2 and self.can_collapse

    def is_standard(self) -> bool:
        """Czy wartość mieści się w standardowej przestrzeni [0,1]³?"""
        return bool(np.all(self.local_value >= 0) and np.all(self.local_value <= 1))

    def alienation_magnitude(self) -> float:
        """
        Miara "alienacji" - jak daleko od [0,1]³?

        Returns:
            0.0 jeśli wartość w [0,1]³, inaczej odległość od najbliższego punktu
        """
        if self.is_standard():
            return 0.0

        # Znajdź najbliższy punkt w [0,1]³
        clamped = np.clip(self.local_value, 0, 1)

        # Odległość od tego punktu
        distance = np.linalg.norm(self.local_value - clamped)

        return float(distance)

    def get_interpretation(self) -> Dict[str, str]:
        """
        Zwraca interpretację semantyczną wartości.
        Szczególnie użyteczne dla wartości poza [0,1]³.
        """
        D, S, E = self.local_value

        interpretation = {}

        # Determination
        if D < 0:
            interpretation['D'] = f"negative ({D:.2f}) - no established denotation"
        elif D > 1:
            interpretation['D'] = f"exceeds 1.0 ({D:.2f}) - overdetermined meaning"
        else:
            interpretation['D'] = f"standard ({D:.2f})"

        # Stability
        if S < 0:
            interpretation['S'] = f"negative ({S:.2f}) - semantic fluidity"
        elif S > 1:
            interpretation['S'] = f"exceeds 1.0 ({S:.2f}) - ossified meaning"
        else:
            interpretation['S'] = f"standard ({S:.2f})"

        # Entropy
        if E < 0:
            interpretation['E'] = f"negative ({E:.2f}) - excessive structure"
        elif E > 1:
            interpretation['E'] = f"exceeds 1.0 ({E:.2f}) - semantic chaos"
        else:
            interpretation['E'] = f"standard ({E:.2f})"

        return interpretation

    def to_dict(self) -> Dict[str, Any]:
        """Serializacja do słownika (np. dla JSON)."""
        return {
            'local_value': self.local_value.tolist(),
            'observer_id': self.observer_id,
            'n_observers': self.n_observers,
            'synchronization_energy': float(self.synchronization_energy),
            'can_collapse': self.can_collapse,
            'is_emerged': self.is_emerged(),
            'is_standard': self.is_standard(),
            'alienation_magnitude': self.alienation_magnitude(),
            'collapse_direction': self.collapse_direction.tolist() if self.collapse_direction is not None else None,
            'interpretation': self.get_interpretation(),
            'timestamp': self.timestamp
        }

    def __repr__(self) -> str:
        status = "emerged" if self.is_emerged() else "alienated"
        standard = "std" if self.is_standard() else f"alien({self.alienation_magnitude():.2f})"
        return (f"AlienatedNumber({status}, {standard}, "
                f"n={self.n_observers}, E_sync={self.synchronization_energy:.3f})")


# =============================================================================
# ADELIC RING - Pierścień semantyczny słowa
# =============================================================================

class AdelicRing:
    """
    Pierścień Adeli dla konkretnego słowa/tekstu.
    𝔸_sem(w) = ℝ_sem × ∏'_{O∈Observers} 𝕂_O

    Przechowuje:
    1. Wartość globalną φ_∞(w) ∈ [0,1]³ (jeśli zemergowała)
    2. Wszystkie lokalne wartości φ_O(w) dla każdego obserwatora O
    3. Historię prób emergencji

    Kluczowe operacje:
    - add_observer_interpretation(): dodaje lokalną interpretację
    - attempt_emergence(): próbuje emergencji adelicznej
    - compute_synchronization_energy(): oblicza V_Comm
    - compute_collapse_gradients(): oblicza ∇V dla każdego obserwatora
    """

    def __init__(self, word: str, base_coords: np.ndarray):
        """
        Inicjalizacja pierścienia adelicznego dla słowa.

        Args:
            word: Słowo/tekst
            base_coords: Bazowe współrzędne z analizy GTMØ [D, S, E]
        """
        self.word = word
        self.base_coords = np.array(base_coords) if not isinstance(base_coords, np.ndarray) else base_coords

        # Wartości semantyczne
        self.global_value: Optional[np.ndarray] = None  # φ_∞ - emerguje przy n≥2
        self.local_values: Dict[str, AlienatedNumber] = {}  # φ_O dla każdego O

        # Historia
        self.emergence_history: List[Dict] = []
        self.creation_time = time.time()

        # Statystyki
        self.emergence_attempts = 0
        self.successful_emergences = 0

    def add_observer_interpretation(
        self,
        observer: Observer,
        local_coords: Optional[np.ndarray] = None
    ) -> AlienatedNumber:
        """
        Dodaje lokalną interpretację obserwatora do pierścienia.

        Args:
            observer: Obserwator
            local_coords: Lokalne współrzędne (jeśli None, oblicza z base_coords + bias)

        Returns:
            AlienatedNumber reprezentujący lokalną wartość
        """
        # Oblicz lokalne współrzędne jeśli nie podano
        if local_coords is None:
            local_coords = observer.apply_interpretation(self.base_coords)

        # Utwórz AlienatedNumber
        alienated = AlienatedNumber(
            local_value=local_coords,
            observer_id=observer.id,
            n_observers=len(self.local_values) + 1,
            synchronization_energy=0.0,  # Obliczone później
            can_collapse=False,
            collapse_direction=None,
            timestamp=time.time()
        )

        # Dodaj do lokalnych wartości
        self.local_values[observer.id] = alienated

        # Aktualizuj n_observers dla wszystkich
        for av in self.local_values.values():
            av.n_observers = len(self.local_values)

        # Zapisz w historii obserwatora
        observer.record_observation(
            text=self.word,
            local_coords=local_coords,
            metadata={'ring_id': id(self)}
        )

        logger.debug(f"Added interpretation from {observer.id} to ring '{self.word}'")

        return alienated

    def compute_synchronization_energy(self, metric: str = 'phi9') -> float:
        """
        Oblicza potencjał komunikacyjny V_Comm.

        V_Comm = (1/2) κ_comm · (1/n(n-1)) · Σᵢ<ⱼ d(φᵢ, φⱼ)²

        Args:
            metric: Metryka do użycia ('phi9' lub 'euclidean')

        Returns:
            Energia synchronizacyjna (0 = pełny konsensus, ∞ = desynchronizacja)
        """
        if len(self.local_values) < 2:
            return 0.0

        # Import metryki (aby uniknąć circular import)
        from gtmo_adelic_metrics import phi9_distance

        coords_list = [av.local_value for av in self.local_values.values()]
        n = len(coords_list)

        total_energy = 0.0
        kappa_comm = 1.0  # Stała komunikacyjna

        # Suma po wszystkich parach
        for i in range(n):
            for j in range(i + 1, n):
                if metric == 'phi9':
                    dist = phi9_distance(coords_list[i], coords_list[j])
                else:
                    dist = np.linalg.norm(coords_list[i] - coords_list[j])

                total_energy += dist ** 2

        # Normalizacja przez liczbę par
        num_pairs = n * (n - 1) / 2

        energy = 0.5 * kappa_comm * (total_energy / num_pairs)

        # Aktualizuj energię w AlienatedNumbers
        for av in self.local_values.values():
            av.synchronization_energy = energy

        return energy

    def attempt_emergence(
        self,
        epsilon: float = 0.15,
        context_attractor: Optional[np.ndarray] = None,
        metric: str = 'phi9'
    ) -> Tuple[bool, Optional[np.ndarray]]:
        """
        Próbuje adelicznej emergencji wartości globalnej.

        Warunek emergencji:
        ∀ O ∈ Observers: d(φ_O, φ_consensus) ≤ ε ⟹ ∃!φ_∞ ∈ [0,1]³

        Args:
            epsilon: Próg synchronizacji
            context_attractor: Opcjonalny atraktor kontekstowy (np. Ψᴷ)
            metric: Metryka do użycia

        Returns:
            (success, global_value)
        """
        self.emergence_attempts += 1

        if len(self.local_values) < 2:
            logger.debug(f"Ring '{self.word}': Cannot emerge with n={len(self.local_values)} < 2")
            return False, None

        # Import metryki
        from gtmo_adelic_metrics import phi9_distance

        # Oblicz consensus jako centroid
        coords_list = [av.local_value for av in self.local_values.values()]
        consensus = np.mean(coords_list, axis=0)

        # Sprawdź warunek adeliczny
        all_close = True
        max_distance = 0.0

        for coords in coords_list:
            if metric == 'phi9':
                distance = phi9_distance(coords, consensus)
            else:
                distance = np.linalg.norm(coords - consensus)

            max_distance = max(max_distance, distance)

            if distance > epsilon:
                all_close = False
                # Nie przerywaj - chcemy wiedzieć max_distance dla logowania

        if all_close:
            # EMERGENCJA!
            self.global_value = np.clip(consensus, 0, 1)
            self.successful_emergences += 1

            # Oznacz wszystkie AlienatedNumbers jako collapsed
            for av in self.local_values.values():
                av.can_collapse = True

            # Zaloguj event emergencji
            emergence_event = {
                'timestamp': time.time(),
                'n_observers': len(self.local_values),
                'consensus': consensus.tolist(),
                'global_value': self.global_value.tolist(),
                'synchronization_energy': self.compute_synchronization_energy(metric),
                'epsilon': epsilon,
                'max_distance': max_distance,
                'metric': metric
            }
            self.emergence_history.append(emergence_event)

            logger.info(f"Ring '{self.word}': EMERGENCE! φ_∞ = {self.global_value} "
                       f"(n={len(self.local_values)}, E_sync={emergence_event['synchronization_energy']:.3f})")

            return True, self.global_value

        else:
            logger.debug(f"Ring '{self.word}': No emergence (max_dist={max_distance:.3f} > ε={epsilon})")
            return False, None

    def compute_collapse_gradients(
        self,
        context_attractor: np.ndarray,
        kappa_comm: float = 1.0,
        kappa_context: float = 0.5
    ) -> Dict[str, np.ndarray]:
        """
        Oblicza gradient kolapsu dla każdego obserwatora.

        ∇V_total = κ_comm (φ_O - φ_consensus) + κ_context (φ_O - Ψ_attractor)

        Gradient wskazuje kierunek, w którym lokalna wartość powinna się
        przesunąć, aby osiągnąć emergencję.

        Args:
            context_attractor: Współrzędne attraktora kontekstowego (Ψᴷ, Ψˢ, etc.)
            kappa_comm: Stała komunikacyjna
            kappa_context: Stała kontekstowa

        Returns:
            Dict {observer_id: gradient_vector}
        """
        if len(self.local_values) < 2:
            return {}

        coords_list = [av.local_value for av in self.local_values.values()]
        consensus = np.mean(coords_list, axis=0)

        gradients = {}

        for obs_id, alienated in self.local_values.items():
            # Gradient komunikacyjny (pcha w stronę consensus)
            grad_comm = kappa_comm * (alienated.local_value - consensus)

            # Gradient kontekstowy (pcha w stronę attraktora)
            grad_context = kappa_context * (alienated.local_value - context_attractor)

            # Gradient łączny (minus - kierunek spadku energii)
            total_gradient = -(grad_comm + grad_context)

            gradients[obs_id] = total_gradient

            # Zapisz w AlienatedNumber
            alienated.collapse_direction = total_gradient

        return gradients

    def get_status(self) -> Dict[str, Any]:
        """Zwraca status pierścienia adelicznego."""
        return {
            'word': self.word,
            'base_coords': self.base_coords.tolist(),
            'n_observers': len(self.local_values),
            'has_emerged': self.global_value is not None,
            'global_value': self.global_value.tolist() if self.global_value is not None else None,
            'synchronization_energy': self.compute_synchronization_energy(),
            'emergence_attempts': self.emergence_attempts,
            'successful_emergences': self.successful_emergences,
            'age_seconds': time.time() - self.creation_time
        }

    def __repr__(self) -> str:
        status = "emerged" if self.global_value is not None else "alienated"
        return f"AdelicRing('{self.word}', {status}, n={len(self.local_values)})"


# =============================================================================
# POMOCNICZE FUNKCJE
# =============================================================================

def create_standard_observers() -> List[Observer]:
    """
    Tworzy standardowy zestaw obserwatorów dla typowych rejestrów języka.

    Returns:
        Lista 5 standardowych obserwatorów
    """
    observers = [
        Observer(
            id="O_formal",
            interpretation_bias=np.array([0.10, 0.08, -0.06]),
            coherence_threshold=0.85,
            topology_metric='euclidean',
            register='formal'
        ),
        Observer(
            id="O_legal",
            interpretation_bias=np.array([0.15, 0.12, -0.08]),
            coherence_threshold=0.88,
            topology_metric='euclidean',
            register='legal'
        ),
        Observer(
            id="O_casual",
            interpretation_bias=np.array([0.05, 0.00, 0.05]),
            coherence_threshold=0.70,
            topology_metric='euclidean',
            register='casual'
        ),
        Observer(
            id="O_poetic",
            interpretation_bias=np.array([-0.20, -0.15, 0.25]),
            coherence_threshold=0.55,
            topology_metric='phi9',
            register='poetic'
        ),
        Observer(
            id="O_sarcastic",
            interpretation_bias=np.array([-0.25, -0.18, 0.30]),
            coherence_threshold=0.50,
            topology_metric='phi9',
            register='sarcastic'
        ),

        # OBSERWATORZY DOMENOWI
        Observer(
            id="O_medical",
            interpretation_bias=np.array([0.18, 0.15, -0.10]),
            coherence_threshold=0.90,
            topology_metric='euclidean',
            register='medical',
            temperature=0.95  # Bardzo precyzyjny
        ),
        Observer(
            id="O_technical",
            interpretation_bias=np.array([0.20, 0.18, -0.12]),
            coherence_threshold=0.92,
            topology_metric='euclidean',
            register='technical',
            temperature=0.90  # Ekstremalnie precyzyjny
        ),
        Observer(
            id="O_legal_strict",
            interpretation_bias=np.array([0.22, 0.20, -0.15]),
            coherence_threshold=0.95,
            topology_metric='euclidean',
            register='legal_strict',
            temperature=0.85  # Najściślejszy
        ),
        Observer(
            id="O_philosophical",
            interpretation_bias=np.array([0.00, -0.10, 0.20]),
            coherence_threshold=0.60,
            topology_metric='phi9',
            register='philosophical',
            temperature=1.2  # Pozwala na niepewność
        ),
        Observer(
            id="O_journalistic",
            interpretation_bias=np.array([0.08, 0.05, 0.08]),
            coherence_threshold=0.75,
            topology_metric='euclidean',
            register='journalistic',
            temperature=1.0
        ),

        # OBSERWATORZY SPECJALNI
        Observer(
            id="O_nonsense",
            interpretation_bias=np.array([-0.35, -0.40, 0.60]),
            coherence_threshold=0.30,
            topology_metric='phi9',
            register='nonsense',
            temperature=1.8  # Bardzo wysoka temperatura - duży rozrzut
        ),
        Observer(
            id="O_hallucination",
            interpretation_bias=np.array([-0.50, -0.60, 0.80]),
            coherence_threshold=0.15,
            topology_metric='phi9',
            register='hallucination',
            temperature=2.5  # Ekstremalny rozrzut - halucynacje
        ),
        Observer(
            id="O_conspiracy",
            interpretation_bias=np.array([-0.30, -0.35, 0.55]),
            coherence_threshold=0.40,
            topology_metric='phi9',
            register='conspiracy',
            temperature=1.6  # Wysokie niepewności, przesunięcie w stronę chaosu
        ),
        Observer(
            id="O_propaganda",
            interpretation_bias=np.array([0.25, 0.30, -0.20]),
            coherence_threshold=0.65,
            topology_metric='euclidean',
            register='propaganda',
            temperature=0.80  # Niska temperatura - "pewność" propagandy
        )
    ]

    return observers


def create_domain_observer(
    domain: str,
    strictness: float = 0.85,
    chaos_tolerance: float = 0.15
) -> Observer:
    """
    Tworzy obserwatora domenowego z automatycznym doborem parametrów.

    Args:
        domain: Nazwa domeny ('medical', 'legal', 'tech', 'art', etc.)
        strictness: Jak ściśle interpretuje (0=chaos, 1=absolute)
        chaos_tolerance: Tolerancja na entropię (0=zero, 1=full)

    Returns:
        Skonfigurowany Observer
    """
    # Automatyczne mapowanie strictness na bias
    bias_d = 0.10 + (strictness - 0.5) * 0.3
    bias_s = 0.08 + (strictness - 0.5) * 0.25
    bias_e = -0.05 + (chaos_tolerance - 0.5) * 0.4

    coherence = 0.50 + strictness * 0.45
    temp = 1.0 + (1.0 - strictness) * 0.8

    metric = 'euclidean' if strictness > 0.7 else 'phi9'

    return Observer(
        id=f"O_{domain}",
        interpretation_bias=np.array([bias_d, bias_s, bias_e]),
        coherence_threshold=coherence,
        topology_metric=metric,
        register=domain,
        temperature=temp
    )


# =============================================================================
# ADELIC SEMANTIC LAYER - Główna fasada
# =============================================================================

class AdelicSemanticLayer:
    """
    Główna fasada dla warstwy adelicznej.
    Zarządza obserwatorami, pierścieniami adelicznymi i udostępnia high-level API.

    Integruje się z GTMOMorphosyntaxEngine.

    Attributes:
        observers: Lista aktywnych obserwatorów
        epsilon: Próg adelicznej emergencji
        kappa_comm: Stała komunikacyjna
        kappa_context: Stała kontekstowa
        adelic_rings: Cache pierścieni adelicznych
    """

    def __init__(
        self,
        default_observers: Optional[List[Observer]] = None,
        epsilon: float = 0.15,
        kappa_comm: float = 1.0,
        kappa_context: float = 0.5,
        cache_size: int = 1000,
        use_energy_threshold: bool = True,
        energy_threshold_emerged: float = 100.0,
        energy_threshold_borderline: float = 150.0
    ):
        """
        Inicjalizacja warstwy adelicznej.

        Args:
            default_observers: Domyślni obserwatorzy (jeśli None, tworzy standardowy zestaw)
            epsilon: Próg emergencji adelicznej (używany jeśli use_energy_threshold=False)
            kappa_comm: Stała komunikacyjna
            kappa_context: Stała kontekstowa
            cache_size: Maksymalny rozmiar cache pierścieni
            use_energy_threshold: Czy używać progów V_Comm zamiast epsilon (zalecane)
            energy_threshold_emerged: Próg V_Comm dla emergencji (domyślnie 100)
            energy_threshold_borderline: Próg V_Comm dla borderline (domyślnie 150)
        """
        self.observers = default_observers if default_observers is not None else create_standard_observers()
        self.epsilon = epsilon
        self.base_epsilon = epsilon  # Zachowaj oryginalny epsilon
        self.kappa_comm = kappa_comm
        self.kappa_context = kappa_context
        self.cache_size = cache_size

        # Kalibracja energii V_Comm
        self.use_energy_threshold = use_energy_threshold
        self.energy_threshold_emerged = energy_threshold_emerged
        self.energy_threshold_borderline = energy_threshold_borderline
        self.observed_energies: List[float] = []  # Historia energii dla auto-kalibracji
        self.calibrated = False

        # Cache pierścieni adelicznych {word: AdelicRing}
        self.adelic_rings: Dict[str, AdelicRing] = {}

        # Statystyki
        self.total_analyses = 0
        self.successful_emergences = 0
        self.failed_emergences = 0

        if use_energy_threshold:
            logger.info(f"AdelicSemanticLayer initialized with {len(self.observers)} observers, "
                       f"V_Comm thresholds: emerged<{energy_threshold_emerged}, "
                       f"borderline<{energy_threshold_borderline}")
        else:
            logger.info(f"AdelicSemanticLayer initialized with {len(self.observers)} observers, ε={epsilon}")

    def analyze_with_observers(
        self,
        text: str,
        base_coords: np.ndarray,
        observers: Optional[List[Observer]] = None,
        context_attractor: Optional[np.ndarray] = None,
        context_name: Optional[str] = None,
        metric: str = 'phi9'
    ) -> Dict[str, Any]:
        """
        Analiza tekstu z warstwą adeliczną.

        Wykonuje:
        1. Aplikację biasów obserwatorów → lokalne interpretacje
        2. Próbę emergencji adelicznej (jeśli n ≥ 2)
        3. Obliczenie V_Comm i gradientów
        4. Rekomendację kontekstu (jeśli brak emergencji)

        Args:
            text: Tekst do analizy
            base_coords: Bazowe współrzędne z GTMØ [D, S, E]
            observers: Lista obserwatorów (jeśli None, użyj domyślnych)
            context_attractor: Współrzędne attraktora kontekstowego
            context_name: Nazwa attraktora (np. 'Ψᴷ')
            metric: Metryka do użycia ('phi9' lub 'euclidean')

        Returns:
            Dict z wynikami analizy adelicznej:
            {
                'emerged': bool,
                'global_value': Optional[np.ndarray],
                'local_values': Dict[observer_id, coords],
                'synchronization_energy': float,
                'n_observers': int,
                'collapse_gradients': Optional[Dict],
                'recommended_context': Optional[str],
                'status': str
            }
        """
        self.total_analyses += 1

        # Użyj domyślnych obserwatorów jeśli nie podano
        obs_list = observers if observers is not None else self.observers

        if len(obs_list) == 0:
            logger.warning("No observers provided for adelic analysis")
            return {
                'emerged': False,
                'error': 'no_observers',
                'n_observers': 0
            }

        # Pobierz lub utwórz pierścień adeliczny
        ring = self.get_or_create_ring(text, base_coords)

        # Dodaj interpretacje obserwatorów
        for observer in obs_list:
            ring.add_observer_interpretation(observer)

        # Oblicz energię synchronizacyjną
        sync_energy = ring.compute_synchronization_energy(metric=metric)

        # Zapisz energię dla kalibracji
        self.observed_energies.append(sync_energy)

        # Próba emergencji - użyj V_Comm threshold jeśli włączony
        if self.use_energy_threshold:
            # Nowy mechanizm: porównaj V_Comm z progami energii
            if sync_energy < self.energy_threshold_emerged:
                emerged = True
                status = 'emerged'
                # Oblicz globalną wartość jako consensus
                coords_list = [av.local_value for av in ring.local_values.values()]
                global_value = np.clip(np.mean(coords_list, axis=0), 0, 1)
                # Oznacz jako collapsed
                for av in ring.local_values.values():
                    av.can_collapse = True
            elif sync_energy < self.energy_threshold_borderline:
                emerged = False
                status = 'borderline'
                global_value = None
            else:
                emerged = False
                status = 'alienated'
                global_value = None
        else:
            # Stary mechanizm: użyj epsilon i pairwise distances
            emerged, global_value = ring.attempt_emergence(
                epsilon=self.epsilon,
                context_attractor=context_attractor,
                metric=metric
            )
            status = 'emerged' if emerged else 'alienated'

        # Przygotuj wynik
        result = {
            'emerged': emerged,
            'global_value': global_value.tolist() if global_value is not None else None,
            'local_values': {
                obs_id: av.to_dict()
                for obs_id, av in ring.local_values.items()
            },
            'synchronization_energy': sync_energy,
            'n_observers': len(ring.local_values),
            'metric': metric,
            'text': text,
            'status': status
        }

        # Dodaj informacje o progach jeśli używamy energy threshold
        if self.use_energy_threshold:
            result['energy_threshold_emerged'] = self.energy_threshold_emerged
            result['energy_threshold_borderline'] = self.energy_threshold_borderline
            result['calibrated'] = self.calibrated
        else:
            result['epsilon'] = self.epsilon

        # Aktualizuj statystyki
        if emerged:
            self.successful_emergences += 1
        else:
            self.failed_emergences += 1

            # Oblicz gradienty kolapsu jeśli mamy atraktor kontekstowy
            if context_attractor is not None and len(ring.local_values) >= 2:
                gradients = ring.compute_collapse_gradients(
                    context_attractor=context_attractor,
                    kappa_comm=self.kappa_comm,
                    kappa_context=self.kappa_context
                )
                result['collapse_gradients'] = {
                    obs_id: grad.tolist() for obs_id, grad in gradients.items()
                }
                result['context_attractor'] = context_name

            # Diagnoza niepowodzenia (tylko jeśli nie używamy energy threshold)
            if not self.use_energy_threshold:
                from gtmo_adelic_metrics import diagnose_emergence_failure
                diagnosis = diagnose_emergence_failure(
                    [av.local_value for av in ring.local_values.values()],
                    epsilon=self.epsilon,
                    metric=metric
                )
                result['diagnosis'] = diagnosis

        return result

    def compute_dialogue_energy(
        self,
        utterances: List[str],
        base_coords_list: List[np.ndarray],
        speaker_a_observer: Observer,
        speaker_b_observer: Observer,
        metric: str = 'phi9'
    ) -> Dict[str, Any]:
        """
        Oblicza całkowitą energię komunikacyjną dialogu.

        Symuluje komunikację między dwoma rozmówcami (obserwatorami).
        Wysoka energia = trudna komunikacja (rozbieżności interpretacyjne).

        Args:
            utterances: Lista wypowiedzi w dialogu
            base_coords_list: Lista bazowych współrzędnych dla każdej wypowiedzi
            speaker_a_observer: Obserwator reprezentujący mówcę A
            speaker_b_observer: Obserwator reprezentujący mówcę B
            metric: Metryka

        Returns:
            Dict z energią dialogu i szczegółami
        """
        if len(utterances) != len(base_coords_list):
            raise ValueError("Number of utterances must match number of base_coords")

        total_energy = 0.0
        utterance_energies = []

        for utterance, base_coords in zip(utterances, base_coords_list):
            # Analiza z dwoma obserwatorami
            result = self.analyze_with_observers(
                text=utterance,
                base_coords=base_coords,
                observers=[speaker_a_observer, speaker_b_observer],
                metric=metric
            )

            utterance_energies.append({
                'utterance': utterance,
                'energy': result['synchronization_energy'],
                'emerged': result['emerged']
            })

            total_energy += result['synchronization_energy']

        # Średnia energia
        avg_energy = total_energy / len(utterances) if utterances else 0.0

        # Klasyfikacja trudności komunikacyjnej
        if avg_energy < 0.1:
            difficulty = 'easy'
        elif avg_energy < 0.5:
            difficulty = 'moderate'
        elif avg_energy < 1.0:
            difficulty = 'hard'
        else:
            difficulty = 'very_hard'

        return {
            'total_energy': total_energy,
            'average_energy': avg_energy,
            'num_utterances': len(utterances),
            'difficulty': difficulty,
            'utterance_details': utterance_energies,
            'speaker_a': speaker_a_observer.id,
            'speaker_b': speaker_b_observer.id
        }

    def track_semantic_drift(
        self,
        word: str,
        observations: List[Tuple[float, np.ndarray, Observer]],
        metric: str = 'phi9'
    ) -> Dict[str, Any]:
        """
        Śledzi dryf semantyczny słowa w czasie.

        Args:
            word: Słowo do śledzenia
            observations: Lista (timestamp, coords, observer)
            metric: Metryka

        Returns:
            Dict z analizą dryfu semantycznego
        """
        if len(observations) < 2:
            return {'error': 'Need at least 2 observations'}

        # Sortuj po czasie
        observations = sorted(observations, key=lambda x: x[0])

        # Oblicz trajektorię
        trajectory = []
        energies = []

        for i in range(len(observations) - 1):
            t1, coords1, obs1 = observations[i]
            t2, coords2, obs2 = observations[i + 1]

            # Odległość między kolejnymi obserwacjami
            from gtmo_adelic_metrics import phi9_distance
            if metric == 'phi9':
                dist = phi9_distance(coords1, coords2)
            else:
                dist = np.linalg.norm(coords1 - coords2)

            # Velocity (prędkość dryfu)
            dt = t2 - t1
            velocity = dist / dt if dt > 0 else 0.0

            trajectory.append({
                'timestamp': t2,
                'coords': coords2.tolist(),
                'observer': obs2.id,
                'distance_from_previous': dist,
                'time_delta': dt,
                'drift_velocity': velocity
            })

            energies.append(dist)

        # Całkowity dryf
        total_drift = sum(energies)
        avg_velocity = np.mean([t['drift_velocity'] for t in trajectory])

        return {
            'word': word,
            'num_observations': len(observations),
            'time_span': observations[-1][0] - observations[0][0],
            'total_drift': total_drift,
            'average_velocity': avg_velocity,
            'trajectory': trajectory,
            'is_stable': total_drift < 0.3,  # Arbitralny próg
            'metric': metric
        }

    def calibrate_epsilon(self, min_samples: int = 20, percentile_emerged: float = 40, percentile_borderline: float = 70):
        """
        Automatyczna kalibracja progów energii na podstawie obserwowanych wartości V_Comm.

        Analizuje rozkład energii i ustawia progi tak, aby:
        - Teksty faktyczne (niskie energie) były klasyfikowane jako EMERGED
        - Teksty patologiczne (wysokie energie) jako ALIENATED
        - Środkowe jako BORDERLINE

        Args:
            min_samples: Minimalna liczba obserwacji do kalibracji
            percentile_emerged: Percentyl dla progu emergencji (domyślnie 40)
            percentile_borderline: Percentyl dla progu borderline (domyślnie 70)

        Returns:
            True jeśli kalibracja się powiodła, False jeśli za mało danych
        """
        if len(self.observed_energies) < min_samples:
            logger.warning(f"Not enough observations for calibration ({len(self.observed_energies)} < {min_samples})")
            return False

        energies = np.array(self.observed_energies)

        # Oblicz percentyle
        p_emerged = np.percentile(energies, percentile_emerged)
        p_borderline = np.percentile(energies, percentile_borderline)

        # Statystyki dla logowania
        mean_energy = np.mean(energies)
        median_energy = np.median(energies)
        std_energy = np.std(energies)
        min_energy = np.min(energies)
        max_energy = np.max(energies)

        logger.info(f"Calibration based on {len(energies)} observations:")
        logger.info(f"  V_Comm range: [{min_energy:.1f}, {max_energy:.1f}]")
        logger.info(f"  Mean: {mean_energy:.1f}, Median: {median_energy:.1f}, Std: {std_energy:.1f}")
        logger.info(f"  Percentiles: P{percentile_emerged}={p_emerged:.1f}, P{percentile_borderline}={p_borderline:.1f}")

        # Ustaw nowe progi
        self.energy_threshold_emerged = float(p_emerged)
        self.energy_threshold_borderline = float(p_borderline)
        self.calibrated = True

        logger.info(f"✅ Calibrated thresholds: EMERGED<{self.energy_threshold_emerged:.1f}, "
                   f"BORDERLINE<{self.energy_threshold_borderline:.1f}, ALIENATED≥{self.energy_threshold_borderline:.1f}")

        return True

    def get_or_create_ring(self, word: str, base_coords: np.ndarray) -> AdelicRing:
        """
        Pobiera istniejący lub tworzy nowy pierścień adeliczny dla słowa.

        Args:
            word: Słowo
            base_coords: Bazowe współrzędne

        Returns:
            AdelicRing
        """
        # Sprawdź cache
        if word in self.adelic_rings:
            ring = self.adelic_rings[word]
            # Resetuj ring jeśli bazowe współrzędne się zmieniły
            if not np.allclose(ring.base_coords, base_coords):
                logger.debug(f"Base coords changed for '{word}', creating new ring")
                ring = AdelicRing(word, base_coords)
                self.adelic_rings[word] = ring
            return ring

        # Utwórz nowy
        ring = AdelicRing(word, base_coords)
        self.adelic_rings[word] = ring

        # Ogranicz rozmiar cache (LRU-style - usuń najstarszy)
        if len(self.adelic_rings) > self.cache_size:
            oldest_word = min(
                self.adelic_rings.keys(),
                key=lambda w: self.adelic_rings[w].creation_time
            )
            del self.adelic_rings[oldest_word]
            logger.debug(f"Cache full, removed oldest ring '{oldest_word}'")

        return ring

    def create_observer(
        self,
        observer_id: str,
        register: str,
        bias_d: float = 0.0,
        bias_s: float = 0.0,
        bias_e: float = 0.0,
        coherence_threshold: float = 0.75,
        topology_metric: str = 'euclidean',
        temperature: float = 1.0
    ) -> Observer:
        """
        Tworzy nowego obserwatora z podanymi parametrami.

        Args:
            observer_id: Unikalny ID
            register: Rejestr językowy
            bias_d, bias_s, bias_e: Komponenty biasu
            coherence_threshold: Próg spójności
            topology_metric: Metryka
            temperature: Temperatura interpretacji

        Returns:
            Nowy Observer
        """
        observer = Observer(
            id=observer_id,
            interpretation_bias=np.array([bias_d, bias_s, bias_e]),
            coherence_threshold=coherence_threshold,
            topology_metric=topology_metric,
            register=register,
            temperature=temperature
        )

        logger.info(f"Created observer: {observer}")
        return observer

    def get_statistics(self) -> Dict[str, Any]:
        """Zwraca statystyki warstwy adelicznej."""
        return {
            'total_analyses': self.total_analyses,
            'successful_emergences': self.successful_emergences,
            'failed_emergences': self.failed_emergences,
            'emergence_rate': (
                self.successful_emergences / self.total_analyses
                if self.total_analyses > 0 else 0.0
            ),
            'num_observers': len(self.observers),
            'cache_size': len(self.adelic_rings),
            'epsilon': self.epsilon,
            'kappa_comm': self.kappa_comm,
            'kappa_context': self.kappa_context
        }

    def clear_cache(self):
        """Czyści cache pierścieni adelicznych."""
        n_cleared = len(self.adelic_rings)
        self.adelic_rings.clear()
        logger.info(f"Cleared cache ({n_cleared} rings removed)")

    def __repr__(self) -> str:
        return (f"AdelicSemanticLayer(n_obs={len(self.observers)}, "
                f"ε={self.epsilon}, analyses={self.total_analyses})")


# =============================================================================
# TESTY MODUŁU
# =============================================================================

def run_tests():
    """Uruchamia testy wbudowane modułu."""
    print("=" * 60)
    print("GTMØ Adelic Layer - Test modułu")
    print("=" * 60)

    # Test 1: Observer
    print("\n[Test 1] Observer")
    obs = Observer(
        id="test_obs",
        interpretation_bias=np.array([0.1, 0.0, -0.1]),
        coherence_threshold=0.75,
        topology_metric='euclidean',
        register='test'
    )
    print(f"  {obs}")

    base_coords = np.array([0.7, 0.75, 0.3])
    local_coords = obs.apply_interpretation(base_coords)
    print(f"  Base: {base_coords}, Local: {local_coords}")

    # Test 2: AlienatedNumber
    print("\n[Test 2] AlienatedNumber")
    alienated = AlienatedNumber(
        local_value=np.array([-0.15, 0.22, 1.34]),
        observer_id="test_obs",
        n_observers=1,
        synchronization_energy=0.0,
        can_collapse=False
    )
    print(f"  {alienated}")
    print(f"  Is standard: {alienated.is_standard()}")
    print(f"  Alienation: {alienated.alienation_magnitude():.3f}")
    print(f"  Interpretation: {alienated.get_interpretation()}")

    # Test 3: AdelicRing
    print("\n[Test 3] AdelicRing")
    ring = AdelicRing(word="test", base_coords=np.array([0.7, 0.75, 0.3]))

    obs1 = create_standard_observers()[0]
    obs2 = create_standard_observers()[2]

    ring.add_observer_interpretation(obs1)
    ring.add_observer_interpretation(obs2)

    print(f"  {ring}")
    print(f"  Energy: {ring.compute_synchronization_energy():.3f}")

    success, global_val = ring.attempt_emergence(epsilon=0.15)
    print(f"  Emergence: {success}")
    if success:
        print(f"  φ_∞ = {global_val}")

    print("\n" + "=" * 60)
    print("✓ Moduł gtmo_adelic_layer.py załadowany pomyślnie")
    print("=" * 60)


def analyze_file_with_adelic(file_path: str, epsilon: float = None):
    """
    Analizuje plik tekstowy z warstwą adeliczną.

    Args:
        file_path: Ścieżka do pliku tekstowego
        epsilon: Próg emergencji (jeśli None, używa V_Comm thresholds)
    """
    from pathlib import Path

    # Wczytaj plik
    path = Path(file_path)
    if not path.exists():
        print(f"❌ File not found: {file_path}")
        return

    print(f"📂 Loading: {path.name}")
    with open(path, 'r', encoding='utf-8') as f:
        text = f.read()

    # Split na zdania (prosty split)
    import re
    sentences = re.split(r'[.!?]+\s+', text)
    sentences = [s.strip() for s in sentences if s.strip() and len(s.strip()) > 10]

    print(f"📄 Found {len(sentences)} sentences\n")

    # Inicjalizuj adelic layer
    if epsilon is None:
        print(f"🔮 Initializing Adelic Layer with V_Comm thresholds...")
        adelic = AdelicSemanticLayer(
            use_energy_threshold=True,
            energy_threshold_emerged=100.0,
            energy_threshold_borderline=150.0
        )
        print(f"   {len(adelic.observers)} observers loaded")
        print(f"   V_Comm thresholds: EMERGED<100, BORDERLINE<150, ALIENATED≥150\n")
    else:
        print(f"🔮 Initializing Adelic Layer (ε={epsilon})...")
        adelic = AdelicSemanticLayer(epsilon=epsilon, use_energy_threshold=False)
        print(f"   {len(adelic.observers)} observers loaded\n")

    # Analizuj każde zdanie przez GTMØ najpierw
    from gtmo_morphosyntax import analyze_dse_standard

    emergences = 0
    total = 0

    print("=" * 70)
    for idx, sentence in enumerate(sentences, 1):
        print(f"\n📝 Sentence {idx}/{len(sentences)}")
        print(f"   Text: {sentence[:80]}{'...' if len(sentence) > 80 else ''}")

        # GTMØ analysis (standard mode - bez quantum)
        try:
            gtmo_result = analyze_dse_standard(sentence)
            coords = gtmo_result['coordinates']
            base_coords = np.array([
                coords['determination'],
                coords['stability'],
                coords['entropy']
            ])

            print(f"   D-S-E: [{base_coords[0]:.3f}, {base_coords[1]:.3f}, {base_coords[2]:.3f}]")

            if 'ambiguity' in gtmo_result:
                print(f"   Ambiguity: {gtmo_result['ambiguity']:.3f}, Depth: {gtmo_result.get('depth', 0)}")

        except Exception as e:
            print(f"   ⚠️ GTMØ analysis failed: {e}")
            continue

        # Adelic analysis
        try:
            result = adelic.analyze_with_observers(
                text=sentence,
                base_coords=base_coords,
                metric='phi9'
            )

            total += 1
            emerged = result['emerged']
            status = result['status']
            if emerged:
                emergences += 1

            # Ikony statusu
            if status == 'emerged':
                status_icon = "✨"
            elif status == 'borderline':
                status_icon = "🟡"
            else:
                status_icon = "⚠️"

            print(f"   {status_icon} Adelic: {status.upper()}")
            print(f"      V_Comm Energy: {result['synchronization_energy']:.1f}")
            print(f"      Observers: {result['n_observers']}")

            if emerged:
                gv = result['global_value']
                print(f"      Global φ_∞: [{gv[0]:.3f}, {gv[1]:.3f}, {gv[2]:.3f}]")
            elif status == 'borderline':
                print(f"      ⚠️ Borderline - uncertain semantic convergence")

        except Exception as e:
            print(f"   ⚠️ Adelic analysis failed: {e}")

    print("\n" + "=" * 70)
    print(f"\n📊 Summary:")
    print(f"   Total sentences: {total}")
    print(f"   Emerged: {emergences} ({100*emergences/total if total > 0 else 0:.1f}%)")
    print(f"   Alienated: {total - emergences} ({100*(total-emergences)/total if total > 0 else 0:.1f}%)")

    if adelic.use_energy_threshold:
        print(f"   Method: V_Comm thresholds (EMERGED<{adelic.energy_threshold_emerged}, "
              f"BORDERLINE<{adelic.energy_threshold_borderline})")
    else:
        print(f"   Method: epsilon={epsilon}")


if __name__ == "__main__":
    import sys

    # Sprawdź argumenty
    if len(sys.argv) > 1:
        # Pre-import gtmo_morphosyntax BEFORE any stdout manipulation
        from gtmo_morphosyntax import analyze_dse_standard  # noqa: F401

        # Now safe to wrap stdout for emoji output
        import io
        if sys.platform == 'win32' and hasattr(sys.stdout, 'buffer'):
            sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')

        # Plik podany jako argument - analizuj
        file_path = sys.argv[1]

        # Opcjonalny epsilon jako drugi argument
        # Jeśli nie podano, użyje V_Comm thresholds (None)
        epsilon = float(sys.argv[2]) if len(sys.argv) > 2 else None

        analyze_file_with_adelic(file_path, epsilon=epsilon)
    else:
        # Brak argumentów - uruchom testy
        run_tests()
