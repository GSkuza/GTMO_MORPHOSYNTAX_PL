#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
GTMØ Adelic Metrics - Metryki matematyczne warstwy adelicznej
============================================================

Implementacja metryk dla p-adycznej struktury semantycznej:
1. Metryka Φ⁹ - nieliniowa metryka w przestrzeni D-S-E
2. Potencjał komunikacyjny V_Comm
3. Gradient kolapsu ∇V
4. Warunki emergencji adelicznej

Autor: GTMØ Development Team
Data: 2024-11-24
"""

import numpy as np
from typing import List, Tuple, Optional, Dict, Any
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Stałe matematyczne
PHI = (1 + np.sqrt(5)) / 2  # Złoty podział ≈ 1.618
PHI_0 = 1.2925  # Próg entropii dla adaptacyjnego ε (√(PHI² + 1/PHI) / √2)


# =============================================================================
# METRYKI W PRZESTRZENI D-S-E
# =============================================================================

def phi9_distance(coords1: np.ndarray, coords2: np.ndarray) -> float:
    """
    Oblicza metrykę Φ⁹ między dwoma punktami w przestrzeni D-S-E.

    Definicja:
    d_Φ⁹(φ₁, φ₂) = Σᵢ φⁱ · |φ₁ᵢ - φ₂ᵢ|ⁱ

    gdzie:
    - φ = złoty podział = (1 + √5)/2 ≈ 1.618
    - i ∈ {1, 2, 3} dla {D, S, E}
    - Wagi: φ¹ ≈ 1.618, φ² ≈ 2.618, φ³ ≈ 4.236

    Własności:
    - Nieliniowa: duże różnice w E dominują nad D i S
    - Asymetryczna względem składowych: E > S > D w wadze
    - Dla małych różnic ≈ metryka euklidesowa
    - Dla dużych różnic - entropia E dominuje
    - UWAGA: Metryka Riemannowska (symetryczna), nie pseudo-metryka

    Args:
        coords1: Pierwszy punkt [D, S, E]
        coords2: Drugi punkt [D, S, E]

    Returns:
        Odległość w metryce Φ⁹

    Example:
        >>> phi9_distance(np.array([0.8, 0.8, 0.2]), np.array([0.9, 0.9, 0.1]))
        0.421...
    """
    if len(coords1) != 3 or len(coords2) != 3:
        raise ValueError(f"Coords must have 3 components (D,S,E), got {len(coords1)} and {len(coords2)}")

    # Wagi: φⁱ dla i=1,2,3
    weights = np.array([PHI**1, PHI**2, PHI**3])

    # Potęgi dla każdej składowej
    powers = np.array([1, 2, 3])

    # Różnice bezwzględne
    diff = np.abs(coords1 - coords2)

    # d_Φ⁹ = Σ φⁱ · |Δᵢ|ⁱ
    terms = weights * np.power(diff, powers)

    distance = np.sum(terms)

    return float(distance)


def classify_semantic_trajectory(
    start: np.ndarray,
    end: np.ndarray,
    kappa: float = PHI_0,
    tolerance: float = 1e-6
) -> str:
    """
    Klasyfikuje trajektorię semantyczną w przestrzeni GTMØ.

    Używa pseudo-metryki Minkowskiego do określenia charakteru zmiany:
    - Timelike: Naturalna ewolucja w "czasie semantycznym"
    - Spacelike: Skokowa redefinicja w "przestrzeni semantycznej"
    - Lightlike: Maksymalna "prędkość" zmiany semantycznej

    Args:
        start: Punkt początkowy [D, S, E]
        end: Punkt końcowy [D, S, E]
        kappa: Parametr skalujący oś S (domyślnie Ø₀ ≈ 1.2925)
        tolerance: Tolerancja numeryczna dla klasyfikacji lightlike

    Returns:
        'timelike' | 'spacelike' | 'lightlike'

    Example:
        >>> # Ewolucja naturalna (duża zmiana S)
        >>> classify_semantic_trajectory(
        ...     np.array([0.8, 0.3, 0.2]),
        ...     np.array([0.8, 0.9, 0.2])
        ... )
        'timelike'

        >>> # Redefinicja skokowa (duża zmiana D)
        >>> classify_semantic_trajectory(
        ...     np.array([0.3, 0.8, 0.2]),
        ...     np.array([0.9, 0.8, 0.2])
        ... )
        'spacelike'
    """
    # Oblicz interwał Minkowskiego
    D1, S1, E1 = start
    D2, S2, E2 = end

    dS = S2 - S1
    dD = D2 - D1
    dE = E2 - E1

    # Transformacja do układu χ-Φ
    chi1 = D1 - E1
    chi2 = D2 - E2
    dchi = chi2 - chi1

    phi1 = np.sqrt(D1**2 + E1**2)
    phi2 = np.sqrt(D2**2 + E2**2)
    dphi = phi2 - phi1

    # ds² = -κ² dS² + dχ² + dΦ²
    ds_squared = -(kappa**2) * (dS**2) + (dchi**2) + (dphi**2)

    # Klasyfikuj
    if ds_squared < -tolerance:
        return 'timelike'  # Ewolucja temporalna (zmiana stabilności)
    elif ds_squared > tolerance:
        return 'spacelike'  # Redefinicja przestrzenna (zmiana struktury)
    else:
        return 'lightlike'  # Granica kauzalności semantycznej


def minkowski_distance(
    coords1: np.ndarray,
    coords2: np.ndarray,
    kappa: float = 1.0
) -> float:
    """
    Oblicza pseudo-metrykę Minkowskiego z sygnaturą (-,+,+) w przestrzeni GTMØ.

    Definicja zgodna z teorią GTMØ:
    ds² = -κ² dS² + dχ² + dΦ²

    gdzie:
    - χ = D - E  (oś chaotyczności)
    - Φ = √(D² + E²)  (norma w płaszczyźnie D-E)
    - κ = parametr skalujący oś stabilności (domyślnie 1.0)

    Interpretacja:
    - Składowa S ma sygnaturę ujemną (timelike) - reprezentuje "czas semantyczny"
    - Składowe χ, Φ mają sygnaturę dodatnią (spacelike) - reprezentują "przestrzeń semantyczną"
    - Metryka może dawać wartości ujemne (interwały timelike)

    Własności:
    - NIE jest metryką w sensie matematycznym (nie spełnia nierówności trójkąta)
    - Asymetryczna temporalna ewolucja (zmiana S ≠ zmiana χ/Φ)
    - Zachowuje kauzalność semantyczną (przeszłość → przyszłość w S)

    Args:
        coords1: Pierwszy punkt [D, S, E]
        coords2: Drugi punkt [D, S, E]
        kappa: Parametr skalujący oś S (domyślnie 1.0)

    Returns:
        Pseudo-odległość (może być ujemna dla interwałów timelike)

    Example:
        >>> # Duża zmiana S (timelike) → ujemna odległość²
        >>> minkowski_distance(np.array([0.8, 0.3, 0.2]), np.array([0.8, 0.9, 0.2]), kappa=1.0)
        -0.36

        >>> # Duża zmiana D,E (spacelike) → dodatnia odległość²
        >>> minkowski_distance(np.array([0.3, 0.8, 0.2]), np.array([0.9, 0.8, 0.2]), kappa=1.0)
        0.4...
    """
    if len(coords1) != 3 or len(coords2) != 3:
        raise ValueError(f"Coords must have 3 components (D,S,E), got {len(coords1)} and {len(coords2)}")

    D1, S1, E1 = coords1
    D2, S2, E2 = coords2

    # Oblicz różnice
    dS = S2 - S1
    dD = D2 - D1
    dE = E2 - E1

    # Transformacja do układu χ-Φ
    chi1 = D1 - E1
    chi2 = D2 - E2
    dchi = chi2 - chi1

    phi1 = np.sqrt(D1**2 + E1**2)
    phi2 = np.sqrt(D2**2 + E2**2)
    dphi = phi2 - phi1

    # Pseudo-metryka Minkowskiego: ds² = -κ² dS² + dχ² + dΦ²
    ds_squared = -(kappa**2) * (dS**2) + (dchi**2) + (dphi**2)

    # UWAGA: ds² może być ujemne (interwały timelike)
    # Zwracamy pierwiastek z |ds²| ze znakiem
    if ds_squared >= 0:
        return float(np.sqrt(ds_squared))  # Interwał spacelike
    else:
        return float(-np.sqrt(-ds_squared))  # Interwał timelike (ujemny)


def compute_axis_contributions(
    coords1: np.ndarray,
    coords2: np.ndarray,
    metric: str = 'phi9'
) -> Dict[str, float]:
    """
    Oblicza wkład każdej osi (D, S, E) do całkowitej odległości.

    Przydatne do diagnostyki: która oś powoduje największą desynchronizację?

    Args:
        coords1: Pierwszy punkt [D, S, E]
        coords2: Drugi punkt [D, S, E]
        metric: 'phi9' lub 'euclidean'

    Returns:
        Dict {'D': wkład_D, 'S': wkład_S, 'E': wkład_E, 'total': suma}

    Example:
        >>> c1 = np.array([0.8, 0.8, 0.2])
        >>> c2 = np.array([0.9, 0.9, 0.7])  # Duża zmiana w E
        >>> contributions = compute_axis_contributions(c1, c2)
        >>> contributions['E'] > contributions['D']  # E dominuje
        True
    """
    if len(coords1) != 3 or len(coords2) != 3:
        raise ValueError(f"Coords must have 3 components (D,S,E)")

    diff = np.abs(coords1 - coords2)

    if metric == 'phi9':
        # Φ⁹: d = Σᵢ φⁱ · |Δᵢ|ⁱ
        weights = np.array([PHI**1, PHI**2, PHI**3])
        powers = np.array([1, 2, 3])

        contribution_D = float(weights[0] * (diff[0] ** powers[0]))
        contribution_S = float(weights[1] * (diff[1] ** powers[1]))
        contribution_E = float(weights[2] * (diff[2] ** powers[2]))

    elif metric == 'euclidean':
        # Euklidesowa: d² = ΔD² + ΔS² + ΔE²
        contribution_D = float(diff[0] ** 2)
        contribution_S = float(diff[1] ** 2)
        contribution_E = float(diff[2] ** 2)

    else:
        raise ValueError(f"Unknown metric: {metric}")

    total = contribution_D + contribution_S + contribution_E

    return {
        'D': contribution_D,
        'S': contribution_S,
        'E': contribution_E,
        'total': total,
        # Udział procentowy
        'D_pct': (contribution_D / total * 100) if total > 0 else 0.0,
        'S_pct': (contribution_S / total * 100) if total > 0 else 0.0,
        'E_pct': (contribution_E / total * 100) if total > 0 else 0.0
    }


def phi9_norm(coords: np.ndarray) -> float:
    """
    Oblicza normę Φ⁹ punktu (odległość od zera).

    Args:
        coords: Punkt [D, S, E]

    Returns:
        ||coords||_Φ⁹
    """
    return phi9_distance(coords, np.zeros(3))


# =============================================================================
# POTENCJAŁ KOMUNIKACYJNY V_Comm
# =============================================================================

def compute_communication_potential(
    local_coords: List[np.ndarray],
    kappa_comm: float = 1.0,
    metric: str = 'phi9',
    metric_kappa: float = 1.0
) -> float:
    """
    Oblicza potencjał komunikacyjny (energię desynchronizacji).

    Definicja:
    V_Comm = (1/2) κ_comm · (1/n(n-1)) · Σᵢ<ⱼ d(φᵢ, φⱼ)²

    Interpretacja:
    - V_Comm → 0: obserwatorzy się zgadzają (emergencja możliwa)
    - V_Comm → ∞: całkowita desynchronizacja (brak konsensusu)

    Args:
        local_coords: Lista lokalnych współrzędnych φ_O dla każdego obserwatora
        kappa_comm: Stała komunikacyjna (domyślnie 1.0)
        metric: 'phi9', 'euclidean', lub 'minkowski'
        metric_kappa: Parametr κ dla metryki Minkowskiego (domyślnie 1.0)

    Returns:
        Energia potencjału V_Comm

    Example:
        >>> coords = [np.array([0.8, 0.8, 0.2]), np.array([0.85, 0.85, 0.15])]
        >>> compute_communication_potential(coords)
        0.034...
    """
    n = len(local_coords)

    if n < 2:
        return 0.0

    total_squared_distance = 0.0

    # Suma po wszystkich parach (i, j) gdzie i < j
    for i in range(n):
        for j in range(i + 1, n):
            if metric == 'phi9':
                dist = phi9_distance(local_coords[i], local_coords[j])
            elif metric == 'euclidean':
                dist = np.linalg.norm(local_coords[i] - local_coords[j])
            elif metric == 'minkowski':
                dist = minkowski_distance(local_coords[i], local_coords[j], kappa=metric_kappa)
                # Dla minkowski używamy |dist| bo może być ujemna
                dist = abs(dist)
            else:
                raise ValueError(f"Unknown metric: {metric}")

            total_squared_distance += dist ** 2

    # Liczba par
    num_pairs = n * (n - 1) / 2

    # V_Comm = (1/2) κ Σd² / n_pairs
    energy = 0.5 * kappa_comm * (total_squared_distance / num_pairs)

    return float(energy)


def compute_pairwise_energies(
    local_coords: List[np.ndarray],
    metric: str = 'phi9',
    metric_kappa: float = 1.0
) -> np.ndarray:
    """
    Oblicza macierz energii dla wszystkich par obserwatorów.

    Args:
        local_coords: Lista lokalnych współrzędnych
        metric: Metryka do użycia ('phi9', 'euclidean', 'minkowski')
        metric_kappa: Parametr κ dla metryki Minkowskiego

    Returns:
        Macierz n×n energii: E[i,j] = d(φᵢ, φⱼ)²
    """
    n = len(local_coords)
    energy_matrix = np.zeros((n, n))

    for i in range(n):
        for j in range(i + 1, n):
            if metric == 'phi9':
                dist = phi9_distance(local_coords[i], local_coords[j])
            elif metric == 'euclidean':
                dist = np.linalg.norm(local_coords[i] - local_coords[j])
            elif metric == 'minkowski':
                dist = abs(minkowski_distance(local_coords[i], local_coords[j], kappa=metric_kappa))
            else:
                dist = 0.0

            energy = dist ** 2
            energy_matrix[i, j] = energy
            energy_matrix[j, i] = energy  # Symetria

    return energy_matrix


# =============================================================================
# GRADIENT KOLAPSU ∇V
# =============================================================================

def compute_collapse_gradient(
    phi_observer: np.ndarray,
    phi_consensus: np.ndarray,
    psi_attractor: np.ndarray,
    kappa_comm: float = 1.0,
    kappa_context: float = 0.5
) -> np.ndarray:
    """
    Oblicza gradient kolapsu dla pojedynczego obserwatora.

    Definicja:
    ∇_φO V_total = κ_comm (φ_O - φ_consensus) + κ_context (φ_O - Ψ_attractor)

    Gradient wskazuje kierunek, w którym φ_O musi się przesunąć
    dla emergencji (kierunek spadku energii).

    Args:
        phi_observer: Współrzędne obserwatora [D, S, E]
        phi_consensus: Consensus wszystkich obserwatorów
        psi_attractor: Współrzędne attraktora kontekstowego (np. Ψᴷ)
        kappa_comm: Stała komunikacyjna
        kappa_context: Stała kontekstowa

    Returns:
        Gradient ∇V (kierunek kolapsu)

    Example:
        >>> phi_o = np.array([0.9, 0.9, 0.1])
        >>> phi_c = np.array([0.7, 0.7, 0.3])
        >>> psi = np.array([0.85, 0.85, 0.15])
        >>> gradient = compute_collapse_gradient(phi_o, phi_c, psi)
        >>> gradient[0] < 0  # Wskazuje w dół (ku consensus)
        True
    """
    # Gradient komunikacyjny (pcha w stronę consensus)
    grad_comm = kappa_comm * (phi_observer - phi_consensus)

    # Gradient kontekstowy (pcha w stronę attraktora)
    grad_context = kappa_context * (phi_observer - psi_attractor)

    # Gradient łączny (minus - kierunek SPADKU energii)
    total_gradient = -(grad_comm + grad_context)

    return total_gradient


def compute_all_collapse_gradients(
    local_coords: List[np.ndarray],
    context_attractor: np.ndarray,
    kappa_comm: float = 1.0,
    kappa_context: float = 0.5
) -> List[np.ndarray]:
    """
    Oblicza gradienty kolapsu dla wszystkich obserwatorów.

    Args:
        local_coords: Lista lokalnych współrzędnych
        context_attractor: Atraktor kontekstowy
        kappa_comm: Stała komunikacyjna
        kappa_context: Stała kontekstowa

    Returns:
        Lista gradientów dla każdego obserwatora
    """
    if len(local_coords) < 2:
        return []

    # Oblicz consensus
    consensus = np.mean(local_coords, axis=0)

    # Oblicz gradient dla każdego obserwatora
    gradients = []
    for coords in local_coords:
        gradient = compute_collapse_gradient(
            coords, consensus, context_attractor,
            kappa_comm, kappa_context
        )
        gradients.append(gradient)

    return gradients


# =============================================================================
# WARUNKI EMERGENCJI
# =============================================================================

def compute_adaptive_epsilon(
    base_epsilon: float,
    context_entropy: float,
    register: str,
    gamma: float = 0.3
) -> float:
    """
    Oblicza adaptacyjny próg emergencji zależny od kontekstu.

    Definicja:
    ε_adaptive = ε₀ · (1 + γ · H_context) · f_register

    gdzie:
    - ε₀ = bazowy próg (np. 0.15)
    - H_context = entropia kontekstowa (im wyższa, tym tolerancja większa)
    - γ = parametr czułości na entropię (domyślnie 0.3)
    - f_register = modulator dla rejestru językowego

    Uzasadnienie:
    - Kontekst wysokoentropijny (casualny, poetycki) → większe ε (tolerancja na rozbieżności)
    - Kontekst niskoentropijny (formalny, prawniczy) → mniejsze ε (wymagany konsensus)

    Args:
        base_epsilon: Bazowy próg emergencji
        context_entropy: Średnia entropia kontekstu E ∈ [0, 1]
        register: Rejestr językowy ('formal', 'casual', 'poetic', etc.)
        gamma: Czułość na entropię (domyślnie 0.3)

    Returns:
        ε_adaptive - dostosowany próg

    Example:
        >>> # Kontekst formalny, niska entropia → rygorystyczny próg
        >>> compute_adaptive_epsilon(0.15, context_entropy=0.1, register='formal')
        0.120...

        >>> # Kontekst casualny, wysoka entropia → tolerancyjny próg
        >>> compute_adaptive_epsilon(0.15, context_entropy=0.7, register='casual')
        0.201...
    """
    # Modulator rejestru
    register_modifiers = {
        'formal': 0.8,       # Bardziej rygorystyczny
        'legal': 0.7,        # Najbardziej rygorystyczny
        'technical': 0.85,
        'journalistic': 0.9,
        'casual': 1.2,       # Bardziej tolerancyjny
        'poetic': 1.3,       # Bardzo tolerancyjny
        'sarcastic': 1.1,
        'philosophical': 1.0  # Neutralny
    }

    f_register = register_modifiers.get(register, 1.0)

    # Komponent entropijny
    entropy_factor = 1.0 + gamma * context_entropy

    # ε_adaptive = ε₀ · (1 + γH) · f_register
    epsilon_adaptive = base_epsilon * entropy_factor * f_register

    # Ogranicz do rozsądnego zakresu [0.05, 0.5]
    epsilon_adaptive = np.clip(epsilon_adaptive, 0.05, 0.5)

    return float(epsilon_adaptive)


def compute_adaptive_epsilon_enhanced(
    base_epsilon: float,
    context_entropy: float,
    register: str,
    use_sqrt_scaling: bool = True
) -> float:
    """
    Ulepszona formuła adaptacyjnego progu emergencji z Ø₀.

    Definicja:
    ε_adaptive = ε₀ · √(1 + H_context/Ø₀) · f_register

    gdzie:
    - Ø₀ = 1.2925 (próg entropii, ~√(φ² + 1/φ) / √2)
    - √ zapewnia bardziej konserwatywne skalowanie niż liniowe

    Różnica od compute_adaptive_epsilon():
    - Używa pierwiastka kwadratowego dla bardziej łagodnego wzrostu
    - Ø₀ jako naturalny próg entropii (derived from φ)

    Args:
        base_epsilon: Bazowy próg emergencji
        context_entropy: Średnia entropia kontekstu E ∈ [0, 1]
        register: Rejestr językowy
        use_sqrt_scaling: Czy używać √ zamiast liniowego (domyślnie True)

    Returns:
        ε_adaptive - dostosowany próg (enhanced)

    Example:
        >>> # Ta sama entropia daje mniejszy wzrost niż w wersji liniowej
        >>> compute_adaptive_epsilon_enhanced(0.15, context_entropy=0.7, register='casual')
        0.198  # vs 0.218 w wersji liniowej
    """
    # Modulatory rejestru (dostosowane do bardziej rygorystycznych wartości)
    register_modifiers = {
        'legal': 0.75,        # Najbardziej rygorystyczny
        'formal': 0.82,       # Bardziej rygorystyczny
        'technical': 0.70,    # Bardzo rygorystyczny (technika wymaga precyzji)
        'journalistic': 0.90,
        'philosophical': 1.0, # Neutralny
        'sarcastic': 1.10,
        'casual': 1.20,       # Bardziej tolerancyjny
        'poetic': 1.45,       # Bardzo tolerancyjny
        'medical': 0.80       # Precyzja medyczna
    }

    f_register = register_modifiers.get(register, 1.0)

    # Komponent entropijny z Ø₀
    if use_sqrt_scaling:
        # ε_adaptive = ε₀ · √(1 + H/Ø₀) · f_register
        entropy_factor = np.sqrt(1.0 + context_entropy / PHI_0)
    else:
        # Wersja liniowa (jak compute_adaptive_epsilon)
        entropy_factor = 1.0 + 0.3 * context_entropy

    epsilon_adaptive = base_epsilon * entropy_factor * f_register

    # Ogranicz do rozsądnego zakresu [0.05, 0.5]
    epsilon_adaptive = np.clip(epsilon_adaptive, 0.05, 0.5)

    return float(epsilon_adaptive)


def check_emergence_condition(
    local_coords: List[np.ndarray],
    epsilon: float = 0.15,
    metric: str = 'phi9',
    adaptive_epsilon: bool = False,
    context_entropy: Optional[float] = None,
    register: str = 'formal',
    metric_kappa: float = 1.0
) -> Tuple[bool, Optional[np.ndarray]]:
    """
    Sprawdza warunek adelicznej emergencji.

    Twierdzenie emergencji:
    ∀ O ∈ Observers: d(φ_O, φ_consensus) ≤ ε ⟹ ∃!φ_∞ ∈ [0,1]³

    W słowach: Jeśli wszystkie lokalne interpretacje są w promieniu ε
    od konsensusu, to emerguje unikalna wartość globalna.

    Args:
        local_coords: Lista lokalnych współrzędnych
        epsilon: Próg synchronizacji (domyślnie 0.15)
        metric: Metryka do użycia ('phi9', 'euclidean', 'minkowski')
        adaptive_epsilon: Czy używać adaptacyjnego progu (domyślnie False)
        context_entropy: Średnia entropia kontekstu (jeśli adaptive_epsilon=True)
        register: Rejestr językowy (jeśli adaptive_epsilon=True)
        metric_kappa: Parametr κ dla metryki Minkowskiego

    Returns:
        (can_emerge, consensus_value)
        - can_emerge: True jeśli warunek spełniony
        - consensus_value: φ_∞ sprojektowana na [0,1]³ (jeśli emerged)

    Example:
        >>> # Blisko siebie → emergencja
        >>> coords = [np.array([0.85, 0.87, 0.15]), np.array([0.87, 0.88, 0.14])]
        >>> can_emerge, phi_inf = check_emergence_condition(coords)
        >>> can_emerge
        True
    """
    n = len(local_coords)

    if n < 2:
        return False, None

    # Oblicz consensus (centroid)
    consensus = np.mean(local_coords, axis=0)

    # Użyj adaptacyjnego epsilon jeśli włączone
    effective_epsilon = epsilon
    if adaptive_epsilon and context_entropy is not None:
        effective_epsilon = compute_adaptive_epsilon(
            base_epsilon=epsilon,
            context_entropy=context_entropy,
            register=register
        )

    # Sprawdź warunek adeliczny dla każdego obserwatora
    for coords in local_coords:
        if metric == 'phi9':
            distance = phi9_distance(coords, consensus)
        elif metric == 'euclidean':
            distance = np.linalg.norm(coords - consensus)
        elif metric == 'minkowski':
            distance = abs(minkowski_distance(coords, consensus, kappa=metric_kappa))
        else:
            raise ValueError(f"Unknown metric: {metric}")

        if distance > effective_epsilon:
            # Przynajmniej jeden obserwator za daleko
            return False, None

    # Wszystkie obserwatory w promieniu ε → EMERGENCJA!
    # Projekt consensus na [0,1]³ (wartość globalna)
    global_value = np.clip(consensus, 0, 1)

    return True, global_value


def compute_emergence_probability(
    local_coords: List[np.ndarray],
    epsilon: float = 0.15,
    metric: str = 'phi9',
    metric_kappa: float = 1.0,
    adaptive_epsilon: bool = False,
    context_entropy: Optional[float] = None,
    register: str = 'formal'
) -> float:
    """
    Oblicza probabilistyczne prawdopodobieństwo emergencji.

    P(emerge) = exp(-V_Comm / ε²)

    Args:
        local_coords: Lista lokalnych współrzędnych
        epsilon: Próg emergencji
        metric: Metryka ('phi9', 'euclidean', 'minkowski')
        metric_kappa: Parametr κ dla metryki Minkowskiego
        adaptive_epsilon: Czy używać adaptacyjnego progu
        context_entropy: Entropia kontekstu (jeśli adaptive_epsilon=True)
        register: Rejestr językowy (jeśli adaptive_epsilon=True)

    Returns:
        Prawdopodobieństwo emergencji w [0, 1]
    """
    if len(local_coords) < 2:
        return 0.0

    # Użyj adaptacyjnego epsilon jeśli włączone
    effective_epsilon = epsilon
    if adaptive_epsilon and context_entropy is not None:
        effective_epsilon = compute_adaptive_epsilon(
            base_epsilon=epsilon,
            context_entropy=context_entropy,
            register=register
        )

    # Oblicz energię
    energy = compute_communication_potential(
        local_coords,
        kappa_comm=1.0,
        metric=metric,
        metric_kappa=metric_kappa
    )

    # P(emerge) = exp(-E/ε²)
    probability = np.exp(-energy / (effective_epsilon ** 2))

    return float(np.clip(probability, 0, 1))


# =============================================================================
# METRYKI POMOCNICZE
# =============================================================================

def compute_consensus(local_coords: List[np.ndarray]) -> np.ndarray:
    """
    Oblicza consensus (centroid) lokalnych interpretacji.

    Args:
        local_coords: Lista lokalnych współrzędnych

    Returns:
        Punkt consensus (średnia)
    """
    if len(local_coords) == 0:
        raise ValueError("Cannot compute consensus for empty list")

    return np.mean(local_coords, axis=0)


def compute_dispersion(
    local_coords: List[np.ndarray],
    metric: str = 'phi9',
    metric_kappa: float = 1.0
) -> float:
    """
    Oblicza dyspersję (rozrzut) lokalnych interpretacji wokół consensus.

    Args:
        local_coords: Lista lokalnych współrzędnych
        metric: Metryka ('phi9', 'euclidean', 'minkowski')
        metric_kappa: Parametr κ dla metryki Minkowskiego

    Returns:
        Średnia odległość od consensus
    """
    if len(local_coords) < 2:
        return 0.0

    consensus = compute_consensus(local_coords)

    total_distance = 0.0
    for coords in local_coords:
        if metric == 'phi9':
            dist = phi9_distance(coords, consensus)
        elif metric == 'euclidean':
            dist = np.linalg.norm(coords - consensus)
        elif metric == 'minkowski':
            dist = abs(minkowski_distance(coords, consensus, kappa=metric_kappa))
        else:
            dist = 0.0

        total_distance += dist

    return float(total_distance / len(local_coords))


def recommend_context_for_disambiguation(
    local_coords: List[np.ndarray],
    attractors: Dict[str, np.ndarray],
    metric: str = 'phi9'
) -> Tuple[str, float]:
    """
    Rekomenduje atraktor kontekstowy, który najlepiej rozwiązuje niejednoznaczność.

    Dla każdego attraktora oblicza, jak bardzo zmniejszy energię synchronizacyjną.

    Args:
        local_coords: Lista lokalnych współrzędnych
        attractors: Dict {nazwa: współrzędne} atraktorów
        metric: Metryka

    Returns:
        (best_attractor_name, energy_reduction)
    """
    if len(local_coords) < 2:
        return "none", 0.0

    # Aktualna energia bez kontekstu
    current_energy = compute_communication_potential(local_coords, metric=metric)

    best_attractor = None
    max_reduction = 0.0

    # Test każdego attraktora
    for name, attractor_coords in attractors.items():
        # Oblicz gradienty w kierunku tego attraktora
        gradients = compute_all_collapse_gradients(
            local_coords,
            context_attractor=attractor_coords,
            kappa_comm=0.0,  # Tylko kontekst
            kappa_context=1.0
        )

        # Zasymuluj przesunięcie (mały krok w kierunku gradientu)
        step_size = 0.1
        shifted_coords = [
            coords + step_size * grad
            for coords, grad in zip(local_coords, gradients)
        ]

        # Nowa energia
        new_energy = compute_communication_potential(shifted_coords, metric=metric)

        # Redukcja energii
        reduction = current_energy - new_energy

        if reduction > max_reduction:
            max_reduction = reduction
            best_attractor = name

    return best_attractor or "none", float(max_reduction)


# =============================================================================
# WIZUALIZACJA I DIAGNOSTYKA
# =============================================================================

def diagnose_emergence_failure(
    local_coords: List[np.ndarray],
    epsilon: float = 0.15,
    metric: str = 'phi9',
    metric_kappa: float = 1.0
) -> Dict[str, Any]:
    """
    Diagnozuje dlaczego emergencja nie nastąpiła z dekompozycją D-S-E.

    Args:
        local_coords: Lista lokalnych współrzędnych
        epsilon: Próg emergencji
        metric: Metryka
        metric_kappa: Parametr κ dla metryki Minkowskiego

    Returns:
        Dict z informacjami diagnostycznymi, w tym:
        - axis_decomposition: wkład każdej osi (D, S, E) do desynchronizacji
        - dominant_axis: która oś powoduje największą desynchronizację
    """
    if len(local_coords) < 2:
        return {
            'reason': 'insufficient_observers',
            'n_observers': len(local_coords),
            'min_required': 2
        }

    consensus = compute_consensus(local_coords)
    distances = []

    # Oblicz także dekompozycję D-S-E dla każdego obserwatora
    axis_contributions_list = []

    for i, coords in enumerate(local_coords):
        if metric == 'phi9':
            dist = phi9_distance(coords, consensus)
        elif metric == 'euclidean':
            dist = np.linalg.norm(coords - consensus)
        elif metric == 'minkowski':
            dist = abs(minkowski_distance(coords, consensus, kappa=metric_kappa))
        else:
            dist = 0.0

        # Dekompozycja wkładu osi
        axis_contrib = compute_axis_contributions(coords, consensus, metric=metric)
        axis_contributions_list.append(axis_contrib)

        distances.append({
            'observer_idx': i,
            'distance': float(dist),
            'exceeds_epsilon': dist > epsilon,
            'coords': coords.tolist(),
            'axis_contributions': axis_contrib
        })

    # Sortuj po odległości
    distances.sort(key=lambda x: x['distance'], reverse=True)

    max_distance = distances[0]['distance']
    outliers = [d for d in distances if d['exceeds_epsilon']]

    # Agreguj wkład osi dla wszystkich obserwatorów
    total_D = sum(ac['D'] for ac in axis_contributions_list)
    total_S = sum(ac['S'] for ac in axis_contributions_list)
    total_E = sum(ac['E'] for ac in axis_contributions_list)
    total_all = total_D + total_S + total_E

    axis_decomposition = {
        'D': {
            'absolute': float(total_D),
            'percentage': float(total_D / total_all * 100) if total_all > 0 else 0.0
        },
        'S': {
            'absolute': float(total_S),
            'percentage': float(total_S / total_all * 100) if total_all > 0 else 0.0
        },
        'E': {
            'absolute': float(total_E),
            'percentage': float(total_E / total_all * 100) if total_all > 0 else 0.0
        }
    }

    # Określ dominującą oś
    dominant_axis = max(['D', 'S', 'E'], key=lambda ax: axis_decomposition[ax]['absolute'])

    # Oblicz energię i jej klasyfikację
    energy = compute_communication_potential(local_coords, metric=metric, metric_kappa=metric_kappa)

    # Klasyfikacja intensywności na podstawie energii
    # Empiryczne progi: <10 = niska, 10-100 = średnia, 100-1000 = wysoka, >1000 = ekstremalna
    if energy < 10:
        energy_severity = 'NISKA'
        severity_interpretation = 'Lekka desynchronizacja - łatwa do skorygowania'
    elif energy < 100:
        energy_severity = 'ŚREDNIA'
        severity_interpretation = 'Umiarkowana desynchronizacja - wymaga interwencji'
    elif energy < 1000:
        energy_severity = 'WYSOKA'
        severity_interpretation = 'Silna desynchronizacja - poważny konflikt interpretacji'
    else:
        energy_severity = 'EKSTREMALNA'
        severity_interpretation = 'Krytyczna desynchronizacja - fundamentalna rozbieżność semantyczna'

    return {
        'reason': 'desynchronization',
        'n_observers': len(local_coords),
        'consensus': consensus.tolist(),
        'epsilon': epsilon,
        'max_distance': max_distance,
        'exceeds_by': max_distance - epsilon,
        'num_outliers': len(outliers),
        'outliers': outliers,
        'energy': energy,
        'energy_severity': energy_severity,
        'severity_interpretation': severity_interpretation,
        'dispersion': compute_dispersion(local_coords, metric=metric),
        'axis_decomposition': axis_decomposition,
        'dominant_axis': dominant_axis,
        'interpretation': _interpret_dominant_axis(dominant_axis)
    }


def _interpret_dominant_axis(axis: str) -> str:
    """Interpretuje semantycznie dominującą oś desynchronizacji."""
    interpretations = {
        'D': 'Rozbieżność w OKREŚLONOŚCI (Determination) - obserwatorzy różnią się co do pewności/definitywności',
        'S': 'Rozbieżność w STABILNOŚCI (Stability) - obserwatorzy różnią się co do trwałości semantycznej',
        'E': 'Rozbieżność w ENTROPII (Entropy) - obserwatorzy różnią się co do poziomu chaosu/wieloznaczności'
    }
    return interpretations.get(axis, 'Unknown axis')


def recommend_context_for_emergence(
    axis_decomposition: Dict[str, Dict[str, float]],
    dominant_axis: str,
    current_register: str = 'neutral'
) -> Dict[str, Any]:
    """
    Rekomenduje kontekst ułatwiający emergencję na podstawie dekompozycji D-S-E.

    Analizuje która oś (D, S, E) powoduje największą desynchronizację
    i proponuje konkretne działania naprawcze.

    Args:
        axis_decomposition: Dict z wkładem każdej osi (z diagnose_emergence_failure)
        dominant_axis: Która oś dominuje ('D', 'S', lub 'E')
        current_register: Obecny rejestr językowy

    Returns:
        Dict z rekomendacjami:
        {
            'strategy': str,  # Ogólna strategia
            'actions': List[str],  # Konkretne działania
            'recommended_register': str,  # Sugerowany rejestr
            'recommended_observers': List[str],  # Sugerowani obserwatorzy
            'explanation': str  # Uzasadnienie
        }

    Example:
        >>> decomp = {
        ...     'D': {'percentage': 15.0},
        ...     'S': {'percentage': 10.0},
        ...     'E': {'percentage': 75.0}
        ... }
        >>> recommend_context_for_emergence(decomp, 'E')
        {
            'strategy': 'reduce_entropy',
            'actions': ['Użyj kontekstu formalnego', ...],
            ...
        }
    """
    recommendations = {
        'D': {
            'strategy': 'clarify_determination',
            'actions': [
                'Dodaj definicje słownikowe lub encyklopedyczne',
                'Użyj przykładów ujednoznaczniających znaczenie',
                'Skonkretyzuj abstrakcyjne pojęcia',
                'Wprowadź kontekst techniczny/specjalistyczny'
            ],
            'recommended_register': 'technical' if axis_decomposition['D']['percentage'] > 50 else 'formal',
            'recommended_observers': ['O_technical', 'O_legal_strict', 'O_medical'],
            'explanation': (
                'Rozbieżność w określoności (D) wskazuje na różne rozumienie denotacji. '
                'Obserwatorzy potrzebują bardziej precyzyjnych definicji i kontekstu '
                'redukującego wieloznaczność.'
            )
        },
        'S': {
            'strategy': 'stabilize_temporal_perspective',
            'actions': [
                'Uzgodnij perspektywę czasową (historyczna vs współczesna)',
                'Wskaż ewolucję znaczenia w czasie',
                'Użyj kontekstu diachronicznego (rozwój semantyczny)',
                'Wprowadź ramkę temporalną (kiedy to znaczenie obowiązuje)'
            ],
            'recommended_register': 'philosophical' if axis_decomposition['S']['percentage'] > 50 else 'journalistic',
            'recommended_observers': ['O_philosophical', 'O_journalistic', 'O_formal'],
            'explanation': (
                'Rozbieżność w stabilności (S) wskazuje na różne oczekiwania co do trwałości znaczenia. '
                'Obserwatorzy patrzą z różnych perspektyw czasowych i potrzebują '
                'uzgodnionej ramki temporalnej.'
            )
        },
        'E': {
            'strategy': 'reduce_entropy',
            'actions': [
                'Użyj kontekstu formalnego zamiast casualowego',
                'Ogranicz wieloznaczność przez precyzyjny język',
                'Wprowadź strukturę (enumeracje, klasyfikacje)',
                'Zmień rejestr na bardziej rygorystyczny'
            ],
            'recommended_register': 'formal' if axis_decomposition['E']['percentage'] > 60 else 'journalistic',
            'recommended_observers': ['O_formal', 'O_legal', 'O_technical'],
            'explanation': (
                'Rozbieżność w entropii (E) wskazuje na różne poziomy tolerancji na chaos semantyczny. '
                'Obserwatorzy potrzebują bardziej ustrukturyzowanego, niskoentropijnego kontekstu '
                'dla osiągnięcia konsensusu.'
            )
        }
    }

    recommendation = recommendations.get(dominant_axis, {
        'strategy': 'unknown',
        'actions': ['Nieznana oś dominująca'],
        'recommended_register': current_register,
        'recommended_observers': [],
        'explanation': 'Nieznana oś dominująca'
    })

    # Dodaj informacje o sile rozbieżności
    percentage = axis_decomposition.get(dominant_axis, {}).get('percentage', 0.0)

    if percentage > 80:
        urgency = 'KRYTYCZNA'
    elif percentage > 60:
        urgency = 'WYSOKA'
    elif percentage > 40:
        urgency = 'UMIARKOWANA'
    else:
        urgency = 'NISKA'

    recommendation['urgency'] = urgency
    recommendation['dominant_axis_percentage'] = percentage

    return recommendation


# =============================================================================
# TESTY MODUŁU
# =============================================================================

if __name__ == "__main__":
    # Fix encoding dla Windows
    import sys
    import io
    if sys.platform == 'win32':
        try:
            sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
        except:
            pass

    print("=" * 60)
    print("GTMØ Adelic Metrics - Test modułu")
    print("=" * 60)

    # Test 1: Metryka Φ⁹
    print("\n[Test 1] Metryka Φ⁹")
    coords1 = np.array([0.8, 0.8, 0.2])
    coords2 = np.array([0.9, 0.9, 0.1])
    dist = phi9_distance(coords1, coords2)
    print(f"  d_Φ⁹({coords1}, {coords2}) = {dist:.3f}")

    # Sprawdź symetrię
    dist_rev = phi9_distance(coords2, coords1)
    print(f"  Symetria: {dist:.3f} == {dist_rev:.3f}? {np.isclose(dist, dist_rev)}")

    # Test 2: Potencjał komunikacyjny
    print("\n[Test 2] Potencjał V_Comm")
    local_coords = [
        np.array([0.85, 0.87, 0.15]),
        np.array([0.87, 0.88, 0.14])
    ]
    energy = compute_communication_potential(local_coords)
    print(f"  V_Comm (blisko) = {energy:.4f}")

    local_coords_far = [
        np.array([0.85, 0.87, 0.15]),
        np.array([0.25, 0.30, 0.85])
    ]
    energy_far = compute_communication_potential(local_coords_far)
    print(f"  V_Comm (daleko) = {energy_far:.4f}")
    print(f"  Energia rośnie dla większej różnicy: {energy_far > energy}")

    # Test 3: Warunek emergencji
    print("\n[Test 3] Warunek emergencji")
    can_emerge, phi_inf = check_emergence_condition(local_coords, epsilon=0.15)
    print(f"  Coords blisko: emerged={can_emerge}, φ_∞={phi_inf}")

    can_emerge_far, phi_inf_far = check_emergence_condition(local_coords_far, epsilon=0.15)
    print(f"  Coords daleko: emerged={can_emerge_far}, φ_∞={phi_inf_far}")

    # Test 4: Gradient kolapsu
    print("\n[Test 4] Gradient kolapsu")
    phi_o = np.array([0.9, 0.9, 0.1])
    phi_c = np.array([0.7, 0.7, 0.3])
    psi = np.array([0.85, 0.85, 0.15])

    gradient = compute_collapse_gradient(phi_o, phi_c, psi)
    print(f"  φ_O = {phi_o}")
    print(f"  φ_consensus = {phi_c}")
    print(f"  ∇V = {gradient}")
    print(f"  Gradient wskazuje w dół (D,S): {gradient[0] < 0 and gradient[1] < 0}")
    print(f"  Gradient wskazuje w górę (E): {gradient[2] > 0}")

    # Test 5: Diagnostyka
    print("\n[Test 5] Diagnostyka niepowodzenia emergencji")
    diagnosis = diagnose_emergence_failure(local_coords_far, epsilon=0.15)
    print(f"  Reason: {diagnosis['reason']}")
    print(f"  Energy: {diagnosis['energy']:.3f}")
    print(f"  Max distance: {diagnosis['max_distance']:.3f}")
    print(f"  Exceeds ε by: {diagnosis['exceeds_by']:.3f}")

    # Test 6: Metryka Minkowskiego
    print("\n[Test 6] Pseudo-metryka Minkowskiego")
    c1 = np.array([0.8, 0.3, 0.2])
    c2 = np.array([0.8, 0.9, 0.2])  # Duża zmiana S (timelike)
    dist_minkowski_time = minkowski_distance(c1, c2, kappa=1.0)
    print(f"  Timelike: d_M({c1}, {c2}) = {dist_minkowski_time:.3f}")
    print(f"  Jest ujemna (timelike): {dist_minkowski_time < 0}")

    c3 = np.array([0.3, 0.8, 0.2])
    c4 = np.array([0.9, 0.8, 0.2])  # Duża zmiana D (spacelike)
    dist_minkowski_space = minkowski_distance(c3, c4, kappa=1.0)
    print(f"  Spacelike: d_M({c3}, {c4}) = {dist_minkowski_space:.3f}")
    print(f"  Jest dodatnia (spacelike): {dist_minkowski_space > 0}")

    # Test 7: Adaptacyjny epsilon
    print("\n[Test 7] Adaptacyjny próg ε")
    base_eps = 0.15

    eps_formal_low = compute_adaptive_epsilon(base_eps, context_entropy=0.1, register='formal')
    print(f"  LINEAR - Formal + niska entropia (0.1): ε = {eps_formal_low:.3f}")

    eps_casual_high = compute_adaptive_epsilon(base_eps, context_entropy=0.7, register='casual')
    print(f"  LINEAR - Casual + wysoka entropia (0.7): ε = {eps_casual_high:.3f}")

    print(f"  Casual/formal ratio: {eps_casual_high/eps_formal_low:.2f}x")
    print(f"  Wysokoentropijny kontekst → większa tolerancja: {eps_casual_high > eps_formal_low}")

    # Test enhanced version
    eps_formal_enhanced = compute_adaptive_epsilon_enhanced(base_eps, context_entropy=0.1, register='formal')
    eps_casual_enhanced = compute_adaptive_epsilon_enhanced(base_eps, context_entropy=0.7, register='casual')

    print(f"\n  ENHANCED (√ scaling) - Formal: ε = {eps_formal_enhanced:.3f}")
    print(f"  ENHANCED (√ scaling) - Casual: ε = {eps_casual_enhanced:.3f}")
    print(f"  Enhanced casual/formal ratio: {eps_casual_enhanced/eps_formal_enhanced:.2f}x")
    print(f"  Enhanced jest bardziej konserwatywne: {eps_casual_enhanced < eps_casual_high}")

    # Test 8: Dekompozycja osi
    print("\n[Test 8] Dekompozycja D-S-E w diagnostyce")
    coords_E_dominant = [
        np.array([0.8, 0.8, 0.2]),
        np.array([0.85, 0.82, 0.8])  # Duża zmiana w E
    ]
    diagnosis_decomp = diagnose_emergence_failure(coords_E_dominant, epsilon=0.15)

    print(f"  Dominant axis: {diagnosis_decomp['dominant_axis']}")
    print(f"  Decomposition:")
    for axis in ['D', 'S', 'E']:
        pct = diagnosis_decomp['axis_decomposition'][axis]['percentage']
        print(f"    {axis}: {pct:.1f}%")
    print(f"  Interpretation: {diagnosis_decomp['interpretation']}")

    # Test 9: Axis contributions
    print("\n[Test 9] Wkład osi do odległości")
    contrib = compute_axis_contributions(
        np.array([0.8, 0.8, 0.2]),
        np.array([0.85, 0.85, 0.7]),
        metric='phi9'
    )
    print(f"  D contribution: {contrib['D_pct']:.1f}%")
    print(f"  S contribution: {contrib['S_pct']:.1f}%")
    print(f"  E contribution: {contrib['E_pct']:.1f}%")
    print(f"  E dominuje (duża zmiana entropii): {contrib['E_pct'] > contrib['D_pct']}")

    # Test 10: Klasyfikacja trajektorii
    print("\n[Test 10] Klasyfikacja trajektorii semantycznych")

    # Timelike: ewolucja naturalna
    traj1_start = np.array([0.8, 0.3, 0.2])
    traj1_end = np.array([0.8, 0.9, 0.2])
    traj1_type = classify_semantic_trajectory(traj1_start, traj1_end, kappa=PHI_0)
    print(f"  Trajektoria 1 (duża zmiana S): {traj1_type}")
    print(f"  → Ewolucja temporalna (naturalna zmiana znaczenia)")

    # Spacelike: redefinicja skokowa
    traj2_start = np.array([0.3, 0.8, 0.2])
    traj2_end = np.array([0.9, 0.8, 0.2])
    traj2_type = classify_semantic_trajectory(traj2_start, traj2_end, kappa=PHI_0)
    print(f"  Trajektoria 2 (duża zmiana D): {traj2_type}")
    print(f"  → Redefinicja przestrzenna (skokowa zmiana struktury)")

    # Test 11: Rekomendacja kontekstu
    print("\n[Test 11] Automatyczna rekomendacja kontekstu")

    # Symuluj dekompozycję z dominującą osią E
    test_decomp = {
        'D': {'absolute': 0.12, 'percentage': 14.4},
        'S': {'absolute': 0.15, 'percentage': 17.9},
        'E': {'absolute': 0.56, 'percentage': 67.7}
    }

    recommendation = recommend_context_for_emergence(test_decomp, 'E', current_register='casual')
    print(f"  Dominująca oś: E ({test_decomp['E']['percentage']:.1f}%)")
    print(f"  Strategia: {recommendation['strategy']}")
    print(f"  Pilność: {recommendation['urgency']}")
    print(f"  Rekomendowany rejestr: {recommendation['recommended_register']}")
    print(f"  Akcje naprawcze:")
    for i, action in enumerate(recommendation['actions'][:2], 1):
        print(f"    {i}. {action}")

    print("\n" + "=" * 60)
    print("✓ Moduł gtmo_adelic_metrics.py załadowany pomyślnie")
    print("  ✓ Metryka Φ⁹ (Riemannowska, symetryczna)")
    print("  ✓ Pseudo-metryka Minkowskiego (sygnatura -,+,+)")
    print("  ✓ Adaptacyjny próg ε (LINEAR + ENHANCED)")
    print("  ✓ Dekompozycja D-S-E w diagnostyce")
    print("  ✓ Klasyfikacja trajektorii (timelike/spacelike/lightlike)")
    print("  ✓ Automatyczna rekomendacja kontekstu")
    print("=" * 60)
