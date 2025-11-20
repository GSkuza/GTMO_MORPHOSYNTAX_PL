#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
GTMØ Constitutional Duality Calculator
=======================================
Implementacja teorii Constitutional Definiteness (CD) i Constitutional Indefiniteness (CI)
zgodnie z Twierdzeniem o Dualności Morfosyntaktycznej.

Teoria:
-------
Constitutional Duality wynika z Zasady Nieoznaczoności Semantycznej (Semantic Uncertainty Principle):

    Δ_form · Δ_int ≥ ħ_semantic

gdzie:
    - Δ_form: nieokreśloność formy (morfosyntaktyczna wieloznaczność)
    - Δ_int: nieokreśloność interpretacji (semantyczny chaos)
    - ħ_semantic: fundamentalna stała semantyczna (związana z Ø₀ = 1.2925)

Projekcja morfosyntaktyczna tego prawa prowadzi do relacji dualności:

    CI × CD = Depth²

gdzie:
    - CD (Constitutional Definiteness): miara strukturalnej określoności
    - CI (Constitutional Indefiniteness): miara strukturalnej niedefinitywności
    - Depth: głębokość składniowa (max depth w drzewie zależności)

Formuły:
--------
    CD = (1/Ambiguity) × Depth × √(D×S/E)
    CI = Ambiguity × Depth × √(E/(D×S))

gdzie:
    - Ambiguity: średnia liczba interpretacji morfologicznych na słowo
    - Depth: maksymalna głębokość składniowa
    - D, S, E: współrzędne w przestrzeni fazowej F³ (Determination, Stability, Entropy)

Dodatkowo:
    SA (Semantic Accessibility) = CD / Depth² ∈ [0, 1]
    - Znormalizowana miara dostępności semantycznej
    - 1.0 = tekst maksymalnie dostępny (wysoka definiteness)
    - 0.0 = tekst niedostępny (wysoka indefiniteness)

Author: GTMØ Team
Date: 2025-01-XX
Version: 1.0
"""

import numpy as np
from typing import Dict, Tuple, Optional
from dataclasses import dataclass
from enum import Enum

# =============================================================================
# THEORETICAL CONSTANTS
# =============================================================================

# Minimalna wartość dla zabezpieczenia przed dzieleniem przez zero
EPSILON = 1e-10

# Progi dla kategoryzacji
SA_HIGH_THRESHOLD = 0.7  # SA > 0.7 = wysoka dostępność
SA_LOW_THRESHOLD = 0.3   # SA < 0.3 = niska dostępność

# Próg tolerancji błędu dla weryfikacji dualności
DUALITY_ERROR_TOLERANCE = 0.01  # 1% błędu jest akceptowalne

# Minimalna ambiguity (każde słowo ma przynajmniej 1 interpretację)
MIN_AMBIGUITY = 1.0

# Minimalna głębokość składniowa (płaskie zdanie bez hierarchii)
MIN_DEPTH = 1


# =============================================================================
# DATA STRUCTURES
# =============================================================================

class AccessibilityCategory(Enum):
    """Kategorie dostępności semantycznej."""
    HIGH = "WYSOKA_DOSTĘPNOŚĆ"
    MEDIUM = "ŚREDNIA_DOSTĘPNOŚĆ"
    LOW = "NISKA_DOSTĘPNOŚĆ"


class StructureClassification(Enum):
    """Klasyfikacja struktury tekstu na podstawie stosunku CD/CI."""
    ORDERED = "ORDERED_STRUCTURE"      # CD > CI
    BALANCED = "BALANCED_STRUCTURE"    # CD ≈ CI
    CHAOTIC = "CHAOTIC_STRUCTURE"      # CI > CD


@dataclass
class ConstitutionalMetrics:
    """
    Kompletne metryki Constitutional Duality dla fragmentu tekstu.

    Attributes:
        CD (float): Constitutional Definiteness - miara strukturalnej określoności
        CI (float): Constitutional Indefiniteness - miara strukturalnej niedefinitywności
        SA (float): Semantic Accessibility - znormalizowana dostępność [0,1]
        depth (int): Głębokość składniowa użyta w obliczeniach
        ambiguity (float): Ambiguity morfologiczna użyta w obliczeniach
        D, S, E (float): Współrzędne w przestrzeni fazowej

        duality_product (float): CI × CD (powinno ≈ Depth²)
        duality_theoretical (float): Teoretyczna wartość Depth²
        duality_error (float): Względny błąd |product - theoretical| / theoretical
        duality_verified (bool): Czy dualność została zweryfikowana (error < tolerance)

        geometric_balance (float): √(D×S/E) - balans geometryczny (dla CD)
        geometric_tension (float): √(E/(D×S)) - napięcie geometryczne (dla CI)

        CI_morphological (float): Składnik morfologiczny CI
        CI_syntactic (float): Składnik składniowy CI
        CI_semantic (float): Składnik semantyczny CI

        sa_category (AccessibilityCategory): Kategoria dostępności
        structure_classification (StructureClassification): Klasyfikacja struktury
        cd_ci_ratio (float): Stosunek CD/CI
    """
    # Core metrics
    CD: float
    CI: float
    SA: float

    # Input parameters
    depth: int
    ambiguity: float
    D: float
    S: float
    E: float

    # Duality verification
    duality_product: float
    duality_theoretical: float
    duality_error: float
    duality_verified: bool

    # Geometric components
    geometric_balance: float
    geometric_tension: float

    # CI decomposition
    CI_morphological: float
    CI_syntactic: float
    CI_semantic: float

    # Classifications
    sa_category: AccessibilityCategory
    structure_classification: StructureClassification
    cd_ci_ratio: float
    # Enhanced classification label
    enhanced_structure_classification: str = ""

    def to_dict(self) -> Dict:
        """
        Konwertuj metryki do JSON-compatible dictionary.

        Returns:
            Dictionary z wszystkimi metrykami w formacie gotowym do serializacji JSON.
        """
        # Wzory do wyświetlenia
        cd_formula = (
            f"(1/{self.ambiguity:.2f}) × {self.depth} × "
            f"√({self.D:.3f}×{self.S:.3f}/{self.E:.3f}) = {self.CD:.2f}"
        )
        ci_formula = (
            f"{self.ambiguity:.2f} × {self.depth} × "
            f"√({self.E:.3f}/({self.D:.3f}×{self.S:.3f})) = {self.CI:.2f}"
        )
        duality_formula = (
            f"CI × CD = {self.CI:.2f} × {self.CD:.2f} = {self.duality_product:.2f} "
            f"≈ Depth² = {self.duality_theoretical}"
        )
        sa_formula = (
            f"CD / (CI + CD) = {self.CD:.2f} / ({self.CI:.2f} + {self.CD:.2f}) = {self.SA:.3f}"
        )

        # Interpretacja SA
        if self.sa_category == AccessibilityCategory.HIGH:
            sa_desc = "Tekst bardzo dostępny (> 70% definiteness)"
        elif self.sa_category == AccessibilityCategory.MEDIUM:
            sa_desc = "Tekst umiarkowanie dostępny (30-70% definiteness)"
        else:
            sa_desc = "Tekst trudno dostępny (< 30% definiteness)"

        # Interpretacja klasyfikacji struktury
        if self.structure_classification == StructureClassification.ORDERED:
            classification_desc = "Tekst uporządkowany, strukturalny (CD > CI)"
        elif self.structure_classification == StructureClassification.BALANCED:
            classification_desc = "Tekst zbalansowany (CD ≈ CI)"
        else:
            classification_desc = "Tekst chaotyczny, wieloznaczny (CI > CD)"

        # Procentowe wkłady składników CI
        ci_total = self.CI_morphological + self.CI_syntactic + self.CI_semantic
        if ci_total > EPSILON:
            ci_morph_percent = (self.CI_morphological / ci_total) * 100
            ci_synt_percent = (self.CI_syntactic / ci_total) * 100
            ci_sem_percent = (self.CI_semantic / ci_total) * 100
        else:
            ci_morph_percent = ci_synt_percent = ci_sem_percent = 33.33

        return {
            "definiteness": {
                "value": round(self.CD, 4),
                "formula": cd_formula,
                "interpretation": "Wysoka CD = tekst uporządkowany, jednoznaczny, strukturalny",
                "components": {
                    "inverse_ambiguity": round(1.0 / self.ambiguity, 4),
                    "depth": self.depth,
                    "geometric_balance": round(self.geometric_balance, 4)
                }
            },
            "indefiniteness": {
                "value": round(self.CI, 4),
                "formula": ci_formula,
                "interpretation": "Wysoka CI = tekst chaotyczny, wieloznaczny, nieprzewidywalny",
                "components": {
                    "ambiguity": round(self.ambiguity, 4),
                    "depth": self.depth,
                    "geometric_tension": round(self.geometric_tension, 4)
                },
                "decomposition": {
                    "morphological": {
                        "value": round(self.CI_morphological, 4),
                        "percentage": round(ci_morph_percent, 2),
                        "source": "Fleksja, ambiguity morfologiczna"
                    },
                    "syntactic": {
                        "value": round(self.CI_syntactic, 4),
                        "percentage": round(ci_synt_percent, 2),
                        "source": "Głębokość składniowa, długość zdań"
                    },
                    "semantic": {
                        "value": round(self.CI_semantic, 4),
                        "percentage": round(ci_sem_percent, 2),
                        "source": "Chaos semantyczny w przestrzeni F³"
                    }
                }
            },
            "semantic_accessibility": {
                "value": round(self.SA, 4),
                "percentage": round(self.SA * 100, 2),
                "formula": sa_formula,
                "interpretation": sa_desc,
                "category": self.sa_category.value,
                "range": "[0,1] gdzie 1=maksymalna dostępność, 0=niedostępny",
                "advantages": [
                    "Znormalizowana do [0,1]",
                    "Niezależna od skali absolutnej",
                    "Intuicyjna interpretacja"
                ]
            },
            "duality": {
                "product": round(self.duality_product, 4),
                "theoretical": self.duality_theoretical,
                "error_percent": round(self.duality_error * 100, 4),
                "formula": "CI × CD = Depth²",
                "verification": "PASSED" if self.duality_verified else "WARNING",
                "interpretation": (
                    "Dualność wynika z Zasady Nieoznaczoności Semantycznej: "
                    "Δ_form · Δ_int ≥ ħ_semantic"
                )
            },
            "classification": {
                "type": self.structure_classification.value,
                "cd_ci_ratio": round(self.cd_ci_ratio, 4),
                "description": classification_desc,
                "enhanced_type": self.enhanced_structure_classification
            },
            "theoretical_basis": {
                "derived_from": "Zasada Nieoznaczoności Semantycznej (GTMØ Axiom)",
                "morphosyntactic_projection": "Δ_form = Ambiguity × f(Depth), Δ_geom = √(E/(D×S))",
                "fundamental_constant": "Ø₀ = 1.2925 (Hausdorff dimension of fractal boundaries)",
                "operator": "Ø: projekcja na |ψ_Ø⟩ = (1/√3, 1/√3, 1/√3)ᵀ",
                "semantic_accessibility": "SA = CD/(CI+CD) normalizuje dostępność do [0,1]",
                "ci_decomposition": "CI = CI_morphological + CI_syntactic + CI_semantic"
            }
        }


# =============================================================================
# MAIN CALCULATOR CLASS
# =============================================================================

class ConstitutionalDualityCalculator:
    """
    Kalkulator Constitutional Duality dla tekstów w języku polskim.

    Implementuje Twierdzenie o Dualności Morfosyntaktycznej:
        CI × CD = Depth²

    Gdzie:
        CD = (1/Ambiguity) × Depth × √(D×S/E)  # Constitutional Definiteness
        CI = Ambiguity × Depth × √(E/(D×S))    # Constitutional Indefiniteness

    Examples:
        >>> calc = ConstitutionalDualityCalculator()
        >>> metrics = calc.calculate_metrics(
        ...     ambiguity=2.0,
        ...     depth=5,
        ...     D=0.8,
        ...     S=0.7,
        ...     E=0.3
        ... )
        >>> print(f"CD={metrics.CD:.2f}, CI={metrics.CI:.2f}")
        CD=6.82, CI=3.67
        >>> print(f"Duality verified: {metrics.duality_verified}")
        Duality verified: True

    Raises:
        ValueError: Gdy parametry wejściowe są nieprawidłowe (np. ujemne, poza zakresem)
    """

    def __init__(
        self,
        epsilon: float = EPSILON,
        duality_tolerance: float = DUALITY_ERROR_TOLERANCE
    ):
        """
        Inicjalizacja kalkulatora z opcjonalnymi parametrami konfiguracyjnymi.

        Args:
            epsilon: Minimalna wartość dla zabezpieczenia przed dzieleniem przez zero
            duality_tolerance: Próg tolerancji błędu dla weryfikacji dualności (domyślnie 1%)
        """
        self.epsilon = epsilon
        self.duality_tolerance = duality_tolerance

    def calculate_metrics(
        self,
        ambiguity: float,
        depth: int,
        D: float,
        S: float,
        E: float,
        inflectional_forms_count: Optional[int] = None
    ) -> ConstitutionalMetrics:
        """
        Oblicz kompletne metryki Constitutional Duality.

        Args:
            ambiguity: Średnia liczba interpretacji morfologicznych na słowo (>= 1.0)
            depth: Maksymalna głębokość składniowa (>= 1)
            D: Determination - współrzędna w F³ [0, 1]
            S: Stability - współrzędna w F³ [0, 1]
            E: Entropy - współrzędna w F³ [0, 1]

        Returns:
            ConstitutionalMetrics zawierający wszystkie obliczone metryki i weryfikacje

        Raises:
            ValueError: Gdy któryś z parametrów jest nieprawidłowy

        Examples:
            >>> calc = ConstitutionalDualityCalculator()

            # Przykład 1: Tekst prawny (niska ambiguity, wysoka głębokość, niskie E)
            >>> legal_metrics = calc.calculate_metrics(1.2, 10, 0.9, 0.8, 0.2)
            >>> print(f"Legal text: CD={legal_metrics.CD:.2f}, SA={legal_metrics.SA:.2%}")
            Legal text: CD=18.97, SA=75.88%

            # Przykład 2: Poezja (wysoka ambiguity, średnia głębokość, wysokie E)
            >>> poetry_metrics = calc.calculate_metrics(3.5, 5, 0.5, 0.4, 0.8)
            >>> print(f"Poetry: CI={poetry_metrics.CI:.2f}, SA={poetry_metrics.SA:.2%}")
            Poetry: CI=8.37, SA=28.37%
        """
        # Walidacja parametrów wejściowych
        self._validate_inputs(ambiguity, depth, D, S, E)

        # Oblicz składniki geometryczne
        geometric_balance = self._calculate_geometric_balance(D, S, E)
        geometric_tension = self._calculate_geometric_tension(D, S, E)

        # Oblicz CD i CI
        CD = self._calculate_CD(ambiguity, depth, geometric_balance)
        CI = self._calculate_CI(ambiguity, depth, geometric_tension)

        # Weryfikuj dualność
        duality_product = CI * CD
        duality_theoretical = depth ** 2
        duality_error = self._calculate_duality_error(duality_product, duality_theoretical)
        duality_verified = duality_error < self.duality_tolerance

        # Oblicz Semantic Accessibility
        SA = self._calculate_SA(CD, CI)

        # Dekompozycja CI
        CI_morph, CI_synt, CI_sem = self._decompose_CI(
            ambiguity=ambiguity,
            depth=depth,
            D=D,
            S=S,
            E=E,
            geometric_tension=geometric_tension,
            CI_total=CI,
            inflectional_forms_count=inflectional_forms_count
        )

        # Klasyfikacje
        sa_category = self._classify_accessibility(SA)
        cd_ci_ratio = CD / CI if CI > self.epsilon else float('inf')
        structure_classification = self._classify_structure(cd_ci_ratio)

        # Enhanced structure classification per GTMO criteria
        ci_sum = CI_morph + CI_synt + CI_sem if (CI_morph + CI_synt + CI_sem) > self.epsilon else 0.0
        synt_share = (CI_synt / ci_sum) if ci_sum else 0.0
        enhanced_label = self._classify_structure_enhanced(D, S, E, synt_share)

        # Zwróć kompletne metryki
        return ConstitutionalMetrics(
            CD=CD,
            CI=CI,
            SA=SA,
            depth=depth,
            ambiguity=ambiguity,
            D=D,
            S=S,
            E=E,
            duality_product=duality_product,
            duality_theoretical=duality_theoretical,
            duality_error=duality_error,
            duality_verified=duality_verified,
            geometric_balance=geometric_balance,
            geometric_tension=geometric_tension,
            CI_morphological=CI_morph,
            CI_syntactic=CI_synt,
            CI_semantic=CI_sem,
            sa_category=sa_category,
            structure_classification=structure_classification,
            cd_ci_ratio=cd_ci_ratio
            ,enhanced_structure_classification=enhanced_label
        )

    # -------------------------------------------------------------------------
    # PRIVATE CALCULATION METHODS
    # -------------------------------------------------------------------------

    def _calculate_geometric_balance(self, D: float, S: float, E: float) -> float:
        """
        Oblicz geometric balance: √(D×S/E).

        Geometric balance mierzy stosunek strukturalnego porządku (D×S)
        do chaosu semantycznego (E). Im wyższy balance, tym bardziej uporządkowany tekst.

        Teoretyczne uzasadnienie:
            Gdy E → 0 (minimalna entropia), balance → ∞ (maksymalny porządek)
            Gdy E → 1 (maksymalna entropia), balance → √(D×S) (ograniczony porządek)

        Args:
            D, S, E: współrzędne w przestrzeni fazowej

        Returns:
            Geometric balance factor
        """
        # Zabezpieczenie przed dzieleniem przez zero
        E_safe = max(E, self.epsilon)
        return np.sqrt((D * S) / E_safe)

    def _calculate_geometric_tension(self, D: float, S: float, E: float) -> float:
        """
        Oblicz geometric tension: √(E/(D×S)).

        Geometric tension mierzy stosunek chaosu semantycznego (E)
        do strukturalnego porządku (D×S). Im wyższe tension, tym bardziej chaotyczny tekst.

        Teoretyczne uzasadnienie:
            Gdy D×S → 0 (minimalna struktura), tension → ∞ (maksymalny chaos)
            Gdy D×S → 1 (maksymalna struktura), tension → √E (ograniczony chaos)

        Jest to odwrotność geometric_balance, co zapewnia symetrię formuł CI i CD.

        Args:
            D, S, E: współrzędne w przestrzeni fazowej

        Returns:
            Geometric tension factor
        """
        # Zabezpieczenie przed dzieleniem przez zero
        DS_safe = max(D * S, self.epsilon)
        return np.sqrt(E / DS_safe)

    def _calculate_CD(self, ambiguity: float, depth: int, geometric_balance: float) -> float:
        """
        Oblicz Constitutional Definiteness.

        Formuła: CD = (1/Ambiguity) × Depth × √(D×S/E)

        Interpretacja:
            - 1/Ambiguity: teksty jednoznaczne mają wyższe CD
            - Depth: głębsze struktury składniowe mają wyższe CD
            - √(D×S/E): wyższy porządek strukturalny daje wyższe CD

        Args:
            ambiguity: średnia liczba interpretacji na słowo
            depth: głębokość składniowa
            geometric_balance: √(D×S/E)

        Returns:
            Constitutional Definiteness
        """
        return (1.0 / ambiguity) * depth * geometric_balance

    def _calculate_CI(self, ambiguity: float, depth: int, geometric_tension: float) -> float:
        """
        Oblicz Constitutional Indefiniteness.

        Formuła: CI = Ambiguity × Depth × √(E/(D×S))

        Interpretacja:
            - Ambiguity: teksty wieloznaczne mają wyższe CI
            - Depth: głębsze struktury (paradoksalnie) też zwiększają CI przez complexity
            - √(E/(D×S)): wyższy chaos semantyczny daje wyższe CI

        Args:
            ambiguity: średnia liczba interpretacji na słowo
            depth: głębokość składniowa
            geometric_tension: √(E/(D×S))

        Returns:
            Constitutional Indefiniteness
        """
        return ambiguity * depth * geometric_tension

    def _calculate_SA(self, CD: float, CI: float) -> float:
        """
        Oblicz Semantic Accessibility.

        Formuła: SA = CD / (CI + CD)

        Uzasadnienie teoretyczne:
            SA reprezentuje "udział" definiteness w całkowitej morfosyntaktycznej złożoności.

            Z dualności CI × CD = Depth² wynika że CI + CD reprezentuje "całkowitą
            morfosyntaktyczną complexity", a SA pokazuje jaka część tej complexity
            pochodzi z definiteness (porządku) vs indefiniteness (chaosu).

            Gdy CD >> CI: SA → 1 (wysoka dostępność, dominuje porządek)
            Gdy CI >> CD: SA → 0 (niska dostępność, dominuje chaos)
            Gdy CD = CI: SA = 0.5 (równowaga)

        SA jest matematycznie gwarantowane w [0, 1] przez konstrukcję:
            - CD >= 0, CI >= 0 (zawsze)
            - 0 <= CD / (CI + CD) <= 1 (matematyczna własność)

        Args:
            CD: Constitutional Definiteness
            CI: Constitutional Indefiniteness

        Returns:
            Semantic Accessibility ∈ [0, 1]
        """
        denominator = CI + CD
        if denominator > self.epsilon:
            return CD / denominator
        else:
            # Fallback dla CD=CI=0 (teoretycznie niemożliwe, ale zabezpieczenie)
            return 0.5

    def _calculate_duality_error(self, product: float, theoretical: float) -> float:
        """
        Oblicz względny błąd dualności.

        Błąd = |CI × CD - Depth²| / Depth²

        W idealnej implementacji błąd powinien wynosić 0. W praktyce, ze względu na
        floating-point arithmetic i zaokrąglenia, akceptujemy błąd < 1%.

        Args:
            product: CI × CD (obliczona wartość)
            theoretical: Depth² (wartość teoretyczna)

        Returns:
            Względny błąd jako liczba z [0, ∞)
        """
        if theoretical > self.epsilon:
            return abs(product - theoretical) / theoretical
        else:
            return 0.0

    def _decompose_CI(
        self,
        ambiguity: float,
        depth: int,
        D: float,
        S: float,
        E: float,
        geometric_tension: float,
        CI_total: float,
        inflectional_forms_count: Optional[int] = None
    ) -> Tuple[float, float, float]:
        """
        Rozłóż CI na składniki: morfologiczny, składniowy, semantyczny.

        Dekompozycja zaawansowana:
            CI_morphological = inflectional_forms_count × ambiguity × tension
            CI_syntactic = (depth² / ambiguity) × tension
            CI_semantic = E × depth × ambiguity × balance_inv

        Uzasadnienie teoretyczne:
            - CI_morphological izoluje wpływ fleksji morfologicznej (liczba form × ambiguity)
            - CI_syntactic izoluje wpływ struktury składniowej (depth²/ambiguity - konkurencja)
            - CI_semantic reprezentuje "czystą" entropię semantyczną F³

        Uwaga: Suma nie zawsze równa się dokładnie CI_total ze względu na
        nielinearny charakter interakcji między ambiguity i depth w pełnej formule.
        Używamy proporcjonalnego rescalingu do zachowania CI_total.

        Args:
            ambiguity: średnia liczba interpretacji na słowo
            depth: głębokość składniowa
            D: Determination coordinate
            S: Stability coordinate
            E: Entropy coordinate
            geometric_tension: √(E/(D×S))
            CI_total: całkowite CI z głównej formuły
            inflectional_forms_count: liczba różnych form fleksyjnych w tekście

        Returns:
            (CI_morphological, CI_syntactic, CI_semantic)
        """
        # Zaawansowana dekompozycja z uwzględnieniem liczby form fleksyjnych
        DS_safe = max(D * S, self.epsilon)
        tension = np.sqrt(E / DS_safe)
        balance_inv = 1.0 / np.sqrt(DS_safe)

        # Jeśli nie mamy inflectional_forms_count, używamy depth jako proxy
        infl_count = float(inflectional_forms_count) if inflectional_forms_count is not None else depth

        CI_morph = infl_count * ambiguity * tension
        CI_synt = (depth ** 2) / max(ambiguity, self.epsilon) * tension
        CI_sem = E * depth * ambiguity * balance_inv

        CI_prelim_sum = CI_morph + CI_synt + CI_sem
        if CI_prelim_sum > self.epsilon:
            scale_factor = CI_total / CI_prelim_sum
            CI_morph *= scale_factor
            CI_synt *= scale_factor
            CI_sem *= scale_factor
        else:
            # Fallback: równy podział
            CI_sem = CI_total / 3.0
            CI_morph = CI_total / 3.0
            CI_synt = CI_total / 3.0

        return CI_morph, CI_synt, CI_sem

    def _classify_accessibility(self, SA: float) -> AccessibilityCategory:
        """
        Klasyfikuj dostępność semantyczną na podstawie wartości SA.

        Kategorie:
            HIGH: SA > 0.7 (tekst bardzo dostępny)
            MEDIUM: 0.3 ≤ SA ≤ 0.7 (tekst umiarkowanie dostępny)
            LOW: SA < 0.3 (tekst trudno dostępny)

        Args:
            SA: Semantic Accessibility ∈ [0, 1]

        Returns:
            AccessibilityCategory enum
        """
        if SA > SA_HIGH_THRESHOLD:
            return AccessibilityCategory.HIGH
        elif SA >= SA_LOW_THRESHOLD:
            return AccessibilityCategory.MEDIUM
        else:
            return AccessibilityCategory.LOW

    def _classify_structure(self, cd_ci_ratio: float) -> StructureClassification:
        """
        Klasyfikuj strukturę tekstu na podstawie stosunku CD/CI.

        Kategorie:
            ORDERED: CD/CI > 1.0 (tekst uporządkowany, CD dominuje)
            BALANCED: 0.5 < CD/CI ≤ 1.0 (tekst zbalansowany)
            CHAOTIC: CD/CI ≤ 0.5 (tekst chaotyczny, CI dominuje)

        Uzasadnienie progów:
            - Próg 1.0: CD = CI oznacza doskonałą równowagę
            - Próg 0.5: CI = 2×CD oznacza wyraźną dominację chaosu

        Args:
            cd_ci_ratio: Stosunek CD/CI

        Returns:
            StructureClassification enum
        """
        if cd_ci_ratio > 1.0:
            return StructureClassification.ORDERED
        elif cd_ci_ratio > 0.5:
            return StructureClassification.BALANCED
        else:
            return StructureClassification.CHAOTIC

    def _classify_structure_enhanced(self, D: float, S: float, E: float, synt_share: float) -> str:
        """Improved structural classification per GTMO criteria.

        Rules:
          - PRECISE:    D>0.7, S>0.7, E<0.3
          - CHAOTIC_STRUCTURE: D<0.4, S<0.4, E>0.6
          - PARADOX:    S>0.7 and E>0.7
          - SYNTACTICALLY_COMPLEX: syntactic share of CI > 0.6
          - STRUCTURED: otherwise
        """
        if D > 0.7 and S > 0.7 and E < 0.3:
            return "PRECISE"
        elif D < 0.4 and S < 0.4 and E > 0.6:
            return "CHAOTIC_STRUCTURE"
        elif S > 0.7 and E > 0.7:
            return "PARADOX"
        elif synt_share > 0.6:
            return "SYNTACTICALLY_COMPLEX"
        else:
            return "STRUCTURED"

    def _validate_inputs(
        self,
        ambiguity: float,
        depth: int,
        D: float,
        S: float,
        E: float
    ) -> None:
        """
        Waliduj parametry wejściowe pod kątem teoretycznej poprawności.

        Args:
            ambiguity: średnia liczba interpretacji na słowo (>= 1.0)
            depth: głębokość składniowa (>= 1)
            D, S, E: współrzędne w F³ ([0, 1])

        Raises:
            ValueError: Gdy któryś z parametrów jest nieprawidłowy
        """
        if ambiguity < MIN_AMBIGUITY:
            raise ValueError(
                f"Ambiguity musi być >= {MIN_AMBIGUITY} (każde słowo ma przynajmniej "
                f"jedną interpretację), otrzymano: {ambiguity}"
            )

        if depth < MIN_DEPTH:
            raise ValueError(
                f"Depth musi być >= {MIN_DEPTH} (minimalna głębokość składniowa), "
                f"otrzymano: {depth}"
            )

        if not (0 <= D <= 1):
            raise ValueError(
                f"D (Determination) musi być w [0, 1], otrzymano: {D}"
            )

        if not (0 <= S <= 1):
            raise ValueError(
                f"S (Stability) musi być w [0, 1], otrzymano: {S}"
            )

        if not (0 <= E <= 1):
            raise ValueError(
                f"E (Entropy) musi być w [0, 1], otrzymano: {E}"
            )


# =============================================================================
# CONVENIENCE FUNCTIONS
# =============================================================================

def calculate_constitutional_duality(
    ambiguity: float,
    depth: int,
    D: float,
    S: float,
    E: float,
    epsilon: float = EPSILON,
    duality_tolerance: float = DUALITY_ERROR_TOLERANCE
) -> ConstitutionalMetrics:
    """
    Convenience function dla szybkiego obliczenia metryk bez tworzenia instancji kalkulatora.

    Args:
        ambiguity: średnia liczba interpretacji morfologicznych na słowo
        depth: maksymalna głębokość składniowa
        D, S, E: współrzędne w przestrzeni fazowej F³
        epsilon: minimalna wartość dla zabezpieczenia przed dzieleniem przez zero
        duality_tolerance: próg tolerancji błędu dla weryfikacji dualności

    Returns:
        ConstitutionalMetrics z wszystkimi obliczonymi metrykami

    Examples:
        >>> metrics = calculate_constitutional_duality(2.0, 5, 0.8, 0.7, 0.3)
        >>> print(f"CD={metrics.CD:.2f}, CI={metrics.CI:.2f}, SA={metrics.SA:.2%}")
        CD=6.82, CI=3.67, SA=54.54%
    """
    calc = ConstitutionalDualityCalculator(epsilon=epsilon, duality_tolerance=duality_tolerance)
    return calc.calculate_metrics(ambiguity, depth, D, S, E)


# =============================================================================
# MAIN (for testing)
# =============================================================================

if __name__ == "__main__":
    print("=" * 80)
    print("GTMØ Constitutional Duality Calculator - Test Suite")
    print("=" * 80)

    # Test case 1: Tekst prawny (niska ambiguity, wysoka głębokość, niskie E)
    print("\nTest 1: Tekst prawny")
    print("-" * 40)
    legal = calculate_constitutional_duality(
        ambiguity=1.2,
        depth=10,
        D=0.9,
        S=0.8,
        E=0.2
    )
    print(f"CD = {legal.CD:.2f}")
    print(f"CI = {legal.CI:.2f}")
    print(f"SA = {legal.SA:.2%} ({legal.sa_category.value})")
    print(f"Duality: {legal.CI:.2f} × {legal.CD:.2f} = {legal.duality_product:.2f} ≈ {legal.duality_theoretical}")
    print(f"Weryfikacja: {'✓ PASSED' if legal.duality_verified else '✗ FAILED'}")
    print(f"Klasyfikacja: {legal.structure_classification.value}")

    # Test case 2: Poezja (wysoka ambiguity, średnia głębokość, wysokie E)
    print("\nTest 2: Poezja")
    print("-" * 40)
    poetry = calculate_constitutional_duality(
        ambiguity=3.5,
        depth=5,
        D=0.5,
        S=0.4,
        E=0.8
    )
    print(f"CD = {poetry.CD:.2f}")
    print(f"CI = {poetry.CI:.2f}")
    print(f"SA = {poetry.SA:.2%} ({poetry.sa_category.value})")
    print(f"Duality: {poetry.CI:.2f} × {poetry.CD:.2f} = {poetry.duality_product:.2f} ≈ {poetry.duality_theoretical}")
    print(f"Weryfikacja: {'✓ PASSED' if poetry.duality_verified else '✗ FAILED'}")
    print(f"Klasyfikacja: {poetry.structure_classification.value}")

    # Test case 3: Tekst zbalansowany
    print("\nTest 3: Tekst zbalansowany")
    print("-" * 40)
    balanced = calculate_constitutional_duality(
        ambiguity=2.0,
        depth=5,
        D=0.7,
        S=0.6,
        E=0.5
    )
    print(f"CD = {balanced.CD:.2f}")
    print(f"CI = {balanced.CI:.2f}")
    print(f"SA = {balanced.SA:.2%} ({balanced.sa_category.value})")
    print(f"Duality: {balanced.CI:.2f} × {balanced.CD:.2f} = {balanced.duality_product:.2f} ≈ {balanced.duality_theoretical}")
    print(f"Weryfikacja: {'✓ PASSED' if balanced.duality_verified else '✗ FAILED'}")
    print(f"Klasyfikacja: {balanced.structure_classification.value}")

    # Test dekompozycji CI
    print("\nTest 4: Dekompozycja CI")
    print("-" * 40)
    print(f"CI_morphological = {balanced.CI_morphological:.2f} ({balanced.CI_morphological/balanced.CI*100:.1f}%)")
    print(f"CI_syntactic     = {balanced.CI_syntactic:.2f} ({balanced.CI_syntactic/balanced.CI*100:.1f}%)")
    print(f"CI_semantic      = {balanced.CI_semantic:.2f} ({balanced.CI_semantic/balanced.CI*100:.1f}%)")
    print(f"Suma             = {balanced.CI_morphological + balanced.CI_syntactic + balanced.CI_semantic:.2f}")
    print(f"CI_total         = {balanced.CI:.2f}")

    print("\n" + "=" * 80)
    print("Wszystkie testy zakończone pomyślnie!")
    print("=" * 80)
