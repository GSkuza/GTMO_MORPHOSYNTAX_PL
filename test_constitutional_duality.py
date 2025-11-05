#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# Fix Windows console encoding for Unicode characters
import sys
if sys.platform == 'win32':
    import io
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8')

"""
Test Suite for GTMØ Constitutional Duality Calculator
======================================================
Comprehensive unit tests and property-based tests dla weryfikacji
poprawności implementacji Constitutional Definiteness (CD) i
Constitutional Indefiniteness (CI).

Test Categories:
1. Unit Tests - weryfikacja podstawowych obliczeń
2. Duality Tests - weryfikacja CI × CD = Depth²
3. Edge Case Tests - obsługa przypadków brzegowych
4. Property-Based Tests - weryfikacja własności matematycznych (Hypothesis)
5. Regression Tests - golden dataset dla regresji

Author: GTMØ Team
Date: 2025-01-XX
Version: 1.0
"""

import unittest
import numpy as np
from typing import List, Tuple
import sys

# Import modułu do testowania
from gtmo_constitutional_duality import (
    ConstitutionalDualityCalculator,
    calculate_constitutional_duality,
    ConstitutionalMetrics,
    AccessibilityCategory,
    StructureClassification,
    EPSILON,
    SA_HIGH_THRESHOLD,
    SA_LOW_THRESHOLD,
    MIN_AMBIGUITY,
    MIN_DEPTH
)

# Próbuj załadować Hypothesis dla property-based testing
try:
    from hypothesis import given, strategies as st, assume, settings
    from hypothesis import reproduce_failure  # dla debugowania
    HYPOTHESIS_AVAILABLE = True
except ImportError:
    HYPOTHESIS_AVAILABLE = False
    print("⚠️  Hypothesis not available. Install with: pip install hypothesis")
    print("   Property-based tests będą pominięte.")


# =============================================================================
# BASE TEST CLASS
# =============================================================================

class BaseConstitutionalDualityTest(unittest.TestCase):
    """Bazowa klasa testowa z helper methods."""

    def setUp(self):
        """Setup przed każdym testem."""
        self.calc = ConstitutionalDualityCalculator()
        self.default_epsilon = EPSILON

    def assertDualityVerified(self, metrics: ConstitutionalMetrics, msg: str = ""):
        """
        Assert że dualność została zweryfikowana (CI × CD = Depth²).

        Args:
            metrics: ConstitutionalMetrics do sprawdzenia
            msg: Optional custom message
        """
        self.assertTrue(
            metrics.duality_verified,
            msg or f"Duality not verified: CI({metrics.CI:.4f}) × CD({metrics.CD:.4f}) = "
                   f"{metrics.duality_product:.4f} ≠ Depth²({metrics.duality_theoretical}), "
                   f"error={metrics.duality_error:.4%}"
        )

    def assertAlmostEqualWithTolerance(
        self,
        actual: float,
        expected: float,
        tolerance: float = 0.01,
        msg: str = ""
    ):
        """
        Assert że actual ≈ expected w granicach tolerancji.

        Args:
            actual: Wartość rzeczywista
            expected: Wartość oczekiwana
            tolerance: Względna tolerancja (domyślnie 1%)
            msg: Optional custom message
        """
        if expected != 0:
            relative_error = abs(actual - expected) / abs(expected)
        else:
            relative_error = abs(actual - expected)

        self.assertLess(
            relative_error,
            tolerance,
            msg or f"Wartość {actual:.4f} różni się od oczekiwanej {expected:.4f} "
                   f"o {relative_error:.2%} (tolerancja: {tolerance:.2%})"
        )


# =============================================================================
# UNIT TESTS - Basic Functionality
# =============================================================================

class TestBasicCalculations(BaseConstitutionalDualityTest):
    """Testy podstawowych obliczeń CD, CI, SA."""

    def test_simple_case_all_equal(self):
        """Test dla najprostszego przypadku: ambiguity=1, depth=1, D=S=E=0.5."""
        metrics = self.calc.calculate_metrics(
            ambiguity=1.0,
            depth=1,
            D=0.5,
            S=0.5,
            E=0.5
        )

        # CD = (1/1) × 1 × √(0.5×0.5/0.5) = 1 × 1 × √0.5 = 0.707...
        expected_CD = np.sqrt(0.5)
        self.assertAlmostEqual(metrics.CD, expected_CD, places=4)

        # CI = 1 × 1 × √(0.5/(0.5×0.5)) = 1 × √2 = 1.414...
        expected_CI = np.sqrt(2.0)
        self.assertAlmostEqual(metrics.CI, expected_CI, places=4)

        # Duality: CI × CD powinno = Depth² = 1
        self.assertDualityVerified(metrics)
        self.assertAlmostEqual(metrics.duality_product, 1.0, places=4)

    def test_high_ambiguity_increases_CI(self):
        """Wysoka ambiguity powinna zwiększyć CI przy stałych innych parametrach."""
        low_amb = self.calc.calculate_metrics(1.0, 5, 0.7, 0.6, 0.4)
        high_amb = self.calc.calculate_metrics(3.0, 5, 0.7, 0.6, 0.4)

        self.assertGreater(
            high_amb.CI,
            low_amb.CI,
            "Wyższa ambiguity powinna dać wyższe CI"
        )
        self.assertLess(
            high_amb.CD,
            low_amb.CD,
            "Wyższa ambiguity powinna dać niższe CD (odwrotna zależność)"
        )

    def test_high_depth_increases_both_CI_and_CD(self):
        """Wysoka głębokość składniowa powinna zwiększyć zarówno CI jak i CD."""
        shallow = self.calc.calculate_metrics(2.0, 2, 0.7, 0.6, 0.4)
        deep = self.calc.calculate_metrics(2.0, 10, 0.7, 0.6, 0.4)

        self.assertGreater(deep.CI, shallow.CI, "Większa głębokość → wyższe CI")
        self.assertGreater(deep.CD, shallow.CD, "Większa głębokość → wyższe CD")

        # Ale stosunek CI×CD rośnie kwadratowo z depth
        self.assertAlmostEqual(
            deep.duality_product / shallow.duality_product,
            (10 / 2) ** 2,  # (depth_deep / depth_shallow)²
            places=1
        )

    def test_high_entropy_increases_CI_decreases_CD(self):
        """Wysoka entropia powinna zwiększyć CI i zmniejszyć CD."""
        low_E = self.calc.calculate_metrics(2.0, 5, 0.7, 0.6, 0.2)
        high_E = self.calc.calculate_metrics(2.0, 5, 0.7, 0.6, 0.8)

        self.assertGreater(high_E.CI, low_E.CI, "Wyższa entropia → wyższe CI")
        self.assertLess(high_E.CD, low_E.CD, "Wyższa entropia → niższe CD")

    def test_SA_normalization(self):
        """SA powinno zawsze być w przedziale [0, 1]."""
        # Test różnych kombinacji parametrów
        test_cases = [
            (1.0, 1, 0.9, 0.9, 0.1),   # Bardzo uporządkowany
            (5.0, 10, 0.3, 0.2, 0.9),  # Bardzo chaotyczny
            (2.0, 5, 0.5, 0.5, 0.5),   # Zbalansowany
        ]

        for amb, d, D, S, E in test_cases:
            with self.subTest(ambiguity=amb, depth=d, D=D, S=S, E=E):
                metrics = self.calc.calculate_metrics(amb, d, D, S, E)
                self.assertGreaterEqual(
                    metrics.SA, 0.0,
                    f"SA={metrics.SA} nie może być ujemne"
                )
                self.assertLessEqual(
                    metrics.SA, 1.0,
                    f"SA={metrics.SA} nie może przekraczać 1.0"
                )


# =============================================================================
# DUALITY TESTS - Verification of CI × CD = Depth²
# =============================================================================

class TestDualityRelation(BaseConstitutionalDualityTest):
    """Testy weryfikujące relację dualności CI × CD = Depth²."""

    def test_duality_perfect_for_random_inputs(self):
        """Dualność powinna zachodzić dla losowych poprawnych inputów."""
        np.random.seed(42)  # Reproducibility

        for _ in range(100):  # 100 losowych przypadków
            ambiguity = np.random.uniform(1.0, 5.0)
            depth = np.random.randint(1, 20)
            D = np.random.uniform(0.1, 0.9)
            S = np.random.uniform(0.1, 0.9)
            E = np.random.uniform(0.1, 0.9)

            metrics = self.calc.calculate_metrics(ambiguity, depth, D, S, E)

            # Weryfikuj CI × CD = Depth²
            self.assertDualityVerified(metrics)

            # Dodatkowa weryfikacja matematyczna
            expected_product = depth ** 2
            self.assertAlmostEqualWithTolerance(
                metrics.duality_product,
                expected_product,
                tolerance=0.01
            )

    def test_duality_derivation_from_formulas(self):
        """
        Weryfikuj że dualność wynika bezpośrednio z formuł CI i CD.

        Matematycznie:
            CD = (1/A) × D × √(D×S/E)
            CI = A × D × √(E/(D×S))

            CI × CD = [A × D × √(E/(D×S))] × [(1/A) × D × √(D×S/E)]
                    = D² × [√(E/(D×S)) × √(D×S/E)]
                    = D² × √[(E/(D×S)) × (D×S/E)]
                    = D² × √1
                    = D²
        """
        # Test dla konkretnych wartości
        A = 2.5
        D_depth = 7
        D = 0.8
        S = 0.6
        E = 0.4

        metrics = self.calc.calculate_metrics(A, D_depth, D, S, E)

        # Ręczne obliczenie według formuł
        geom_balance = np.sqrt((D * S) / E)
        geom_tension = np.sqrt(E / (D * S))

        CD_manual = (1.0 / A) * D_depth * geom_balance
        CI_manual = A * D_depth * geom_tension

        product_manual = CI_manual * CD_manual

        # Weryfikuj zgodność
        self.assertAlmostEqual(metrics.CD, CD_manual, places=6)
        self.assertAlmostEqual(metrics.CI, CI_manual, places=6)
        self.assertAlmostEqual(product_manual, D_depth ** 2, places=6)

    def test_SA_equals_CD_over_CI_plus_CD(self):
        """Weryfikuj że SA = CD / (CI + CD)."""
        metrics = self.calc.calculate_metrics(2.0, 5, 0.7, 0.6, 0.3)

        expected_SA = metrics.CD / (metrics.CI + metrics.CD)
        self.assertAlmostEqual(metrics.SA, expected_SA, places=6)


# =============================================================================
# EDGE CASE TESTS
# =============================================================================

class TestEdgeCases(BaseConstitutionalDualityTest):
    """Testy obsługi przypadków brzegowych."""

    def test_minimal_valid_inputs(self):
        """Test dla minimalnych poprawnych wartości."""
        # Minimalne: ambiguity=1.0, depth=1, D=S=E=ε (prawie 0)
        metrics = self.calc.calculate_metrics(
            ambiguity=1.0,
            depth=1,
            D=0.01,
            S=0.01,
            E=0.01
        )

        # Powinno działać bez błędów
        self.assertIsInstance(metrics, ConstitutionalMetrics)
        self.assertDualityVerified(metrics)

    def test_maximal_valid_inputs(self):
        """Test dla maksymalnych poprawnych wartości."""
        # Maksymalne: wysokie ambiguity, głębokość, D=S=E=1.0
        metrics = self.calc.calculate_metrics(
            ambiguity=10.0,
            depth=50,
            D=1.0,
            S=1.0,
            E=1.0
        )

        self.assertIsInstance(metrics, ConstitutionalMetrics)
        self.assertDualityVerified(metrics)

    def test_near_zero_entropy(self):
        """Test dla bardzo niskiej entropii (E → 0)."""
        # Gdy E → 0, CD powinno rosnąć (tekst uporządkowany)
        metrics = self.calc.calculate_metrics(
            ambiguity=1.5,
            depth=5,
            D=0.9,
            S=0.8,
            E=0.001  # Bardzo niska entropia
        )

        # CD powinno być wysokie (uporządkowany tekst)
        self.assertGreater(metrics.CD, 10.0)

        # CI powinno być niskie
        self.assertLess(metrics.CI, 5.0)

        # SA powinno być wysokie (> 0.7)
        self.assertGreater(metrics.SA, 0.7)
        self.assertEqual(metrics.sa_category, AccessibilityCategory.HIGH)

        # Dualność nadal zachowana
        self.assertDualityVerified(metrics)

    def test_near_zero_D_times_S(self):
        """Test dla bardzo niskiego D×S (minimalna struktura)."""
        metrics = self.calc.calculate_metrics(
            ambiguity=3.0,
            depth=5,
            D=0.001,
            S=0.001,
            E=0.8
        )

        # CI powinno być bardzo wysokie (chaos)
        self.assertGreater(metrics.CI, 20.0)

        # CD powinno być bardzo niskie
        self.assertLess(metrics.CD, 1.0)

        # SA powinno być niskie
        self.assertLess(metrics.SA, 0.3)
        self.assertEqual(metrics.sa_category, AccessibilityCategory.LOW)

        # Dualność nadal zachowana
        self.assertDualityVerified(metrics)

    def test_invalid_ambiguity_below_one(self):
        """Ambiguity < 1.0 powinno rzucić ValueError."""
        with self.assertRaises(ValueError) as ctx:
            self.calc.calculate_metrics(0.5, 5, 0.7, 0.6, 0.3)

        self.assertIn("Ambiguity", str(ctx.exception))
        self.assertIn("1.0", str(ctx.exception))

    def test_invalid_depth_zero(self):
        """Depth = 0 powinno rzucić ValueError."""
        with self.assertRaises(ValueError) as ctx:
            self.calc.calculate_metrics(2.0, 0, 0.7, 0.6, 0.3)

        self.assertIn("Depth", str(ctx.exception))

    def test_invalid_coordinates_out_of_range(self):
        """D, S, E poza [0, 1] powinny rzucić ValueError."""
        # D > 1
        with self.assertRaises(ValueError):
            self.calc.calculate_metrics(2.0, 5, 1.5, 0.6, 0.3)

        # S < 0
        with self.assertRaises(ValueError):
            self.calc.calculate_metrics(2.0, 5, 0.7, -0.1, 0.3)

        # E > 1
        with self.assertRaises(ValueError):
            self.calc.calculate_metrics(2.0, 5, 0.7, 0.6, 1.2)


# =============================================================================
# CLASSIFICATION TESTS
# =============================================================================

class TestClassifications(BaseConstitutionalDualityTest):
    """Testy klasyfikacji SA i struktury."""

    def test_sa_classification_high(self):
        """SA > 0.7 powinno dać HIGH accessibility."""
        # Tekst prawny: niska amb, niska E, wysoka D,S
        metrics = self.calc.calculate_metrics(1.2, 10, 0.9, 0.8, 0.2)

        self.assertGreater(metrics.SA, SA_HIGH_THRESHOLD)
        self.assertEqual(metrics.sa_category, AccessibilityCategory.HIGH)

    def test_sa_classification_medium(self):
        """0.3 ≤ SA ≤ 0.7 powinno dać MEDIUM accessibility."""
        metrics = self.calc.calculate_metrics(2.0, 5, 0.8, 0.7, 0.3)

        self.assertGreaterEqual(metrics.SA, SA_LOW_THRESHOLD)
        self.assertLessEqual(metrics.SA, SA_HIGH_THRESHOLD)
        self.assertEqual(metrics.sa_category, AccessibilityCategory.MEDIUM)

    def test_sa_classification_low(self):
        """SA < 0.3 powinno dać LOW accessibility."""
        # Tekst chaotyczny: wysoka amb, wysoka E, niskie D,S
        metrics = self.calc.calculate_metrics(4.0, 8, 0.3, 0.2, 0.9)

        self.assertLess(metrics.SA, SA_LOW_THRESHOLD)
        self.assertEqual(metrics.sa_category, AccessibilityCategory.LOW)

    def test_structure_classification_ordered(self):
        """CD > CI powinno dać ORDERED structure."""
        metrics = self.calc.calculate_metrics(1.2, 10, 0.9, 0.8, 0.2)

        self.assertGreater(metrics.cd_ci_ratio, 1.0)
        self.assertEqual(metrics.structure_classification, StructureClassification.ORDERED)

    def test_structure_classification_chaotic(self):
        """CI > 2×CD powinno dać CHAOTIC structure."""
        metrics = self.calc.calculate_metrics(4.0, 8, 0.3, 0.2, 0.9)

        self.assertLess(metrics.cd_ci_ratio, 0.5)
        self.assertEqual(metrics.structure_classification, StructureClassification.CHAOTIC)

    def test_structure_classification_balanced(self):
        """0.5 < CD/CI ≤ 1.0 powinno dać BALANCED structure."""
        metrics = self.calc.calculate_metrics(1.1, 5, 0.7, 0.6, 0.4)

        self.assertGreater(metrics.cd_ci_ratio, 0.5)
        self.assertLessEqual(metrics.cd_ci_ratio, 1.0)
        self.assertEqual(metrics.structure_classification, StructureClassification.BALANCED)


# =============================================================================
# CI DECOMPOSITION TESTS
# =============================================================================

class TestCIDecomposition(BaseConstitutionalDualityTest):
    """Testy dekompozycji CI na składniki morfologiczny, składniowy, semantyczny."""

    def test_decomposition_sum_equals_total(self):
        """Suma składników CI powinna być ≈ CI_total."""
        metrics = self.calc.calculate_metrics(2.5, 7, 0.7, 0.6, 0.4)

        ci_sum = (
            metrics.CI_morphological +
            metrics.CI_syntactic +
            metrics.CI_semantic
        )

        # Z powodu rescalingu w dekompozycji, suma powinna być bliska CI
        self.assertAlmostEqualWithTolerance(ci_sum, metrics.CI, tolerance=0.05)

    def test_high_ambiguity_increases_morphological_component(self):
        """Wysoka ambiguity powinna zwiększyć składnik morfologiczny."""
        low_amb = self.calc.calculate_metrics(1.2, 5, 0.7, 0.6, 0.4)
        high_amb = self.calc.calculate_metrics(4.0, 5, 0.7, 0.6, 0.4)

        # Przy tej samej depth i D,S,E, wysokie ambiguity → większy CI_morph
        self.assertGreater(
            high_amb.CI_morphological,
            low_amb.CI_morphological
        )

    def test_high_depth_increases_syntactic_component(self):
        """Wysoka głębokość powinna zwiększyć składnik składniowy."""
        shallow = self.calc.calculate_metrics(2.0, 2, 0.7, 0.6, 0.4)
        deep = self.calc.calculate_metrics(2.0, 15, 0.7, 0.6, 0.4)

        self.assertGreater(deep.CI_syntactic, shallow.CI_syntactic)

    def test_all_components_non_negative(self):
        """Wszystkie składniki CI powinny być nieujemne."""
        # Test dla różnych kombinacji
        test_cases = [
            (1.0, 1, 0.9, 0.9, 0.1),
            (5.0, 20, 0.3, 0.2, 0.9),
            (2.5, 10, 0.5, 0.5, 0.5),
        ]

        for amb, d, D, S, E in test_cases:
            metrics = self.calc.calculate_metrics(amb, d, D, S, E)

            self.assertGreaterEqual(metrics.CI_morphological, 0.0)
            self.assertGreaterEqual(metrics.CI_syntactic, 0.0)
            self.assertGreaterEqual(metrics.CI_semantic, 0.0)


# =============================================================================
# JSON SERIALIZATION TESTS
# =============================================================================

class TestJSONSerialization(BaseConstitutionalDualityTest):
    """Testy serializacji do JSON."""

    def test_to_dict_returns_valid_structure(self):
        """Metoda to_dict() powinna zwrócić poprawną strukturę JSON."""
        metrics = self.calc.calculate_metrics(2.0, 5, 0.7, 0.6, 0.4)
        result = metrics.to_dict()

        # Sprawdź główne klucze
        self.assertIn("definiteness", result)
        self.assertIn("indefiniteness", result)
        self.assertIn("semantic_accessibility", result)
        self.assertIn("duality", result)
        self.assertIn("classification", result)
        self.assertIn("theoretical_basis", result)

        # Sprawdź strukturę definiteness
        self.assertIn("value", result["definiteness"])
        self.assertIn("formula", result["definiteness"])
        self.assertIn("components", result["definiteness"])

        # Sprawdź dekompozycję CI
        self.assertIn("decomposition", result["indefiniteness"])
        decomp = result["indefiniteness"]["decomposition"]
        self.assertIn("morphological", decomp)
        self.assertIn("syntactic", decomp)
        self.assertIn("semantic", decomp)

    def test_to_dict_json_serializable(self):
        """Wynik to_dict() powinien być JSON-serializable."""
        import json

        metrics = self.calc.calculate_metrics(2.0, 5, 0.7, 0.6, 0.4)
        result = metrics.to_dict()

        # Nie powinno rzucić wyjątku
        try:
            json_str = json.dumps(result, ensure_ascii=False)
            self.assertIsInstance(json_str, str)
        except TypeError as e:
            self.fail(f"to_dict() nie jest JSON-serializable: {e}")


# =============================================================================
# PROPERTY-BASED TESTS (Hypothesis)
# =============================================================================

if HYPOTHESIS_AVAILABLE:

    class TestPropertyBased(BaseConstitutionalDualityTest):
        """Property-based tests używające Hypothesis."""

        @given(
            ambiguity=st.floats(min_value=1.0, max_value=10.0),
            depth=st.integers(min_value=1, max_value=50),
            D=st.floats(min_value=0.01, max_value=0.99),
            S=st.floats(min_value=0.01, max_value=0.99),
            E=st.floats(min_value=0.01, max_value=0.99)
        )
        @settings(max_examples=200)  # 200 losowych przypadków
        def test_duality_always_holds(self, ambiguity, depth, D, S, E):
            """
            PROPERTY: Dualność CI × CD = Depth² ZAWSZE zachodzi dla poprawnych inputów.
            """
            metrics = self.calc.calculate_metrics(ambiguity, depth, D, S, E)

            # Weryfikuj dualność
            self.assertDualityVerified(metrics)

        @given(
            ambiguity=st.floats(min_value=1.0, max_value=10.0),
            depth=st.integers(min_value=1, max_value=50),
            D=st.floats(min_value=0.01, max_value=0.99),
            S=st.floats(min_value=0.01, max_value=0.99),
            E=st.floats(min_value=0.01, max_value=0.99)
        )
        @settings(max_examples=200)
        def test_CD_always_positive(self, ambiguity, depth, D, S, E):
            """PROPERTY: CD zawsze dodatnie dla poprawnych inputów."""
            metrics = self.calc.calculate_metrics(ambiguity, depth, D, S, E)
            self.assertGreater(metrics.CD, 0.0, "CD musi być dodatnie")

        @given(
            ambiguity=st.floats(min_value=1.0, max_value=10.0),
            depth=st.integers(min_value=1, max_value=50),
            D=st.floats(min_value=0.01, max_value=0.99),
            S=st.floats(min_value=0.01, max_value=0.99),
            E=st.floats(min_value=0.01, max_value=0.99)
        )
        @settings(max_examples=200)
        def test_CI_always_positive(self, ambiguity, depth, D, S, E):
            """PROPERTY: CI zawsze dodatnie dla poprawnych inputów."""
            metrics = self.calc.calculate_metrics(ambiguity, depth, D, S, E)
            self.assertGreater(metrics.CI, 0.0, "CI musi być dodatnie")

        @given(
            ambiguity=st.floats(min_value=1.0, max_value=10.0),
            depth=st.integers(min_value=1, max_value=50),
            D=st.floats(min_value=0.01, max_value=0.99),
            S=st.floats(min_value=0.01, max_value=0.99),
            E=st.floats(min_value=0.01, max_value=0.99)
        )
        @settings(max_examples=200)
        def test_SA_in_valid_range(self, ambiguity, depth, D, S, E):
            """PROPERTY: SA zawsze w [0, 1]."""
            metrics = self.calc.calculate_metrics(ambiguity, depth, D, S, E)
            self.assertGreaterEqual(metrics.SA, 0.0, "SA nie może być < 0")
            self.assertLessEqual(metrics.SA, 1.0, "SA nie może być > 1")

        @given(
            depth=st.integers(min_value=1, max_value=50),
            D=st.floats(min_value=0.01, max_value=0.99),
            S=st.floats(min_value=0.01, max_value=0.99),
            E=st.floats(min_value=0.01, max_value=0.99)
        )
        @settings(max_examples=100)
        def test_increasing_ambiguity_increases_CI(self, depth, D, S, E):
            """PROPERTY: Zwiększenie ambiguity (przy stałych innych) zwiększa CI."""
            # Dwa poziomy ambiguity
            amb_low = 1.5
            amb_high = 4.0

            metrics_low = self.calc.calculate_metrics(amb_low, depth, D, S, E)
            metrics_high = self.calc.calculate_metrics(amb_high, depth, D, S, E)

            self.assertGreater(
                metrics_high.CI,
                metrics_low.CI,
                f"Wyższa ambiguity ({amb_high}) powinna dać wyższe CI niż ({amb_low})"
            )

        @given(
            ambiguity=st.floats(min_value=1.0, max_value=10.0),
            D=st.floats(min_value=0.01, max_value=0.99),
            S=st.floats(min_value=0.01, max_value=0.99),
            E=st.floats(min_value=0.01, max_value=0.99)
        )
        @settings(max_examples=100)
        def test_increasing_depth_increases_both_CI_CD(self, ambiguity, D, S, E):
            """PROPERTY: Zwiększenie depth zwiększa zarówno CI jak i CD."""
            depth_low = 2
            depth_high = 15

            metrics_low = self.calc.calculate_metrics(ambiguity, depth_low, D, S, E)
            metrics_high = self.calc.calculate_metrics(ambiguity, depth_high, D, S, E)

            self.assertGreater(metrics_high.CI, metrics_low.CI)
            self.assertGreater(metrics_high.CD, metrics_low.CD)


# =============================================================================
# REGRESSION TESTS - Golden Dataset
# =============================================================================

class TestRegressionGoldenDataset(BaseConstitutionalDualityTest):
    """
    Regression tests używające golden dataset z known-good outputs.

    Te testy służą do wykrycia regresji - jeśli zmiana kodu zmienia wyniki
    dla tych przypadków, coś poszło nie tak.
    """

    # Golden dataset: (input params, expected CD, expected CI, expected SA)
    # Wygenerowane przez zweryfikowaną implementację v1.0
    GOLDEN_CASES = [
        # (ambiguity, depth, D, S, E) -> (expected_CD, expected_CI, expected_SA)
        ((1.0, 1, 0.5, 0.5, 0.5), (0.7071, 1.4142, 0.3333)),
        ((2.0, 5, 0.8, 0.7, 0.3), (3.4157, 7.3193, 0.3182)),
        ((1.2, 10, 0.9, 0.8, 0.2), (15.8114, 6.3246, 0.7143)),
        ((3.5, 5, 0.5, 0.4, 0.8), (0.7143, 35.0000, 0.0200)),
        ((2.5, 7, 0.7, 0.6, 0.4), (2.8691, 17.0783, 0.1438)),
    ]

    def test_golden_dataset_CD(self):
        """Weryfikuj że CD dla golden cases jest identyczne."""
        for (inputs, expected) in self.GOLDEN_CASES:
            amb, d, D, S, E = inputs
            expected_CD, _, _ = expected

            with self.subTest(inputs=inputs):
                metrics = self.calc.calculate_metrics(amb, d, D, S, E)
                self.assertAlmostEqual(
                    metrics.CD,
                    expected_CD,
                    places=4,
                    msg=f"Regression detected in CD for {inputs}"
                )

    def test_golden_dataset_CI(self):
        """Weryfikuj że CI dla golden cases jest identyczne."""
        for (inputs, expected) in self.GOLDEN_CASES:
            amb, d, D, S, E = inputs
            _, expected_CI, _ = expected

            with self.subTest(inputs=inputs):
                metrics = self.calc.calculate_metrics(amb, d, D, S, E)
                self.assertAlmostEqual(
                    metrics.CI,
                    expected_CI,
                    places=4,
                    msg=f"Regression detected in CI for {inputs}"
                )

    def test_golden_dataset_SA(self):
        """Weryfikuj że SA dla golden cases jest identyczne."""
        for (inputs, expected) in self.GOLDEN_CASES:
            amb, d, D, S, E = inputs
            _, _, expected_SA = expected

            with self.subTest(inputs=inputs):
                metrics = self.calc.calculate_metrics(amb, d, D, S, E)
                self.assertAlmostEqual(
                    metrics.SA,
                    expected_SA,
                    places=4,
                    msg=f"Regression detected in SA for {inputs}"
                )


# =============================================================================
# CONVENIENCE FUNCTION TESTS
# =============================================================================

class TestConvenienceFunction(unittest.TestCase):
    """Testy convenience function calculate_constitutional_duality()."""

    def test_convenience_function_works(self):
        """Convenience function powinno działać identycznie jak kalkulator."""
        inputs = (2.0, 5, 0.7, 0.6, 0.4)

        # Przez kalkulator
        calc = ConstitutionalDualityCalculator()
        metrics_calc = calc.calculate_metrics(*inputs)

        # Przez convenience function
        metrics_func = calculate_constitutional_duality(*inputs)

        # Powinny być identyczne
        self.assertAlmostEqual(metrics_calc.CD, metrics_func.CD, places=10)
        self.assertAlmostEqual(metrics_calc.CI, metrics_func.CI, places=10)
        self.assertAlmostEqual(metrics_calc.SA, metrics_func.SA, places=10)


# =============================================================================
# TEST RUNNER
# =============================================================================

def run_tests(verbosity=2):
    """
    Uruchom wszystkie testy.

    Args:
        verbosity: Poziom szczegółowości (0=quiet, 1=normal, 2=verbose)

    Returns:
        TestResult object
    """
    # Stwórz test suite
    loader = unittest.TestLoader()
    suite = unittest.TestSuite()

    # Dodaj wszystkie test classes
    suite.addTests(loader.loadTestsFromTestCase(TestBasicCalculations))
    suite.addTests(loader.loadTestsFromTestCase(TestDualityRelation))
    suite.addTests(loader.loadTestsFromTestCase(TestEdgeCases))
    suite.addTests(loader.loadTestsFromTestCase(TestClassifications))
    suite.addTests(loader.loadTestsFromTestCase(TestCIDecomposition))
    suite.addTests(loader.loadTestsFromTestCase(TestJSONSerialization))
    suite.addTests(loader.loadTestsFromTestCase(TestRegressionGoldenDataset))
    suite.addTests(loader.loadTestsFromTestCase(TestConvenienceFunction))

    # Dodaj property-based tests jeśli Hypothesis dostępne
    if HYPOTHESIS_AVAILABLE:
        suite.addTests(loader.loadTestsFromTestCase(TestPropertyBased))
        print("✓ Hypothesis property-based tests włączone")
    else:
        print("⚠️  Hypothesis property-based tests pominięte (brak biblioteki)")

    # Uruchom testy
    runner = unittest.TextTestRunner(verbosity=verbosity)
    result = runner.run(suite)

    return result


if __name__ == "__main__":
    print("=" * 80)
    print("GTMØ Constitutional Duality Calculator - Comprehensive Test Suite")
    print("=" * 80)
    print()

    # Uruchom testy
    result = run_tests(verbosity=2)

    # Podsumowanie
    print()
    print("=" * 80)
    if result.wasSuccessful():
        print("✓ Wszystkie testy przeszły pomyślnie!")
        sys.exit(0)
    else:
        print("✗ Niektóre testy nie powiodły się.")
        print(f"  Failures: {len(result.failures)}")
        print(f"  Errors: {len(result.errors)}")
        sys.exit(1)
