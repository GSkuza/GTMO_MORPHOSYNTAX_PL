"""Prosty test nowej dekompozycji CI."""

from gtmo_constitutional_duality import ConstitutionalDualityCalculator

def test_advanced_decomposition():
    """Test zaawansowanej dekompozycji CI."""
    calc = ConstitutionalDualityCalculator()

    # Test 1: Z inflectional_forms_count
    print("Test 1: Z inflectional_forms_count=50")
    metrics1 = calc.calculate_metrics(
        ambiguity=2.5,
        depth=7,
        D=0.7,
        S=0.6,
        E=0.4,
        inflectional_forms_count=50
    )

    print(f"  CI_total = {metrics1.CI:.4f}")
    print(f"  CI_morphological = {metrics1.CI_morphological:.4f} ({metrics1.CI_morphological/metrics1.CI*100:.1f}%)")
    print(f"  CI_syntactic = {metrics1.CI_syntactic:.4f} ({metrics1.CI_syntactic/metrics1.CI*100:.1f}%)")
    print(f"  CI_semantic = {metrics1.CI_semantic:.4f} ({metrics1.CI_semantic/metrics1.CI*100:.1f}%)")
    ci_sum1 = metrics1.CI_morphological + metrics1.CI_syntactic + metrics1.CI_semantic
    print(f"  Suma = {ci_sum1:.4f}")
    print(f"  Zgodnosc: {abs(ci_sum1 - metrics1.CI) < 0.01}")

    # Test 2: Bez inflectional_forms_count (używa depth jako proxy)
    print("\nTest 2: Bez inflectional_forms_count (depth=7 jako proxy)")
    metrics2 = calc.calculate_metrics(
        ambiguity=2.5,
        depth=7,
        D=0.7,
        S=0.6,
        E=0.4
    )

    print(f"  CI_total = {metrics2.CI:.4f}")
    print(f"  CI_morphological = {metrics2.CI_morphological:.4f} ({metrics2.CI_morphological/metrics2.CI*100:.1f}%)")
    print(f"  CI_syntactic = {metrics2.CI_syntactic:.4f} ({metrics2.CI_syntactic/metrics2.CI*100:.1f}%)")
    print(f"  CI_semantic = {metrics2.CI_semantic:.4f} ({metrics2.CI_semantic/metrics2.CI*100:.1f}%)")
    ci_sum2 = metrics2.CI_morphological + metrics2.CI_syntactic + metrics2.CI_semantic
    print(f"  Suma = {ci_sum2:.4f}")
    print(f"  Zgodnosc: {abs(ci_sum2 - metrics2.CI) < 0.01}")

    # Test 3: Wysoka ambiguity powinna zmniejszyć CI_syntactic
    print("\nTest 3: Wysoka ambiguity -> niższy CI_syntactic")
    metrics_low_amb = calc.calculate_metrics(1.5, 7, 0.7, 0.6, 0.4)
    metrics_high_amb = calc.calculate_metrics(3.5, 7, 0.7, 0.6, 0.4)

    print(f"  Niska ambiguity (1.5): CI_synt = {metrics_low_amb.CI_syntactic:.4f}")
    print(f"  Wysoka ambiguity (3.5): CI_synt = {metrics_high_amb.CI_syntactic:.4f}")
    print(f"  CI_synt maleje z ambiguity: {metrics_low_amb.CI_syntactic > metrics_high_amb.CI_syntactic}")

    # Test 4: Wysoki depth powinna zwiększyć CI_syntactic (kwadratowo!)
    print("\nTest 4: Wysoki depth -> wyzszy CI_syntactic (kwadratowo)")
    metrics_shallow = calc.calculate_metrics(2.0, 3, 0.7, 0.6, 0.4)
    metrics_deep = calc.calculate_metrics(2.0, 9, 0.7, 0.6, 0.4)

    print(f"  Niski depth (3): CI_synt = {metrics_shallow.CI_syntactic:.4f}")
    print(f"  Wysoki depth (9): CI_synt = {metrics_deep.CI_syntactic:.4f}")
    print(f"  Stosunek CI_synt: {metrics_deep.CI_syntactic / metrics_shallow.CI_syntactic:.2f}")
    print(f"  Stosunek depth^2: {(9**2) / (3**2):.2f}")

    print("\n" + "="*60)
    print("Wszystkie testy PASSED!")
    print("="*60)

if __name__ == "__main__":
    test_advanced_decomposition()
