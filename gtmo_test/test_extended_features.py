#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Test Extended Features Integration
===================================
Test wszystkich nowych funkcjonalności z gtmo_extended.py:
- Analiza temporalna
- Wykrywanie ironii/sarkazmu
- Analiza kwantowa (superpozycja stanów)
- Wykrywanie paradoksów
- Kalkulator Depth
- Pełna dekompozycja CI
"""

import numpy as np

print("=" * 80)
print("GTMØ EXTENDED FEATURES TEST")
print("=" * 80)

# Test 1: Temporal Coords
print("\n1️⃣ Test TEMPORAL_COORDS")
print("-" * 40)
try:
    from gtmo_morphosyntax import TEMPORAL_COORDS
    print(f"✓ TEMPORAL_COORDS loaded: {len(TEMPORAL_COORDS)} temporal states")
    for time_type, coords in list(TEMPORAL_COORDS.items())[:3]:
        print(f"  - {time_type}: D={coords[0]:.2f}, S={coords[1]:.2f}, E={coords[2]:.2f}")
except Exception as e:
    print(f"✗ Error: {e}")

# Test 2: Rhetorical Patterns
print("\n2️⃣ Test RHETORICAL_PATTERNS")
print("-" * 40)
try:
    from gtmo_morphosyntax import RHETORICAL_PATTERNS
    print(f"✓ RHETORICAL_PATTERNS loaded:")
    print(f"  - Irony markers: {len(RHETORICAL_PATTERNS['irony_markers'])}")
    print(f"  - Paradox markers: {len(RHETORICAL_PATTERNS['paradox_markers'])}")
    print(f"  - Sarcasm patterns: {len(RHETORICAL_PATTERNS['sarcasm_patterns'])}")
    print(f"  Examples: {RHETORICAL_PATTERNS['irony_markers'][:3]}")
except Exception as e:
    print(f"✗ Error: {e}")

# Test 3: Temporal Analysis Function
print("\n3️⃣ Test analyze_temporality()")
print("-" * 40)
try:
    from gtmo_morphosyntax import analyze_temporality

    # Test without spaCy (fallback)
    coords, meta = analyze_temporality("Test text", doc=None)
    print(f"✓ analyze_temporality() callable")
    print(f"  Coords: D={coords[0]:.2f}, S={coords[1]:.2f}, E={coords[2]:.2f}")
    print(f"  Metadata: {meta}")
except Exception as e:
    print(f"✗ Error: {e}")

# Test 4: Enhanced Rhetorical Detection
print("\n4️⃣ Test detect_enhanced_rhetorical_mode()")
print("-" * 40)
try:
    from gtmo_morphosyntax import detect_enhanced_rhetorical_mode

    base_coords = np.array([0.7, 0.6, 0.3])
    morph_meta = {'pos': {'adj': 5, 'verb': 3}}

    # Test normal text
    coords, mode, meta = detect_enhanced_rhetorical_mode(
        "Dzień był piękny.", base_coords, morph_meta
    )
    print(f"✓ Normal text: mode={mode}, score={meta.get('irony_score', 0)}")

    # Test ironic text
    coords_irony, mode_irony, meta_irony = detect_enhanced_rhetorical_mode(
        "Świetnie, znowu pada deszcz...", base_coords, morph_meta
    )
    print(f"✓ Ironic text: mode={mode_irony}, irony_score={meta_irony.get('irony_score', 0):.2f}")

except Exception as e:
    print(f"✗ Error: {e}")

# Test 5: Quantum Ambiguity Analyzer
print("\n5️⃣ Test QuantumAmbiguityAnalyzer")
print("-" * 40)
try:
    from gtmo_morphosyntax import QuantumAmbiguityAnalyzer

    qa = QuantumAmbiguityAnalyzer()

    # Test ambiguous text
    needs_quantum, meta = qa.detect_ambiguity("Może przyjdę, a może nie...")
    print(f"✓ Ambiguous text detected: {needs_quantum}")
    print(f"  Ambiguity score: {meta.get('ambiguity_score', 0):.2f}")
    print(f"  Markers: {meta.get('ambiguity_markers', [])}")

    # Test normal text
    needs_quantum2, meta2 = qa.detect_ambiguity("Jutro jest niedziela.")
    print(f"✓ Normal text: {needs_quantum2} (score={meta2.get('ambiguity_score', 0):.2f})")

except Exception as e:
    print(f"✗ Error: {e}")

# Test 6: Generate Alternative Interpretations
print("\n6️⃣ Test generate_alternative_interpretations()")
print("-" * 40)
try:
    from gtmo_morphosyntax import generate_alternative_interpretations

    base_coords = np.array([0.7, 0.6, 0.3])

    # Test with question
    result = generate_alternative_interpretations("Czy to prawda?", base_coords)
    print(f"✓ Question text: {len(result['states'])} alternative states")
    print(f"  Probabilities: {result['probabilities']}")

    # Test with conditional
    result2 = generate_alternative_interpretations("Może będzie lepiej", base_coords)
    print(f"✓ Conditional text: {len(result2['states'])} alternative states")

except Exception as e:
    print(f"✗ Error: {e}")

# Test 7: Superposition Creation
print("\n7️⃣ Test QuantumAmbiguityAnalyzer.create_superposition()")
print("-" * 40)
try:
    from gtmo_morphosyntax import QuantumAmbiguityAnalyzer
    import numpy as np

    qa = QuantumAmbiguityAnalyzer()

    interpretations = [
        np.array([0.7, 0.6, 0.3]),
        np.array([0.5, 0.5, 0.5]),
        np.array([0.4, 0.3, 0.7])
    ]
    probs = [0.5, 0.3, 0.2]

    state = qa.create_superposition(interpretations, probs)
    print(f"✓ Superposition created:")
    print(f"  Base state: [{state['base_state'][0]:.3f}, {state['base_state'][1]:.3f}, {state['base_state'][2]:.3f}]")
    print(f"  Is superposed: {state['superposition']}")
    print(f"  Von Neumann entropy: {state['von_neumann_entropy']:.4f}")
    print(f"  Uncertainty: {state['uncertainty']:.4f}")

except Exception as e:
    print(f"✗ Error: {e}")

# Test 8: Constitutional Duality Calculator
print("\n8️⃣ Test ConstitutionalDualityCalculator")
print("-" * 40)
try:
    from gtmo_constitutional_duality import ConstitutionalDualityCalculator

    calc = ConstitutionalDualityCalculator()
    metrics = calc.calculate_metrics(
        ambiguity=2.0,
        depth=5,
        D=0.7,
        S=0.6,
        E=0.4
    )

    print(f"✓ Constitutional metrics calculated:")
    print(f"  CD = {metrics.CD:.2f}")
    print(f"  CI = {metrics.CI:.2f}")
    print(f"  SA = {metrics.SA:.2%}")
    print(f"  Duality: CI × CD = {metrics.duality_product:.2f} ≈ Depth² = {metrics.duality_theoretical}")
    print(f"  Verification: {'PASSED' if metrics.duality_verified else 'FAILED'}")
    print(f"  CI Decomposition:")
    print(f"    - Morphological: {metrics.CI_morphological:.2f}")
    print(f"    - Syntactic: {metrics.CI_syntactic:.2f}")
    print(f"    - Semantic: {metrics.CI_semantic:.2f}")

except Exception as e:
    print(f"✗ Error: {e}")

# Test 9: Full Integration (without morfeusz/spaCy)
print("\n9️⃣ Test Full Integration (QuantumMorphosyntaxEngine)")
print("-" * 40)
try:
    from gtmo_morphosyntax import QuantumMorphosyntaxEngine

    engine = QuantumMorphosyntaxEngine()
    print(f"✓ QuantumMorphosyntaxEngine initialized")
    print(f"  Has rhetorical_analyzer: {engine.rhetorical_analyzer is not None}")
    print(f"  Has constitutional_calculator: {engine.constitutional_calculator is not None}")
    print(f"  Has quantum_ambiguity_analyzer: {engine.quantum_ambiguity_analyzer is not None}")

except Exception as e:
    print(f"✗ Error: {e}")

# Test 10: Semantic Contradiction Detection
print("\n🔟 Test has_semantic_contradiction()")
print("-" * 40)
try:
    from gtmo_morphosyntax import has_semantic_contradiction

    # Test with contradiction
    has_contr1 = has_semantic_contradiction("Kocham cię i jednocześnie nienawidzę")
    print(f"✓ Contradiction detected: {has_contr1}")

    # Test without contradiction
    has_contr2 = has_semantic_contradiction("Dzień był piękny i słoneczny")
    print(f"✓ No contradiction: {not has_contr2}")

except Exception as e:
    print(f"✗ Error: {e}")

print("\n" + "=" * 80)
print("✅ ALL TESTS COMPLETED!")
print("=" * 80)
