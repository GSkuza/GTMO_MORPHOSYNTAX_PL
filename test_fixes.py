#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Test script for corrected GTMØ morphology implementation"""

from gtmo_morphosyntax import analyze_quantum_with_axioms

def test_simple_sentence():
    """Test 1: Simple sentence - should show high D, high S, low CI_morph"""
    text = 'Rzeczy mają znaczenie.'
    print('='*70)
    print('TEST 1: Proste zdanie (rzeczy mają znaczenie)')
    print('='*70)

    result = analyze_quantum_with_axioms(text)

    # Handle different possible result structures
    if 'gtmo_coordinates' in result:
        coords = result['gtmo_coordinates']
    elif 'coordinates' in result:
        coords = result['coordinates']
    else:
        coords = {'determination': result.get('D', 0.5), 'stability': result.get('S', 0.5), 'entropy': result.get('E', 0.5)}

    morph = result.get('morphology', {})
    ci_decomp = result.get('constitutional_metrics', {}).get('indefiniteness', {}).get('decomposition', {})

    print(f"\nGTMØ Coordinates:")
    print(f"  D (Determination): {coords['determination']:.3f}")
    print(f"  S (Stability):     {coords['stability']:.3f}")
    print(f"  E (Entropy):       {coords['entropy']:.3f}")

    # Check D-S correlation
    if coords['determination'] > 0.7:
        if coords['stability'] >= coords['determination'] - 0.15:
            print(f"  ✓ D-S correlation OK: High D ({coords['determination']:.3f}) → High S ({coords['stability']:.3f})")
        else:
            print(f"  ✗ D-S correlation FAIL: High D ({coords['determination']:.3f}) but low S ({coords['stability']:.3f})")

    print(f"\nMorphology Analysis:")
    print(f"  Disambiguations: {morph.get('disambiguations', 0)}")
    print(f"  Ambiguity ratio: {morph.get('ambiguity_ratio', 0):.2f} (before: would be counted as separate morphemes)")
    if 'disambiguation_log' in morph:
        print(f"  Examples: {morph['disambiguation_log'][:3]}")

    print(f"\nConstitutional Indefiniteness (CI) Decomposition:")
    ci_morph = ci_decomp['morphological']
    ci_synt = ci_decomp['syntactic']
    ci_sem = ci_decomp['semantic']

    print(f"  CI_morphological: {ci_morph['value']:.4f} ({ci_morph['percentage']:.1f}%)")
    print(f"  CI_syntactic:     {ci_synt['value']:.4f} ({ci_synt['percentage']:.1f}%)")
    print(f"  CI_semantic:      {ci_sem['value']:.4f} ({ci_sem['percentage']:.1f}%)")

    # Verify morphology is minimal
    if ci_morph['percentage'] <= 20:
        print(f"  ✓ Morphology contribution OK: {ci_morph['percentage']:.1f}% ≤ 20%")
    else:
        print(f"  ✗ Morphology contribution TOO HIGH: {ci_morph['percentage']:.1f}% > 20%")

    # Verify semantics dominates
    if ci_sem['percentage'] >= 50:
        print(f"  ✓ Semantic dominance OK: {ci_sem['percentage']:.1f}% ≥ 50%")
    else:
        print(f"  ✗ Semantic dominance FAIL: {ci_sem['percentage']:.1f}% < 50%")

    print(f"\nQuantum Properties:")
    print(f"  Coherence: {result.get('quantum_coherence', 0):.3f} (1.0 = coherent, 0.0 = decoherent)")
    print(f"  Classification: {result.get('quantum_superposition_type', 'N/A')}")

    return result

def test_complex_sentence():
    """Test 2: Complex ambiguous sentence"""
    text = 'Może kiedyś zrozumiemy, czy prawda istnieje obiektywnie, czy jest konstrukcją społeczną.'
    print('\n' + '='*70)
    print('TEST 2: Złożone wieloznaczne zdanie')
    print('='*70)

    result = analyze_quantum_with_axioms(text)

    # Handle different possible result structures
    if 'gtmo_coordinates' in result:
        coords = result['gtmo_coordinates']
    elif 'coordinates' in result:
        coords = result['coordinates']
    else:
        coords = {'determination': result.get('D', 0.5), 'stability': result.get('S', 0.5), 'entropy': result.get('E', 0.5)}

    ci_decomp = result.get('constitutional_metrics', {}).get('indefiniteness', {}).get('decomposition', {})

    print(f"\nGTMØ Coordinates:")
    print(f"  D (Determination): {coords['determination']:.3f}")
    print(f"  S (Stability):     {coords['stability']:.3f}")
    print(f"  E (Entropy):       {coords['entropy']:.3f}")

    print(f"\nCI Decomposition:")
    print(f"  Morphological: {ci_decomp['morphological']['percentage']:.1f}%")
    print(f"  Syntactic:     {ci_decomp['syntactic']['percentage']:.1f}%")
    print(f"  Semantic:      {ci_decomp['semantic']['percentage']:.1f}%")

    # For complex sentence, expect higher semantic contribution
    if ci_decomp['semantic']['percentage'] > ci_decomp['morphological']['percentage']:
        print(f"  ✓ Semantic > Morphological in complex sentence")

    return result

if __name__ == '__main__':
    print("\n🔧 TESTING CORRECTED GTMØ MORPHOLOGY IMPLEMENTATION")
    print("="*70)
    print("\nKey Fixes:")
    print("1. Disambiguation: Select ONE Morfeusz2 interpretation (not aggregate all)")
    print("2. Composition: Tensor product (geometric), not arithmetic mean")
    print("3. D-S Correlation: High determination → high stability")
    print("4. CI Decomposition: Semantic 50%+, Morphology max 20%")
    print("5. Phase Coherence: 1.0 = coherent, 0.0 = decoherent")
    print()

    try:
        test_simple_sentence()
        test_complex_sentence()

        print("\n" + "="*70)
        print("✓ ALL TESTS COMPLETED")
        print("="*70)

    except Exception as e:
        print(f"\n✗ TEST FAILED: {e}")
        import traceback
        traceback.print_exc()
