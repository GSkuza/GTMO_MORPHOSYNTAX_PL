#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
GTMØ Adelic Analysis Runner
============================
Uruchamia pełną analizę GTMØ z warstwą adeliczną i zapisuje wyniki do JSON.

Użycie:
    python -X utf8 run_adelic_analysis.py
    python -X utf8 run_adelic_analysis.py "Twój tekst do analizy"
    python -X utf8 run_adelic_analysis.py --file tekst.txt
"""

import sys
import json
import argparse
from pathlib import Path

# Import głównego silnika GTMØ
# Note: gtmo_morphosyntax.py handles encoding itself
from gtmo_morphosyntax import QuantumMorphosyntaxEngine, EnhancedGTMOProcessor
import numpy as np

# Import Truth Observer (τ component)
from gtmo_truth_observer import LLMTruthObserver, combine_gtmo_and_truth, ClaimType


def convert_numpy_types(obj):
    """Recursively convert numpy types to native Python types for JSON serialization."""
    if isinstance(obj, dict):
        return {key: convert_numpy_types(value) for key, value in obj.items()}
    elif isinstance(obj, list):
        return [convert_numpy_types(item) for item in obj]
    elif isinstance(obj, np.integer):
        return int(obj)
    elif isinstance(obj, np.floating):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    else:
        return obj

def analyze_text_with_adelic(
    text: str,
    save_to_file: bool = True,
    output_file: str = None,
    verify_truth: bool = True,
    truth_provider: str = "mock"
) -> dict:
    """
    Analizuje tekst z warstwą adeliczną i opcjonalną weryfikacją prawdziwości (τ).

    Args:
        text: Tekst do analizy
        save_to_file: Czy zapisać wyniki do pliku JSON
        output_file: Nazwa pliku wyjściowego (domyślnie: adelic_result.json)
        verify_truth: Czy wykonać weryfikację prawdziwości (τ)
        truth_provider: Dostawca LLM dla τ ("mock", "anthropic", "openai", "auto")

    Returns:
        Słownik z pełnymi wynikami analizy (GTMO + adelic + τ)
    """
    print("=" * 80)
    print("  GTMØ ADELIC ANALYSIS")
    print("=" * 80)
    print(f"\n📝 Text: {text}")
    print(f"   Length: {len(text)} characters, {len(text.split())} words\n")

    # Inicjalizacja silnika
    print("🔧 Initializing GTMØ Engine...")
    engine = QuantumMorphosyntaxEngine()

    if engine.adelic_layer is None:
        print("❌ ERROR: Adelic layer not available!")
        print("   Make sure gtmo_adelic_layer.py and gtmo_adelic_metrics.py are present.")
        return None

    print(f"✅ Engine initialized with {len(engine.adelic_layer.observers)} observers")
    print(f"   Epsilon: {engine.adelic_layer.epsilon}")
    print(f"   Observers: {[obs.id for obs in engine.adelic_layer.observers[:5]]}...\n")

    # Wykonaj analizę GTMØ
    print("🌟 Starting GTMØ quantum analysis...")
    base_result = engine.gtmo_analyze_quantum(text, source_file='adelic_analysis')

    # Ekstrahuj bazowe współrzędne
    import numpy as np
    base_coords = np.array([
        base_result['coordinates']['determination'],
        base_result['coordinates']['stability'],
        base_result['coordinates']['entropy']
    ])

    # Przygotuj atraktor kontekstowy
    context_attractor = np.array([0.85, 0.85, 0.15])  # Ψᴷ - Knowledge/Certainty

    # Wykonaj analizę adeliczną
    print("🌟 Starting adelic layer analysis...\n")
    adelic_result = engine.adelic_layer.analyze_with_observers(
        text=text,
        base_coords=base_coords,
        observers=None,  # Użyje domyślnych obserwatorów
        context_attractor=context_attractor,
        context_name='Ψᴷ',
        metric='phi9'
    )

    # Połącz wyniki
    result = base_result
    result['adelic'] = adelic_result

    # === TRUTH VERIFICATION (τ) ===
    if verify_truth:
        print("🔍 Starting truth verification (τ)...")
        truth_observer = LLMTruthObserver(provider=truth_provider)
        truth_verdict = truth_observer.verify(text)

        # Combine GTMO + truth
        result = combine_gtmo_and_truth(result, truth_verdict)

        print(f"   Provider: {truth_observer.provider}")
        print(f"   Claim type: {truth_verdict.claim_type.value}")
        if truth_verdict.truth_value is not None:
            print(f"   τ (truth value): {truth_verdict.truth_value:.2f}")
        else:
            print(f"   τ (truth value): N/A (not verifiable)")
        print(f"   Combined verdict: {result['combined_verdict']['verdict']}\n")

    # Wyświetl kluczowe wyniki
    print("\n" + "=" * 80)
    print("  RESULTS")
    print("=" * 80)

    # GTMØ coordinates
    coords = result['coordinates']
    print(f"\n📊 GTMØ Coordinates:")
    print(f"   D (Determination): {coords['determination']:.4f}")
    print(f"   S (Stability):     {coords['stability']:.4f}")
    print(f"   E (Entropy):       {coords['entropy']:.4f}")

    # Adelic results
    if 'adelic' in result:
        adelic = result['adelic']
        print(f"\n⚛️  ADELIC LAYER:")
        print(f"   Emerged: {adelic['emerged']}")
        print(f"   Status: {adelic['status']}")
        print(f"   Synchronization Energy: {adelic['synchronization_energy']:.4f}")
        print(f"   Number of Observers: {adelic['n_observers']}")

        if adelic['emerged']:
            print(f"\n   ✨ Global Value φ_∞:")
            global_val = adelic['global_value']
            print(f"      D = {global_val[0]:.4f}")
            print(f"      S = {global_val[1]:.4f}")
            print(f"      E = {global_val[2]:.4f}")
        else:
            print(f"\n   💥 No emergence (semantic divergence)")

            if 'diagnosis' in adelic:
                diag = adelic['diagnosis']
                print(f"   📋 Diagnosis:")
                print(f"      Reason: {diag.get('reason', 'unknown')}")
                print(f"      Max distance: {diag.get('max_distance', 0):.4f}")
                print(f"      Exceeds ε by: {diag.get('exceeds_by', 0):.4f}")

        # Local interpretations
        print(f"\n   👁️  Local Interpretations:")
        for obs_id, local_data in list(adelic['local_values'].items())[:5]:
            local_val = local_data['local_value']
            is_std = local_data['is_standard']
            alien_mag = local_data['alienation_magnitude']

            status = "✓ standard" if is_std else f"⚠ alienated ({alien_mag:.3f})"
            print(f"      {obs_id}: [{local_val[0]:.3f}, {local_val[1]:.3f}, {local_val[2]:.3f}] {status}")

        if len(adelic['local_values']) > 5:
            print(f"      ... and {len(adelic['local_values']) - 5} more observers")

    # Quantum metrics
    if 'quantum_metrics' in result:
        qm = result['quantum_metrics']
        print(f"\n⚛️  QUANTUM METRICS:")
        print(f"   Coherence: {qm['total_coherence']:.4f}")
        print(f"   Quantum words: {qm['quantum_words']}")
        print(f"   Entanglements: {qm['entanglements']}")

    # Rhetorical analysis
    if 'rhetorical_analysis' in result:
        rhet = result['rhetorical_analysis']
        print(f"\n🎭 RHETORICAL MODE: {rhet['mode']}")
        if rhet['mode'] != 'literal':
            print(f"   Irony score: {rhet['irony_score']:.4f}")
            print(f"   Paradox score: {rhet['paradox_score']:.4f}")

    # Truth verification results
    if 'truth' in result:
        truth = result['truth']
        combined = result.get('combined_verdict', {})

        print(f"\n🔍 TRUTH VERIFICATION (τ):")
        print(f"   Claim type: {truth['claim_type']}")
        if truth['truth_value'] is not None:
            print(f"   τ (truth value): {truth['truth_value']:.2f}")
        else:
            print(f"   τ (truth value): N/A")
        print(f"   Confidence: {truth['confidence']:.2f}")
        print(f"   Reasoning: {truth['reasoning']}")

        if truth.get('flags'):
            flags_str = ", ".join(f"{k}" for k, v in truth['flags'].items() if v)
            if flags_str:
                print(f"   ⚠️  Flags: {flags_str}")

        print(f"\n📊 COMBINED VERDICT: {combined.get('verdict', 'N/A')}")
        print(f"   Confidence: {combined.get('confidence', 0):.2f}")

    # Save to file
    if save_to_file:
        if output_file is None:
            output_file = "adelic_result.json"

        # Convert numpy types to native Python types
        result_serializable = convert_numpy_types(result)

        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(result_serializable, f, indent=2, ensure_ascii=False)

        print(f"\n💾 Results saved to: {output_file}")
        print(f"   File size: {Path(output_file).stat().st_size} bytes")

    print("\n" + "=" * 80)
    print("✅ Analysis complete!")
    print("=" * 80 + "\n")

    return result


def main():
    """Główna funkcja programu."""
    parser = argparse.ArgumentParser(
        description='GTMØ Adelic Analysis - Analyze text with p-adic semantic emergence',
        formatter_class=argparse.RawDescriptionHelpFormatter
    )

    parser.add_argument(
        'text',
        nargs='?',
        default=None,
        help='Text to analyze (if not provided, uses example texts)'
    )

    parser.add_argument(
        '--file', '-f',
        type=str,
        help='Read text from file'
    )

    parser.add_argument(
        '--output', '-o',
        type=str,
        default='adelic_result.json',
        help='Output JSON file (default: adelic_result.json)'
    )

    parser.add_argument(
        '--no-save',
        action='store_true',
        help='Do not save results to file (only print to console)'
    )

    parser.add_argument(
        '--no-truth',
        action='store_true',
        help='Skip truth verification (τ component)'
    )

    parser.add_argument(
        '--truth-provider',
        type=str,
        default='mock',
        choices=['mock', 'anthropic', 'openai', 'auto'],
        help='LLM provider for truth verification (default: mock)'
    )

    args = parser.parse_args()

    # Determine text to analyze
    if args.file:
        # Read from file
        try:
            with open(args.file, 'r', encoding='utf-8') as f:
                text = f.read().strip()
            print(f"📄 Loaded text from: {args.file}")
        except Exception as e:
            print(f"❌ Error reading file: {e}")
            sys.exit(1)
    elif args.text:
        # Use provided text
        text = args.text
    else:
        # Use example texts
        example_texts = [
            "Rzeczpospolita Polska przestrzega wiążącego ją prawa międzynarodowego.",
            "Świetny pomysł!",
            "To zdanie nie istnieje.",
        ]

        print("No text provided. Analyzing example texts:\n")

        for i, example in enumerate(example_texts, 1):
            print(f"\n{'='*80}")
            print(f"EXAMPLE {i}/{len(example_texts)}")
            print('='*80)

            output_file = f"adelic_example_{i}.json" if not args.no_save else None
            analyze_text_with_adelic(
                text=example,
                save_to_file=not args.no_save,
                output_file=output_file,
                verify_truth=not args.no_truth,
                truth_provider=args.truth_provider
            )

        return

    # Analyze single text
    analyze_text_with_adelic(
        text=text,
        save_to_file=not args.no_save,
        output_file=args.output,
        verify_truth=not args.no_truth,
        truth_provider=args.truth_provider
    )


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n⚠️  Analysis interrupted by user")
        sys.exit(0)
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
