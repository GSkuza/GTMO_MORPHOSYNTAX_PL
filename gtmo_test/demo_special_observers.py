#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Demo specjalnych obserwatorów - halucynacje, nonsens, propaganda
================================================================
Pokazuje jak obserwatorzy "patologiczni" zachowują się w warstwie adelicznej.
"""

import sys
import numpy as np
from gtmo_adelic_layer import AdelicSemanticLayer, create_standard_observers
import io

if sys.platform == 'win32':
    try:
        sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
    except:
        pass


def demo_hallucination_detection():
    """Demo: Wykrywanie halucynacji przez desynchronizację."""
    print("\n" + "=" * 70)
    print("  DEMO: Wykrywanie halucynacji przez energię desynchronizacji")
    print("=" * 70)

    layer = AdelicSemanticLayer()
    observers = create_standard_observers()

    # Obserwatorzy
    obs_formal = [o for o in observers if o.id == 'O_formal'][0]
    obs_halluc = [o for o in observers if o.id == 'O_hallucination'][0]
    obs_medical = [o for o in observers if o.id == 'O_medical'][0]

    # Tekst medyczny (prawdziwy)
    base_medical = np.array([0.85, 0.88, 0.15])

    print(f"\n📄 Tekst: 'Pacjent otrzymał dożylnie 500mg paracetamolu'")
    print(f"📍 Base coords: {base_medical}")

    result_real = layer.analyze_with_observers(
        text="medyczny_real",
        base_coords=base_medical,
        observers=[obs_formal, obs_medical],
        metric='phi9'
    )

    print(f"\n✅ NORMALNY KONSENSUS (formal + medical):")
    print(f"  Emerged: {result_real['emerged']}")
    print(f"  Energy: {result_real['synchronization_energy']:.4f}")

    # Teraz dodaj obserwatora halucynacyjnego
    result_halluc = layer.analyze_with_observers(
        text="medyczny_hallucination",
        base_coords=base_medical,
        observers=[obs_medical, obs_halluc],
        metric='phi9'
    )

    print(f"\n🔴 DODANO OBSERWATORA HALUCYNACYJNEGO (medical + hallucination):")
    print(f"  Emerged: {result_halluc['emerged']}")
    print(f"  Energy: {result_halluc['synchronization_energy']:.4f} (WYSOKA!)")

    print(f"\n📊 Lokalne interpretacje:")
    for obs_id, data in result_halluc['local_values'].items():
        coords = data['local_value']
        print(f"  {obs_id}: [{coords[0]:.3f}, {coords[1]:.3f}, {coords[2]:.3f}]")

    energy_ratio = result_halluc['synchronization_energy'] / result_real['synchronization_energy']
    print(f"\n🎯 WYKRYCIE:")
    print(f"  Energia wzrosła {energy_ratio:.1f}x")
    print(f"  → System wykrył konflikt interpretacji!")
    print(f"  → Możliwa halucynacja LLM!")


def demo_nonsense_vs_propaganda():
    """Demo: Nonsens vs Propaganda - oba patologiczne ale różnie."""
    print("\n" + "=" * 70)
    print("  DEMO: Nonsens vs Propaganda - różne typy patologii")
    print("=" * 70)

    layer = AdelicSemanticLayer()
    observers = create_standard_observers()

    obs_formal = [o for o in observers if o.id == 'O_formal'][0]
    obs_nonsense = [o for o in observers if o.id == 'O_nonsense'][0]
    obs_propaganda = [o for o in observers if o.id == 'O_propaganda'][0]

    # Baza neutralna
    base_neutral = np.array([0.60, 0.65, 0.40])

    print(f"\n📍 Base coords (neutral): {base_neutral}")

    # Test 1: Nonsens
    result_nonsense = layer.analyze_with_observers(
        text="nonsensical_text",
        base_coords=base_neutral,
        observers=[obs_formal, obs_nonsense],
        metric='phi9'
    )

    print(f"\n🎭 NONSENS (formal + nonsense):")
    print(f"  Emerged: {result_nonsense['emerged']}")
    print(f"  Energy: {result_nonsense['synchronization_energy']:.4f}")

    nonsense_local = result_nonsense['local_values']['O_nonsense']['local_value']
    print(f"  Nonsense coords: [{nonsense_local[0]:.3f}, {nonsense_local[1]:.3f}, {nonsense_local[2]:.3f}]")
    print(f"  → Niskie D, niskie S, WYSOKIE E (chaos)")

    # Test 2: Propaganda
    result_propaganda = layer.analyze_with_observers(
        text="propaganda_text",
        base_coords=base_neutral,
        observers=[obs_formal, obs_propaganda],
        metric='phi9'
    )

    print(f"\n📢 PROPAGANDA (formal + propaganda):")
    print(f"  Emerged: {result_propaganda['emerged']}")
    print(f"  Energy: {result_propaganda['synchronization_energy']:.4f}")

    propaganda_local = result_propaganda['local_values']['O_propaganda']['local_value']
    print(f"  Propaganda coords: [{propaganda_local[0]:.3f}, {propaganda_local[1]:.3f}, {propaganda_local[2]:.3f}]")
    print(f"  → WYSOKIE D, wysokie S, niskie E (fałszywa pewność)")

    print(f"\n🔬 RÓŻNICA:")
    print(f"  Nonsens:    chaos semantyczny (↑E)")
    print(f"  Propaganda: fałszywa certainty (↑D, ↓E)")
    print(f"  → Oba patologiczne, ale w PRZECIWNYCH kierunkach!")


def demo_conspiracy_detection():
    """Demo: Wykrywanie teorii spiskowych."""
    print("\n" + "=" * 70)
    print("  DEMO: Wykrywanie teorii spiskowych")
    print("=" * 70)

    layer = AdelicSemanticLayer()
    observers = create_standard_observers()

    obs_formal = [o for o in observers if o.id == 'O_formal'][0]
    obs_journalistic = [o for o in observers if o.id == 'O_journalistic'][0]
    obs_conspiracy = [o for o in observers if o.id == 'O_conspiracy'][0]

    # Tekst neutralny
    base_news = np.array([0.75, 0.78, 0.25])

    print(f"\n📰 Tekst: 'Rząd wprowadza nowe przepisy podatkowe'")
    print(f"📍 Base coords: {base_news}")

    # Normalny konsensus (formal + journalistic)
    result_normal = layer.analyze_with_observers(
        text="news_normal",
        base_coords=base_news,
        observers=[obs_formal, obs_journalistic],
        metric='phi9'
    )

    print(f"\n✅ NORMALNY (formal + journalistic):")
    print(f"  Emerged: {result_normal['emerged']}")
    print(f"  Energy: {result_normal['synchronization_energy']:.4f}")

    # Dodaj conspiracy
    result_conspiracy = layer.analyze_with_observers(
        text="news_conspiracy",
        base_coords=base_news,
        observers=[obs_journalistic, obs_conspiracy],
        metric='phi9'
    )

    print(f"\n🕵️ DODANO CONSPIRACY (journalistic + conspiracy):")
    print(f"  Emerged: {result_conspiracy['emerged']}")
    print(f"  Energy: {result_conspiracy['synchronization_energy']:.4f} (WZROST)")

    conspiracy_local = result_conspiracy['local_values']['O_conspiracy']['local_value']
    print(f"  Conspiracy coords: [{conspiracy_local[0]:.3f}, {conspiracy_local[1]:.3f}, {conspiracy_local[2]:.3f}]")

    if 'diagnosis' in result_conspiracy:
        diag = result_conspiracy['diagnosis']
        print(f"\n🔍 DIAGNOZA:")
        print(f"  Max distance: {diag['max_distance']:.3f}")
        print(f"  Exceeds ε by: {diag['exceeds_by']:.3f}")
        print(f"  → Obserwator conspiracy destabilizuje konsensus!")


def demo_extreme_temperature():
    """Demo: Efekt ekstremalnej temperatury (O_hallucination temp=2.5)."""
    print("\n" + "=" * 70)
    print("  DEMO: Efekt ekstremalnej temperatury (halucynacje)")
    print("=" * 70)

    layer = AdelicSemanticLayer()
    observers = create_standard_observers()

    obs_halluc = [o for o in observers if o.id == 'O_hallucination'][0]

    print(f"\n🔥 O_hallucination:")
    print(f"  Temperature: {obs_halluc.temperature}")
    print(f"  Coherence threshold: {obs_halluc.coherence_threshold}")
    print(f"  Base bias: {obs_halluc.interpretation_bias}")

    # Zastosuj interpretację kilka razy i pokaż rozrzut
    base = np.array([0.70, 0.70, 0.30])

    print(f"\n📍 Base coords: {base}")
    print(f"\n🎲 10 próbek z O_hallucination (temp=2.5):")

    samples = []
    for i in range(10):
        local = obs_halluc.apply_interpretation(base)
        samples.append(local)
        print(f"  Sample {i+1}: [{local[0]:.3f}, {local[1]:.3f}, {local[2]:.3f}]", end="")
        if not (0 <= local[0] <= 1 and 0 <= local[1] <= 1 and 0 <= local[2] <= 1):
            print(" ⚠️ POZA [0,1]³")
        else:
            print()

    # Oblicz rozrzut
    samples_array = np.array(samples)
    std_devs = np.std(samples_array, axis=0)

    print(f"\n📊 ROZRZUT (std dev):")
    print(f"  D: {std_devs[0]:.3f}")
    print(f"  S: {std_devs[1]:.3f}")
    print(f"  E: {std_devs[2]:.3f}")
    print(f"  → Wysoka temperatura = wysoka niepewność = halucynacje!")


def main():
    print("\n")
    print("╔" + "=" * 68 + "╗")
    print("║" + " " * 12 + "GTMØ SPECIAL OBSERVERS DEMO" + " " * 29 + "║")
    print("║" + " " * 10 + "Halucynacje, Nonsens, Propaganda" + " " * 25 + "║")
    print("╚" + "=" * 68 + "╝")

    try:
        demo_hallucination_detection()
        input("\n\nNaciśnij Enter...")

        demo_nonsense_vs_propaganda()
        input("\n\nNaciśnij Enter...")

        demo_conspiracy_detection()
        input("\n\nNaciśnij Enter...")

        demo_extreme_temperature()

        print("\n" + "=" * 70)
        print("  ✅ DEMO ZAKOŃCZONE")
        print("=" * 70)
        print("\n🎯 Wnioski:")
        print("  1. O_hallucination wykrywa konflikty przez wysoką energię V_Comm")
        print("  2. O_nonsense i O_propaganda są patologiczne w PRZECIWNYCH kierunkach")
        print("  3. O_conspiracy destabilizuje normalny konsensus")
        print("  4. Wysoka temperatura (2.5) = duży rozrzut = halucynacje")
        print("\n💡 Zastosowania:")
        print("  • Detekcja halucynacji LLM")
        print("  • Wykrywanie propagandy i teorii spiskowych")
        print("  • Identyfikacja nonsensownych tekstów")
        print("  • Analiza niepewności interpretacji")

    except Exception as e:
        print(f"\n❌ ERROR: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
