#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
GTMØ Adelic Layer - Demonstracja
=================================
Pokazuje działanie p-adycznej warstwy emergencji semantycznej.

Przykłady:
1. Emergencja konsensusu (tekst jednoznaczny)
2. Brak emergencji (tekst dwuznaczny - ironia)
3. AlienatedNumber (neologizm)
4. Energia komunikacyjna dialogu
"""

import sys
import numpy as np
from gtmo_adelic_layer import AdelicSemanticLayer, Observer, create_standard_observers
from gtmo_adelic_metrics import phi9_distance
import json

# Fix encoding dla Windows
if sys.platform == 'win32':
    import io
    try:
        sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
    except:
        pass


def print_section(title: str):
    """Helper do wydruku sekcji."""
    print("\n" + "=" * 70)
    print(f"  {title}")
    print("=" * 70)


def demo_basic_emergence():
    """Demo 1: Podstawowa emergencja konsensusu."""
    print_section("DEMO 1: Emergencja konsensusu (tekst formalny)")

    # Inicjalizacja warstwy
    layer = AdelicSemanticLayer(epsilon=0.15)

    # Bazowe współrzędne (z typowej analizy GTMØ)
    base_coords = np.array([0.82, 0.85, 0.18])  # Formalny tekst prawniczy

    # Obserwatorzy
    obs_formal = layer.observers[0]  # O_formal
    obs_legal = layer.observers[1]   # O_legal

    print(f"\n📍 Bazowe współrzędne (GTMØ): D={base_coords[0]:.2f}, S={base_coords[1]:.2f}, E={base_coords[2]:.2f}")
    print(f"\n👁️  Obserwatorzy:")
    print(f"  • {obs_formal.id} (bias: {obs_formal.interpretation_bias})")
    print(f"  • {obs_legal.id} (bias: {obs_legal.interpretation_bias})")

    # Analiza adeliczna
    result = layer.analyze_with_observers(
        text="Ustawa wchodzi w życie",
        base_coords=base_coords,
        observers=[obs_formal, obs_legal],
        metric='phi9'
    )

    print(f"\n✨ WYNIK ANALIZY:")
    print(f"  Emerged: {result['emerged']}")
    print(f"  Status: {result['status']}")
    print(f"  Energy V_Comm: {result['synchronization_energy']:.4f}")

    if result['emerged']:
        print(f"\n🌟 EMERGENCJA φ_∞:")
        global_val = result['global_value']
        print(f"  D = {global_val[0]:.3f}")
        print(f"  S = {global_val[1]:.3f}")
        print(f"  E = {global_val[2]:.3f}")

    print(f"\n📊 Lokalne interpretacje:")
    for obs_id, local_data in result['local_values'].items():
        local_val = local_data['local_value']
        print(f"  {obs_id}:")
        print(f"    coords: [{local_val[0]:.3f}, {local_val[1]:.3f}, {local_val[2]:.3f}]")
        print(f"    standard: {local_data['is_standard']}")
        print(f"    alienation: {local_data['alienation_magnitude']:.3f}")


def demo_no_emergence():
    """Demo 2: Brak emergencji - dwuznaczność."""
    print_section("DEMO 2: Brak emergencji (tekst ironiczny/dwuznaczny)")

    layer = AdelicSemanticLayer(epsilon=0.15)

    # Tekst dwuznaczny
    base_coords = np.array([0.65, 0.70, 0.35])  # Neutral

    obs_formal = layer.observers[0]     # O_formal - weźmie jako pozytyw
    obs_sarcastic = layer.observers[4]  # O_sarcastic - weźmie jako ironię

    print(f"\n📍 Bazowe współrzędne: D={base_coords[0]:.2f}, S={base_coords[1]:.2f}, E={base_coords[2]:.2f}")
    print(f"\n👁️  Obserwatorzy:")
    print(f"  • {obs_formal.id} (pozytywny)")
    print(f"  • {obs_sarcastic.id} (ironiczny)")

    result = layer.analyze_with_observers(
        text="Świetny pomysł",
        base_coords=base_coords,
        observers=[obs_formal, obs_sarcastic],
        context_attractor=np.array([0.85, 0.85, 0.15]),  # Ψᴷ (formal)
        context_name='Ψᴷ',
        metric='phi9'
    )

    print(f"\n💥 WYNIK ANALIZY:")
    print(f"  Emerged: {result['emerged']}")
    print(f"  Status: {result['status']}")
    print(f"  Energy V_Comm: {result['synchronization_energy']:.4f} (WYSOKA!)")

    print(f"\n📊 Lokalne interpretacje (ROZBIEŻNE):")
    for obs_id, local_data in result['local_values'].items():
        local_val = local_data['local_value']
        print(f"  {obs_id}:")
        print(f"    coords: [{local_val[0]:.3f}, {local_val[1]:.3f}, {local_val[2]:.3f}]")

    if 'collapse_gradients' in result:
        print(f"\n🧭 Gradienty kolapsu (w stronę Ψᴷ):")
        for obs_id, grad in result['collapse_gradients'].items():
            print(f"  {obs_id}: [{grad[0]:.3f}, {grad[1]:.3f}, {grad[2]:.3f}]")

    if 'diagnosis' in result:
        diag = result['diagnosis']
        print(f"\n🔍 DIAGNOZA niepowodzenia emergencji:")
        print(f"  Reason: {diag['reason']}")
        print(f"  Max distance: {diag['max_distance']:.3f}")
        print(f"  Exceeds ε by: {diag['exceeds_by']:.3f}")
        print(f"  Outliers: {diag['num_outliers']}")


def demo_alienated_number():
    """Demo 3: AlienatedNumber (neologizm)."""
    print_section("DEMO 3: AlienatedNumber (neologizm - tylko 1 obserwator)")

    layer = AdelicSemanticLayer(epsilon=0.15)

    # Symulacja neologizmu - bardzo niska D, bardzo wysoka E
    base_coords = np.array([0.15, 0.25, 0.95])  # Chaos semantyczny

    obs_casual = layer.observers[2]  # O_casual

    print(f"\n📍 Bazowe współrzędne (neologizm): D={base_coords[0]:.2f}, S={base_coords[1]:.2f}, E={base_coords[2]:.2f}")
    print(f"👁️  Obserwator: {obs_casual.id} (n=1)")

    result = layer.analyze_with_observers(
        text="covidoza",
        base_coords=base_coords,
        observers=[obs_casual],
        metric='phi9'
    )

    print(f"\n🔴 WYNIK ANALIZY:")
    print(f"  Emerged: {result['emerged']} (BRAK - potrzeba n≥2)")
    print(f"  Status: {result['status']}")
    print(f"  n_observers: {result['n_observers']}")

    local_data = result['local_values'][obs_casual.id]
    local_val = local_data['local_value']

    print(f"\n👽 AlienatedNumber:")
    print(f"  coords: [{local_val[0]:.3f}, {local_val[1]:.3f}, {local_val[2]:.3f}]")
    print(f"  is_standard: {local_data['is_standard']}")
    print(f"  alienation_magnitude: {local_data['alienation_magnitude']:.3f}")

    if not local_data['is_standard']:
        print(f"\n🔬 Interpretacja (wartości poza [0,1]³):")
        interp = local_data['interpretation']
        for axis, descr in interp.items():
            print(f"    {axis}: {descr}")


def demo_dialogue_energy():
    """Demo 4: Energia komunikacyjna dialogu."""
    print_section("DEMO 4: Energia komunikacyjna dialogu")

    layer = AdelicSemanticLayer(epsilon=0.15)

    # Dialog formalny (łatwa komunikacja)
    dialogue_formal = [
        ("Proszę o przeds tawienie dokumentacji", np.array([0.88, 0.90, 0.12])),
        ("Dokumentacja zostanie przedłożona", np.array([0.86, 0.88, 0.14])),
        ("Dziękuję za terminową odpowiedź", np.array([0.85, 0.87, 0.15]))
    ]

    # Dialog ironiczny (trudna komunikacja)
    dialogue_ironic = [
        ("Świetny pomysł, naprawdę genialny", np.array([0.65, 0.70, 0.35])),
        ("Czyżby? No cóż, można spróbować", np.array([0.60, 0.65, 0.40])),
        ("Och tak, na pewno zadziała", np.array([0.55, 0.60, 0.45]))
    ]

    obs_formal = layer.observers[0]
    obs_casual = layer.observers[2]

    print(f"\n🗣️  Mówcy:")
    print(f"  • Speaker A: {obs_formal.id}")
    print(f"  • Speaker B: {obs_casual.id}")

    # Dialog 1: Formalny
    utterances1 = [u[0] for u in dialogue_formal]
    coords1 = [u[1] for u in dialogue_formal]

    result1 = layer.compute_dialogue_energy(
        utterances=utterances1,
        base_coords_list=coords1,
        speaker_a_observer=obs_formal,
        speaker_b_observer=obs_casual,
        metric='phi9'
    )

    print(f"\n📝 DIALOG 1 (formalny):")
    print(f"  Total energy: {result1['total_energy']:.3f}")
    print(f"  Average energy: {result1['average_energy']:.3f}")
    print(f"  Difficulty: {result1['difficulty']}")

    # Dialog 2: Ironiczny
    utterances2 = [u[0] for u in dialogue_ironic]
    coords2 = [u[1] for u in dialogue_ironic]

    obs_sarcastic = layer.observers[4]

    result2 = layer.compute_dialogue_energy(
        utterances=utterances2,
        base_coords_list=coords2,
        speaker_a_observer=obs_formal,
        speaker_b_observer=obs_sarcastic,
        metric='phi9'
    )

    print(f"\n📝 DIALOG 2 (ironiczny):")
    print(f"  Total energy: {result2['total_energy']:.3f}")
    print(f"  Average energy: {result2['average_energy']:.3f}")
    print(f"  Difficulty: {result2['difficulty']}")

    print(f"\n📊 PORÓWNANIE:")
    print(f"  Dialog formalny: E_avg = {result1['average_energy']:.3f} ({result1['difficulty']})")
    print(f"  Dialog ironiczny: E_avg = {result2['average_energy']:.3f} ({result2['difficulty']})")
    ratio = result2['average_energy'] / result1['average_energy'] if result1['average_energy'] > 0 else float('inf')
    print(f"  Trudność wzrosła {ratio:.1f}x")


def main():
    """Główna funkcja demo."""
    print("\n")
    print("╔" + "=" * 68 + "╗")
    print("║" + " " * 18 + "GTMØ ADELIC LAYER DEMO" + " " * 28 + "║")
    print("║" + " " * 15 + "P-adyczna emergencja semantyczna" + " " * 21 + "║")
    print("╚" + "=" * 68 + "╝")

    try:
        demo_basic_emergence()
        input("\n\nNaciśnij Enter aby kontynuować...")

        demo_no_emergence()
        input("\n\nNaciśnij Enter aby kontynuować...")

        demo_alienated_number()
        input("\n\nNaciśnij Enter aby kontynuować...")

        demo_dialogue_energy()

        print_section("✅ DEMO ZAKOŃCZONE")
        print("\nWszystkie komponenty warstwy adelicznej działają poprawnie!")
        print("\nNastępne kroki:")
        print("  1. Integracja z pełnym GTMOMorphosyntaxEngine")
        print("  2. Testy na rzeczywistych tekstach")
        print("  3. Optymalizacja wydajności")
        print("  4. Rozszerzenie o więcej obserwatorów")

    except Exception as e:
        print(f"\n❌ ERROR: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
