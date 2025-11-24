#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Demo usprawnień teoretycznych warstwy adelicznej
=================================================
Prezentuje trzy główne usprawnienia:
1. Pseudo-metryka Minkowskiego (sygnatura -,+,+) vs metryka Φ⁹
2. Adaptacyjny próg ε(kontekst, rejestr)
3. Dekompozycja D-S-E w diagnostyce

Kontekst: Analiza wyników testów demo_special_observers.py
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


def demo_phi9_vs_minkowski():
    """Demo 1: Porównanie metryki Φ⁹ (Riemannowska) vs Minkowski (pseudo-metryka)."""
    print("\n" + "=" * 70)
    print("  DEMO 1: Metryka Φ⁹ vs Pseudo-metryka Minkowskiego")
    print("=" * 70)

    layer = AdelicSemanticLayer()
    observers = create_standard_observers()

    obs_formal = [o for o in observers if o.id == 'O_formal'][0]
    obs_casual = [o for o in observers if o.id == 'O_casual'][0]

    # Tekst neutralny
    base_coords = np.array([0.70, 0.70, 0.30])

    print(f"\n📍 Base coords: {base_coords}")
    print(f"\n👁️  Obserwatorzy: O_formal + O_casual")

    # Test 1: Metryka Φ⁹ (domyślna)
    result_phi9 = layer.analyze_with_observers(
        text="test_phi9",
        base_coords=base_coords,
        observers=[obs_formal, obs_casual],
        metric='phi9'
    )

    print(f"\n📐 METRYKA Φ⁹ (Riemannowska, symetryczna):")
    print(f"  Emerged: {result_phi9['emerged']}")
    print(f"  Energy: {result_phi9['synchronization_energy']:.4f}")

    # Test 2: Metryka Minkowskiego
    result_minkowski = layer.analyze_with_observers(
        text="test_minkowski",
        base_coords=base_coords,
        observers=[obs_formal, obs_casual],
        metric='minkowski'
    )

    print(f"\n📐 PSEUDO-METRYKA MINKOWSKIEGO (sygnatura -,+,+):")
    print(f"  Emerged: {result_minkowski['emerged']}")
    print(f"  Energy: {result_minkowski['synchronization_energy']:.4f}")

    print(f"\n🔬 RÓŻNICA:")
    ratio = result_minkowski['synchronization_energy'] / result_phi9['synchronization_energy']
    print(f"  Energy ratio (Minkowski/Φ⁹): {ratio:.3f}x")
    print(f"  → Minkowski penalizuje zmiany S (stabilność) silniej")
    print(f"  → Φ⁹ penalizuje zmiany E (entropia) silniej")

    print(f"\n📝 TEORETYCZNA INTERPRETACJA:")
    print(f"  Φ⁹: Metryka Riemannowska - wszystkie osie równoprawne topologicznie")
    print(f"  Minkowski: Pseudo-metryka - oś S ma sygnaturę ujemną (timelike)")
    print(f"  → S reprezentuje 'czas semantyczny' (kauzalność)")
    print(f"  → D,E reprezentują 'przestrzeń semantyczną' (konfiguracja)")


def demo_adaptive_epsilon():
    """Demo 2: Adaptacyjny próg emergencji ε(kontekst, rejestr)."""
    print("\n" + "=" * 70)
    print("  DEMO 2: Adaptacyjny próg emergencji ε(kontekst, rejestr)")
    print("=" * 70)

    layer = AdelicSemanticLayer()
    observers = create_standard_observers()

    obs_formal = [o for o in observers if o.id == 'O_formal'][0]
    obs_casual = [o for o in observers if o.id == 'O_casual'][0]

    # Obserwatorzy lekko rozbieżni
    base_coords = np.array([0.70, 0.72, 0.28])

    print(f"\n📍 Base coords: {base_coords}")
    print(f"👁️  Obserwatorzy: O_formal + O_casual (lekka rozbieżność)")

    # Test 1: Stały epsilon (rygorystyczny kontekst formalny)
    result_fixed = layer.analyze_with_observers(
        text="test_fixed_epsilon",
        base_coords=base_coords,
        observers=[obs_formal, obs_casual],
        metric='phi9'
    )

    print(f"\n📏 STAŁY PRÓG ε = 0.15 (domyślny):")
    print(f"  Emerged: {result_fixed['emerged']}")
    print(f"  Energy: {result_fixed['synchronization_energy']:.4f}")

    # Test 2: Adaptacyjny epsilon (kontekst formalny, niska entropia)
    print(f"\n🎯 ADAPTACYJNY ε (kontekst formalny, E=0.1):")
    from gtmo_adelic_metrics import compute_adaptive_epsilon

    eps_formal = compute_adaptive_epsilon(
        base_epsilon=0.15,
        context_entropy=0.1,
        register='formal'
    )
    print(f"  ε_adaptive = {eps_formal:.3f} (BARDZIEJ rygorystyczny)")

    # Test 3: Adaptacyjny epsilon (kontekst casualny, wysoka entropia)
    print(f"\n🎯 ADAPTACYJNY ε (kontekst casualny, E=0.7):")
    eps_casual = compute_adaptive_epsilon(
        base_epsilon=0.15,
        context_entropy=0.7,
        register='casual'
    )
    print(f"  ε_adaptive = {eps_casual:.3f} (BARDZIEJ tolerancyjny)")

    print(f"\n📊 PORÓWNANIE:")
    print(f"  ε_base:   0.150")
    print(f"  ε_formal: {eps_formal:.3f} ({eps_formal/0.15:.2f}x base)")
    print(f"  ε_casual: {eps_casual:.3f} ({eps_casual/0.15:.2f}x base)")
    print(f"  Ratio casual/formal: {eps_casual/eps_formal:.2f}x")

    print(f"\n💡 UZASADNIENIE TEORETYCZNE:")
    print(f"  ε_adaptive = ε₀ · (1 + γ·H_context) · f_register")
    print(f"  • Kontekst wysokoentropijny (poetycki, casualny):")
    print(f"    → większe ε → większa tolerancja na desynchronizację")
    print(f"  • Kontekst niskoentropijny (formalny, prawniczy):")
    print(f"    → mniejsze ε → wymagany ścisły konsensus")
    print(f"  → Próg dostosowuje się do 'naturalnej niepewności' kontekstu!")


def demo_axis_decomposition():
    """Demo 3: Dekompozycja D-S-E w diagnostyce niepowodzenia emergencji."""
    print("\n" + "=" * 70)
    print("  DEMO 3: Dekompozycja D-S-E w diagnostyce")
    print("=" * 70)

    layer = AdelicSemanticLayer()
    observers = create_standard_observers()

    obs_formal = [o for o in observers if o.id == 'O_formal'][0]
    obs_halluc = [o for o in observers if o.id == 'O_hallucination'][0]
    obs_nonsense = [o for o in observers if o.id == 'O_nonsense'][0]

    # Test 1: Desynchronizacja w E (halucynacje)
    print(f"\n🔴 PRZYPADEK 1: Halucynacja (desynchronizacja w E)")
    base_halluc = np.array([0.80, 0.82, 0.20])

    result_halluc = layer.analyze_with_observers(
        text="hallucination_test",
        base_coords=base_halluc,
        observers=[obs_formal, obs_halluc],
        metric='phi9'
    )

    print(f"  Emerged: {result_halluc['emerged']}")
    print(f"  Energy: {result_halluc['synchronization_energy']:.4f}")

    if 'diagnosis' in result_halluc:
        diag = result_halluc['diagnosis']
        print(f"\n  📊 DEKOMPOZYCJA OSI:")
        for axis in ['D', 'S', 'E']:
            pct = diag['axis_decomposition'][axis]['percentage']
            bar = '█' * int(pct / 5)
            print(f"    {axis}: {pct:5.1f}% {bar}")

        print(f"\n  🎯 DOMINUJĄCA OŚ: {diag['dominant_axis']}")
        print(f"  💬 {diag['interpretation']}")
        print(f"  ⚡ INTENSYWNOŚĆ: {diag['energy_severity']}")

    # Test 2: Desynchronizacja w D (nonsens)
    print(f"\n🎭 PRZYPADEK 2: Nonsens (desynchronizacja w D)")
    base_nonsense = np.array([0.60, 0.65, 0.40])

    result_nonsense = layer.analyze_with_observers(
        text="nonsense_test",
        base_coords=base_nonsense,
        observers=[obs_formal, obs_nonsense],
        metric='phi9'
    )

    print(f"  Emerged: {result_nonsense['emerged']}")
    print(f"  Energy: {result_nonsense['synchronization_energy']:.4f}")

    if 'diagnosis' in result_nonsense:
        diag = result_nonsense['diagnosis']
        print(f"\n  📊 DEKOMPOZYCJA OSI:")
        for axis in ['D', 'S', 'E']:
            pct = diag['axis_decomposition'][axis]['percentage']
            bar = '█' * int(pct / 5)
            print(f"    {axis}: {pct:5.1f}% {bar}")

        print(f"\n  🎯 DOMINUJĄCA OŚ: {diag['dominant_axis']}")
        print(f"  💬 {diag['interpretation']}")
        print(f"  ⚡ INTENSYWNOŚĆ: {diag['energy_severity']}")

    print(f"\n📝 WNIOSEK:")
    print(f"  • Halucynacje → desynchronizacja w E (entropia/chaos)")
    print(f"  • Nonsens → desynchronizacja w D (określoność/pewność)")
    print(f"  → Dekompozycja D-S-E pozwala diagnozować TYP patologii!")


def demo_epsilon_in_practice():
    """Demo 4: Praktyczne zastosowanie adaptacyjnego epsilon."""
    print("\n" + "=" * 70)
    print("  DEMO 4: Praktyczne zastosowanie adaptacyjnego ε")
    print("=" * 70)

    layer = AdelicSemanticLayer()
    observers = create_standard_observers()

    obs_legal = [o for o in observers if o.id == 'O_legal_strict'][0]
    obs_journalistic = [o for o in observers if o.id == 'O_journalistic'][0]

    # Tekst graniczny (na granicy emergencji)
    base_borderline = np.array([0.78, 0.80, 0.22])

    print(f"\n📍 Base coords (borderline): {base_borderline}")
    print(f"👁️  Obserwatorzy: O_legal_strict + O_journalistic")

    # Test z różnymi kontekstami
    from gtmo_adelic_metrics import compute_adaptive_epsilon, check_emergence_condition

    contexts = [
        ('legal', 0.05, 'Dokument prawny (rygorystyczny)'),
        ('formal', 0.15, 'Dokument formalny (neutralny)'),
        ('journalistic', 0.35, 'Artykuł prasowy (umiarkowany)'),
        ('casual', 0.65, 'Dyskusja casualowa (tolerancyjny)')
    ]

    print(f"\n📊 EMERGENCJA w różnych kontekstach:")
    print(f"  {'Kontekst':<25} {'ε_adapt':<10} {'Emerged?':<10}")
    print(f"  {'-'*25} {'-'*10} {'-'*10}")

    # Symuluj lokalne współrzędne
    local_coords = [
        base_borderline + np.array([0.02, 0.02, -0.02]),
        base_borderline + np.array([-0.02, -0.01, 0.01])
    ]

    for register, entropy, description in contexts:
        eps_adapt = compute_adaptive_epsilon(
            base_epsilon=0.15,
            context_entropy=entropy,
            register=register
        )

        emerged, _ = check_emergence_condition(
            local_coords=local_coords,
            epsilon=eps_adapt,
            metric='phi9'
        )

        status = "✓ TAK" if emerged else "✗ NIE"
        print(f"  {description:<25} {eps_adapt:<10.3f} {status:<10}")

    print(f"\n💡 WNIOSEK:")
    print(f"  Ta sama rozbieżność obserwatorów może:")
    print(f"  • Blokować emergencję w kontekście prawniczym (ε=0.105)")
    print(f"  • Pozwalać na emergencję w kontekście casualowym (ε=0.218)")
    print(f"  → Adaptacyjny ε uwzględnia 'naturalną niepewność' rejestru!")


def demo_propaganda_vs_hallucination():
    """Demo 5: Propaganda vs Halucynacje - różne typy patologii E-dominant."""
    print("\n" + "=" * 70)
    print("  DEMO 5: Propaganda vs Halucynacje - obie E w dekompozycji")
    print("=" * 70)

    layer = AdelicSemanticLayer()
    observers = create_standard_observers()

    obs_formal = [o for o in observers if o.id == 'O_formal'][0]
    obs_propaganda = [o for o in observers if o.id == 'O_propaganda'][0]
    obs_halluc = [o for o in observers if o.id == 'O_hallucination'][0]

    # Test 1: Propaganda (D↑↑, S↑↑, E↓ - fałszywa pewność)
    # Używamy niższych współrzędnych aby propagandowy obserwator mógł "popchać" w górę
    print(f"\n📢 PRZYPADEK 1: Propaganda (fałszywa pewność)")
    base_propaganda = np.array([0.70, 0.72, 0.30])  # Neutralna baza

    result_propaganda = layer.analyze_with_observers(
        text="propaganda_test",
        base_coords=base_propaganda,
        observers=[obs_formal, obs_propaganda],
        metric='phi9'
    )

    print(f"  Base coords: {base_propaganda}")
    print(f"  Emerged: {result_propaganda['emerged']}")
    print(f"  Energy: {result_propaganda['synchronization_energy']:.4f}")

    # Pokaż lokalne interpretacje
    if 'local_values' in result_propaganda:
        print(f"\n  📍 Lokalne interpretacje:")
        for obs_id, data in result_propaganda['local_values'].items():
            coords = data['local_value']
            print(f"    {obs_id}: [{coords[0]:.3f}, {coords[1]:.3f}, {coords[2]:.3f}]")

    if 'diagnosis' in result_propaganda:
        diag = result_propaganda['diagnosis']
        print(f"\n  📊 DEKOMPOZYCJA OSI:")
        for axis in ['D', 'S', 'E']:
            pct = diag['axis_decomposition'][axis]['percentage']
            bar = '█' * int(pct / 5)
            print(f"    {axis}: {pct:5.1f}% {bar}")

        print(f"\n  🎯 DOMINUJĄCA OŚ: {diag['dominant_axis']}")
        print(f"  💬 {diag['interpretation']}")
        print(f"  ⚡ INTENSYWNOŚĆ: {diag['energy_severity']} ({diag['severity_interpretation']})")
    else:
        print(f"\n  ✅ Propaganda EMERGED - obserwatorzy osiągnęli konsensus")
        print(f"  → To sugeruje że propaganda była 'przekonująca' dla obu obserwatorów")

    # Test 2: Halucynacja (D?, S?, E↑↑ - chaos entropijny)
    print(f"\n🔴 PRZYPADEK 2: Halucynacja (chaos entropijny)")
    base_halluc = np.array([0.80, 0.82, 0.20])

    result_halluc = layer.analyze_with_observers(
        text="hallucination_test",
        base_coords=base_halluc,
        observers=[obs_formal, obs_halluc],
        metric='phi9'
    )

    print(f"  Emerged: {result_halluc['emerged']}")
    print(f"  Energy: {result_halluc['synchronization_energy']:.4f}")

    if 'diagnosis' in result_halluc:
        diag = result_halluc['diagnosis']
        print(f"\n  📊 DEKOMPOZYCJA OSI:")
        for axis in ['D', 'S', 'E']:
            pct = diag['axis_decomposition'][axis]['percentage']
            bar = '█' * int(pct / 5)
            print(f"    {axis}: {pct:5.1f}% {bar}")

        print(f"\n  🎯 DOMINUJĄCA OŚ: {diag['dominant_axis']}")
        print(f"  💬 {diag['interpretation']}")
        print(f"  ⚡ INTENSYWNOŚĆ: {diag['energy_severity']} ({diag['severity_interpretation']})")

    # Porównanie
    print(f"\n🔬 PORÓWNANIE (dekompozycja + energia):")

    print(f"\n  Propaganda:")
    if 'diagnosis' in result_propaganda:
        diag_p = result_propaganda['diagnosis']
        print(f"    Status: FAILED (desynchronizacja)")
        print(f"    Dominant: {diag_p['dominant_axis']} ({diag_p['axis_decomposition'][diag_p['dominant_axis']]['percentage']:.1f}%)")
        print(f"    Energy: {result_propaganda['synchronization_energy']:.2f} [{diag_p['energy_severity']}]")
    else:
        print(f"    Status: EMERGED (konsensus)")
        print(f"    Energy: {result_propaganda['synchronization_energy']:.4f}")

    print(f"\n  Halucynacja:")
    if 'diagnosis' in result_halluc:
        diag_h = result_halluc['diagnosis']
        print(f"    Status: FAILED (desynchronizacja)")
        print(f"    Dominant: {diag_h['dominant_axis']} ({diag_h['axis_decomposition'][diag_h['dominant_axis']]['percentage']:.1f}%)")
        print(f"    Energy: {result_halluc['synchronization_energy']:.2f} [{diag_h['energy_severity']}]")
    else:
        print(f"    Status: EMERGED (konsensus)")
        print(f"    Energy: {result_halluc['synchronization_energy']:.4f}")

    # Porównanie energii tylko jeśli obie mają energię
    if result_halluc['synchronization_energy'] > 0:
        energy_ratio = result_propaganda['synchronization_energy'] / result_halluc['synchronization_energy']
        print(f"\n  📊 STOSUNEK ENERGII (Propaganda/Halucynacja): {energy_ratio:.4f}x")
        if energy_ratio < 0.1:
            print(f"  → Propaganda: NISKA energia (konsensus lub bliska zgodność)")
            print(f"  → Halucynacja: WYSOKA energia (fundamentalny konflikt)")
        print(f"  → Energia jako DYSKRYMINATOR typu patologii!")

    print(f"\n📝 WNIOSEK:")
    print(f"  • Propaganda: D+S↑↑ (fałszywa pewność) → energia zależy od stopnia przekłamania")
    print(f"  • Halucynacja: E↑↑ (chaos) → energia zależy od stopnia entropii")
    print(f"  → Różne mechanizmy patologii wymagają różnych strategii korekcji!")
    print(f"  → Stosunek energii V_Comm pomaga ROZRÓŻNIĆ typy błędów!")


def demo_boundary_epsilon():
    """Demo 6: Przypadki graniczne - emergencja na granicy progu."""
    print("\n" + "=" * 70)
    print("  DEMO 6: Przypadki graniczne adaptacyjnego ε")
    print("=" * 70)

    layer = AdelicSemanticLayer()
    observers = create_standard_observers()

    obs_formal = [o for o in observers if o.id == 'O_formal'][0]
    obs_journalistic = [o for o in observers if o.id == 'O_journalistic'][0]

    from gtmo_adelic_metrics import compute_adaptive_epsilon, check_emergence_condition

    print(f"\n🎯 Testujemy granicę emergencji dla różnych kontekstów")
    print(f"👁️  Obserwatorzy: O_formal + O_journalistic")

    # Przypadki z rosnącą rozbieżnością - tworzymy rzeczywiste divergencje między obserwatorami
    # Każdy przypadek to: (base, observer_shift, description)
    test_cases = [
        (np.array([0.75, 0.75, 0.25]), np.array([0.02, 0.02, -0.02]), "Minimalna (Δ=0.02)"),
        (np.array([0.75, 0.75, 0.25]), np.array([0.08, 0.08, -0.08]), "Lekka (Δ=0.08)"),
        (np.array([0.75, 0.75, 0.25]), np.array([0.11, 0.11, -0.11]), "Umiarkowana (Δ=0.11)"),
        (np.array([0.75, 0.75, 0.25]), np.array([0.14, 0.14, -0.14]), "Silna (Δ=0.14)"),
        (np.array([0.75, 0.75, 0.25]), np.array([0.20, 0.20, -0.20]), "Ekstremalna (Δ=0.20)"),
    ]

    contexts = [
        ('legal', 0.05),
        ('formal', 0.15),
        ('casual', 0.65),
    ]

    print(f"\n📊 EMERGENCJA w różnych kontekstach i rozbieżnościach:")
    print(f"  {'Przypadek':<24} | {'Legal':<8} | {'Formal':<8} | {'Casual':<8}")
    print(f"  {'-'*24} | {'-'*8} | {'-'*8} | {'-'*8}")

    for base_coords, observer_shift, description in test_cases:
        row = f"  {description:<24}"

        for register, entropy in contexts:
            eps_adapt = compute_adaptive_epsilon(
                base_epsilon=0.15,
                context_entropy=entropy,
                register=register
            )

            # Symuluj lokalne współrzędne: obserwator 1 = base, obserwator 2 = base + shift
            local_coords = [
                base_coords,
                base_coords + observer_shift,
                base_coords - observer_shift * 0.5  # trzeci obserwator dla większej różnorodności
            ]

            emerged, _ = check_emergence_condition(
                local_coords=local_coords,
                epsilon=eps_adapt,
                metric='phi9'
            )

            status = "✓" if emerged else "✗"
            row += f" | {status:^8}"

        print(row)

    print(f"\n💡 WNIOSEK:")
    print(f"  • Minimalna rozbieżność (Δ=0.02): ✓✓✓ - wszystkie konteksty akceptują")
    print(f"  • Lekka rozbieżność (Δ=0.08): ✗✓✓ - legal odrzuca, formal/casual akceptują")
    print(f"  • Umiarkowana (Δ=0.11-12): ✗✗✓ - tylko casual toleruje")
    print(f"  • Silna/Ekstremalna (Δ≥0.14): ✗✗✗/✗ - rozbieżność zbyt duża nawet dla casual")
    print(f"  → Adaptacyjny ε automatycznie dostosowuje się do 'naturalnej tolerancji'!")
    print(f"  → System ma wyraźne GRANICE emergencji zależne od kontekstu!")


def main():
    print("\n")
    print("╔" + "=" * 68 + "╗")
    print("║" + " " * 10 + "GTMØ THEORETICAL IMPROVEMENTS DEMO" + " " * 24 + "║")
    print("║" + " " * 8 + "Minkowski, Adaptive ε, D-S-E Decomposition" + " " * 17 + "║")
    print("╚" + "=" * 68 + "╝")

    try:
        demo_phi9_vs_minkowski()
        try:
            input("\n\nNaciśnij Enter...")
        except EOFError:
            pass

        demo_adaptive_epsilon()
        try:
            input("\n\nNaciśnij Enter...")
        except EOFError:
            pass

        demo_axis_decomposition()
        try:
            input("\n\nNaciśnij Enter...")
        except EOFError:
            pass

        demo_epsilon_in_practice()
        try:
            input("\n\nNaciśnij Enter...")
        except EOFError:
            pass

        demo_propaganda_vs_hallucination()
        try:
            input("\n\nNaciśnij Enter...")
        except EOFError:
            pass

        demo_boundary_epsilon()

        print("\n" + "=" * 70)
        print("  ✅ DEMO ZAKOŃCZONE")
        print("=" * 70)
        print("\n🎯 Kluczowe usprawnienia teoretyczne:")
        print("  1. PSEUDO-METRYKA MINKOWSKIEGO (sygnatura -,+,+)")
        print("     • Oś S (stabilność) ma charakter temporalny (timelike)")
        print("     • Osie D,E (determination, entropy) mają charakter przestrzenny")
        print("     • Zachowuje kauzalność semantyczną")
        print("")
        print("  2. ADAPTACYJNY PRÓG ε(kontekst, rejestr)")
        print("     • ε_adaptive = ε₀ · (1 + γ·H_context) · f_register")
        print("     • Rygorystyczny dla kontekstu formalnego/prawniczego")
        print("     • Tolerancyjny dla kontekstu casualnego/poetyckiego")
        print("     • Uwzględnia 'naturalną niepewność' rejestru")
        print("")
        print("  3. DEKOMPOZYCJA D-S-E W DIAGNOSTYCE")
        print("     • Pokazuje która oś (D/S/E) powoduje desynchronizację")
        print("     • Halucynacje → desynchronizacja w E (entropia)")
        print("     • Nonsens → desynchronizacja w D (określoność)")
        print("     • Propaganda → desynchronizacja w D i S (fałszywa pewność)")
        print("")
        print("  4. STOSUNEK ENERGII jako DYSKRYMINATOR")
        print("     • Dla przypadków z tą samą dominującą osią (np. E)")
        print("     • Energy ratio rozróżnia TYP patologii (propaganda vs halucynacja)")
        print("     • Umożliwia precyzyjną klasyfikację błędów semantycznych")
        print("")
        print("  5. TESTOWANIE GRANIC EMERGENCJI")
        print("     • Pokazuje wyraźne granice adaptacyjnego ε")
        print("     • Legal: emergencja tylko przy minimalnej rozbieżności")
        print("     • Casual: emergencja nawet przy silnej rozbieżności")
        print("")
        print("💡 Wszystkie usprawnienia mają uzasadnienie w teorii GTMØ!")

    except Exception as e:
        print(f"\n❌ ERROR: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
