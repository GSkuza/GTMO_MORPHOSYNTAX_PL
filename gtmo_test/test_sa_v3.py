#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Test script for SA v3.0 - porównanie z SA v2.0
"""

from gtmo_constitutional_duality import ConstitutionalDualityCalculator, SAv3Calculator, SAv3Config

# === DANE Z PIERWOTNEGO ZAPYTANIA ===
TEXT_CONTENT = """Sygn akt: I C 75/24 WYROK W IMIENIU RZECZYPOSPOLITEJ POLSKIEJ Dnia 20 marca 2025 r Sąd Rejonowy w Gdyni I Wydział Cywilny w składzie następującym: Przewodniczący: SSR Joanna Jank po rozpoznaniu na rozprawie w dniu 20 marca 2025 r w G sprawie z powództwa K K przeciwko Z Ż o zachowek I zasądza od pozwanej na rzecz powoda kwotę 39 318, 75 zł (trzydzieści dziewięć tysięcy trzysta osiemnaście złotych i siedemdziesiąt pięć groszy) wraz z odsetkami ustawowymi za opóźnienie od 5 sierpnia 2023 r do dnia zapłaty"""

# Metryki z GTMØ v2.0 (z JSONa)
INPUT_METRICS = {
    "CD": 2.7041,
    "CI_components": {
        "morph": 4.009,
        "synt": 12.3476,
        "sem": 1.7639
    },
    "Phase": {
        "D": 0.618,
        "S": 0.591,
        "E": 0.501
    },
    "Kinetic_Power_Est": 0.85,  # Wyrok sądowy - wysoka moc kinetyczna
    "SA_v2_Original": 0.130  # Z JSONa
}

def main():
    print("=" * 80)
    print("TEST SA v3.0 - Porównanie z SA v2.0")
    print("=" * 80)
    print(f"\nTekst: '{TEXT_CONTENT[:80]}...'")
    print(f"Długość: {len(TEXT_CONTENT)} znaków\n")

    # === TEST 1: Bezpośrednie użycie SAv3Calculator ===
    print("\n" + "=" * 80)
    print("TEST 1: Bezpośrednie użycie SAv3Calculator")
    print("=" * 80)

    sa_v3_calc = SAv3Calculator()
    result = sa_v3_calc.calculate_sa_v3(
        text=TEXT_CONTENT,
        CD=INPUT_METRICS["CD"],
        CI_morph=INPUT_METRICS["CI_components"]["morph"],
        CI_synt=INPUT_METRICS["CI_components"]["synt"],
        CI_sem=INPUT_METRICS["CI_components"]["sem"],
        D=INPUT_METRICS["Phase"]["D"],
        S=INPUT_METRICS["Phase"]["S"],
        E=INPUT_METRICS["Phase"]["E"],
        kinetic_power_est=INPUT_METRICS["Kinetic_Power_Est"]
    )

    print("\nRESULTS COMPARISON:")
    print("-" * 80)
    print(f"SA v2.0 (from JSON):   {INPUT_METRICS['SA_v2_Original']:.4f} ({INPUT_METRICS['SA_v2_Original']*100:.1f}%)")
    print(f"SA v2.0 (calculated):  {result['SA_v2']:.4f} ({result['SA_v2']*100:.1f}%)")
    print(f"SA v3.0 (NEW):         {result['SA_v3']:.4f} ({result['SA_v3']*100:.1f}%)")
    print(f"Delta (v3.0 - v2.0):   {result['Delta']:+.4f} ({result['Delta']*100:+.1f}%)")

    # Interpretacja
    print("\nINTERPRETATION:")
    print("-" * 80)
    if result['SA_v2'] < 0.3:
        cat_v2 = "NISKA (Chaos/Specjalistyczny)"
    elif result['SA_v2'] < 0.5:
        cat_v2 = "ŚREDNIA"
    else:
        cat_v2 = "WYSOKA (Powszechny)"

    if result['SA_v3'] < 0.3:
        cat_v3 = "NISKA (Chaos/Specjalistyczny)"
    elif result['SA_v3'] < 0.5:
        cat_v3 = "ŚREDNIA"
    else:
        cat_v3 = "WYSOKA (Powszechny)"

    print(f"Kategoria v2.0: {cat_v2}")
    print(f"Kategoria v3.0: {cat_v3}")

    # Komponenty SA v3.0
    print("\nSA v3.0 COMPONENTS:")
    print("-" * 80)
    comp = result['Components']
    print(f"CI_Weighted:      {comp['CI_Weighted']:.4f} (vs CI_total={sum(INPUT_METRICS['CI_components'].values()):.4f})")
    print(f"SA_Base:          {comp['SA_Base']:.4f}")
    print(f"Hoyer_Raw:        {comp['Hoyer_Raw']:.4f}")
    print(f"Focus_Score:      {comp['Focus_Score']:.4f} (CDF scaled)")
    print(f"Q_Phase:          {comp['Q_Phase']:.4f}")
    print(f"Q_Kinetic:        {comp['Q_Kinetic']:.4f}")
    print(f"Q_Phi:            {comp['Q_Phi']:.4f}")
    print(f"Balance:          {comp['Balance']:.4f}")
    print(f"Boost_Factor:     {comp['Boost_Factor']:.4f}")

    # === TEST 2: Przez ConstitutionalDualityCalculator ===
    print("\n" + "=" * 80)
    print("TEST 2: Przez ConstitutionalDualityCalculator (pełna integracja)")
    print("=" * 80)

    # Oblicz ambiguity i depth z CI/CD
    # Z formuł: CD = (1/ambiguity) × depth × √(D×S/E)
    #           CI = ambiguity × depth × √(E/(D×S))
    # Możemy oszacować ambiguity i depth
    import numpy as np
    D = INPUT_METRICS["Phase"]["D"]
    S = INPUT_METRICS["Phase"]["S"]
    E = INPUT_METRICS["Phase"]["E"]
    geometric_balance = np.sqrt((D * S) / E)
    geometric_tension = np.sqrt(E / (D * S))

    # Zakładamy depth=5 (typowy dla wyroku)
    depth = 5
    # Z CD = (1/ambiguity) × depth × geometric_balance
    ambiguity = (depth * geometric_balance) / INPUT_METRICS["CD"]

    print(f"\nParametry wejściowe (szacowane):")
    print(f"  ambiguity: {ambiguity:.4f}")
    print(f"  depth: {depth}")
    print(f"  D, S, E: {D:.3f}, {S:.3f}, {E:.3f}")

    # Test z SA v3.0
    calc_v3 = ConstitutionalDualityCalculator(use_sa_v3=True)
    metrics_v3 = calc_v3.calculate_metrics(
        ambiguity=ambiguity,
        depth=depth,
        D=D,
        S=S,
        E=E,
        text=TEXT_CONTENT,
        kinetic_power_est=INPUT_METRICS["Kinetic_Power_Est"]
    )

    print(f"\nRESULTS FROM CALCULATOR:")
    print("-" * 80)
    print(f"CD:               {metrics_v3.CD:.4f} (expected: {INPUT_METRICS['CD']:.4f})")
    print(f"CI:               {metrics_v3.CI:.4f} (expected: {sum(INPUT_METRICS['CI_components'].values()):.4f})")
    print(f"SA v2.0:          {metrics_v3.SA:.4f} ({metrics_v3.SA*100:.1f}%)")
    print(f"SA v3.0:          {metrics_v3.SA_v3:.4f} ({metrics_v3.SA_v3*100:.1f}%)")
    print(f"Delta:            {metrics_v3.SA_v3 - metrics_v3.SA:+.4f}")

    print("\n" + "=" * 80)
    print("TESTS COMPLETED")
    print("=" * 80)
    print("\nCONCLUSIONS:")
    print(f"  • SA v2.0 bazuje tylko na CD/(CI+CD)")
    print(f"  • SA v3.0 uwzględnia:")
    print(f"    - Weighted CI (morfologia × {SAv3Config.W_MORPH}, składnia × {SAv3Config.W_SYNT}, semantyka × {SAv3Config.W_SEM})")
    print(f"    - HerBERT embeddings (Hoyer sparsity)")
    print(f"    - Phase Quality (D, S, E)")
    print(f"    - Kinetic Power")
    print(f"    - Topological Balance")
    print(f"  • SA v3.0 daje bardziej realistyczną ocenę dostępności semantycznej")

if __name__ == "__main__":
    main()
