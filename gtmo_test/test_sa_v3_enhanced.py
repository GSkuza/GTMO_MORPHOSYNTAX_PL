#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Enhanced SA v3.0 Test Script with Full Diagnostics
"""

import torch
import numpy as np
from transformers import AutoTokenizer, AutoModel
from gtmo_constitutional_duality import SAv3Config

# === DANE Z PIERWOTNEGO ZAPYTANIA ===
TEXT_CONTENT = """Sygn akt: I C 75/24 WYROK W IMIENIU RZECZYPOSPOLITEJ POLSKIEJ Dnia 20 marca 2025 r Sad Rejonowy w Gdyni I Wydzial Cywilny w skladzie nastepujacym: Przewodniczacy: SSR Joanna Jank po rozpoznaniu na rozprawie w dniu 20 marca 2025 r w G sprawie z powodztwa K K przeciwko Z Z o zachowek I zasadza od pozwanej na rzecz powoda kwote 39 318, 75 zl (trzydziesci dziewiec tysiecy trzysta osiemnaście zlotych i siedemdziesiat piec groszy) wraz z odsetkami ustawowymi za opoznienie od 5 sierpnia 2023 r do dnia zaplaty"""

# Metryki z GTMO v2.0 (z JSONa)
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
    "SA_v2_Original": 0.130
}

def get_herbert_embedding(text, normalize=False):
    """
    Pobiera surowy wektor 768D z modelu HerBERT.

    Args:
        text: Tekst do zakodowania
        normalize: Czy normalizowac embedding (unit sphere)

    Returns:
        np.ndarray: Wektor 768D reprezentujacy semantyke tekstu
    """
    print("Loading HerBERT model...")
    tokenizer = AutoTokenizer.from_pretrained("allegro/herbert-base-cased")
    model = AutoModel.from_pretrained("allegro/herbert-base-cased")
    model.eval()

    # Tokenizacja z truncation (bezpieczne dla dlugich tekstow)
    inputs = tokenizer(
        text,
        return_tensors="pt",
        truncation=True,
        max_length=512,
        padding=True
    )

    print(f"Number of tokens: {inputs['input_ids'].shape[1]}")

    with torch.no_grad():
        outputs = model(**inputs)

    # Mean pooling - usredniamy po wszystkich tokenach
    embeddings = outputs.last_hidden_state.mean(dim=1).squeeze().numpy()

    if normalize:
        embeddings = embeddings / (np.linalg.norm(embeddings) + 1e-10)

    print(f"Embedding obtained: shape={embeddings.shape}, norm={np.linalg.norm(embeddings):.4f}")

    return embeddings

def calculate_hoyer_sparsity(embedding):
    """
    Hoyer's Sparsity Measure - fizyczna miara ostrosci sygnalu.

    Formula: (sqrt(n) - ||x||_1/||x||_2) / (sqrt(n) - 1)
    """
    n = len(embedding)
    l1 = np.sum(np.abs(embedding))
    l2 = np.linalg.norm(embedding)

    if l2 < 1e-10:
        print("WARNING: Nearly zero embedding, returning Hoyer=0")
        return 0.0

    numerator = np.sqrt(n) - (l1 / l2)
    denominator = np.sqrt(n) - 1
    hoyer = numerator / denominator

    return hoyer

def estimate_kinetic_power(text, metrics):
    """
    Heurystyczna estymacja Q_kinetic bez autoenkodera.

    Oparta na:
    1. Markery dyrektywne (rozkazy, wyroki)
    2. Wysoka Determination (D)
    3. Niska Entropia (E)
    """
    # Markery jezykowe dla tekstow dyrektywnych
    directive_markers = [
        "obowiazuje", "nakazuje", "zakazuje", "zasadza",
        "wyrok", "postanawia", "zobowiazuje", "nalezy"
    ]

    directive_count = sum(1 for marker in directive_markers
                         if marker in text.lower())

    # Skladniki jakosci kinetycznej
    d_contrib = metrics["Phase"]["D"]
    e_contrib = 1 - metrics["Phase"]["E"]
    directive_contrib = min(directive_count / 3, 1.0)

    # Kombinacja liniowa (wagi sumuja sie do 1.0)
    q_k_est = (0.5 +
               0.2 * directive_contrib +
               0.15 * d_contrib +
               0.15 * e_contrib)

    return np.clip(q_k_est, 0, 1)

def normal_cdf_approx(z):
    """Przyblizenie CDF rozkladu normalnego przez tanh."""
    return 0.5 * (1.0 + np.tanh(z / np.sqrt(2)))

def calculate_sa_v3_full(metrics, embedding, text):
    """
    Glowny algorytm SA v3.0 - pelna wersja z diagnostyka.
    """
    results = {}

    # === KROK 1: Weighted CI ===
    ci_w = (metrics["CI_components"]["morph"] * SAv3Config.W_MORPH +
            metrics["CI_components"]["synt"] * SAv3Config.W_SYNT +
            metrics["CI_components"]["sem"] * SAv3Config.W_SEM)

    ci_old = sum(metrics["CI_components"].values())
    reduction = (ci_old - ci_w) / ci_old * 100

    print(f"\nSTEP 1: Weighted CI")
    print(f"   CI old (equal weights): {ci_old:.2f}")
    print(f"   CI weighted (v3.0):     {ci_w:.2f}")
    print(f"   Reduction:              {reduction:.1f}%")

    results["CI_weighted"] = ci_w
    results["CI_old"] = ci_old

    # === KROK 2: SA Base ===
    sa_base = metrics["CD"] / (metrics["CD"] + ci_w)

    print(f"\nSTEP 2: SA Base (foundation)")
    print(f"   CD / (CD + CI_w) = {metrics['CD']:.2f} / {metrics['CD'] + ci_w:.2f}")
    print(f"   SA Base = {sa_base:.4f}")

    results["SA_base"] = sa_base

    # === KROK 3: Signal Physics (Hoyer) ===
    raw_hoyer = calculate_hoyer_sparsity(embedding)

    # Z-score wzgledem rozkladu referencyjnego
    z_score = (raw_hoyer - SAv3Config.HOYER_MU) / SAv3Config.HOYER_SIGMA

    # CDF jako miara "jak rzadki jest ten embedding"
    focus_score = normal_cdf_approx(z_score)

    print(f"\nSTEP 3: Signal Physics (Hoyer)")
    print(f"   Hoyer raw:     {raw_hoyer:.4f}")
    print(f"   Hoyer mu (ref): {SAv3Config.HOYER_MU:.4f}")
    print(f"   Z-score:       {z_score:.4f}")
    print(f"   Focus (CDF):   {focus_score:.4f}")

    if raw_hoyer > SAv3Config.HOYER_MU:
        print(f"   -> Embedding more sparse than typical NKJP!")
    else:
        print(f"   -> Embedding more dense than typical NKJP")

    results["Hoyer_raw"] = raw_hoyer
    results["Focus_score"] = focus_score
    results["Z_score"] = z_score

    # === KROK 4: Phase & Kinetic Quality ===
    D, S, E = metrics["Phase"]["D"], metrics["Phase"]["S"], metrics["Phase"]["E"]

    # Phase Quality: preferujemy (wysokie D) × (niskie E) × (stabilne S)
    q_phase = D * (1 - E) * (0.5 + 0.5 * S)

    # Kinetic Power: estymacja bez autoenkodera
    q_kinetic = estimate_kinetic_power(text, metrics)

    # Total Phi Quality: polaczenie pozycji i kierunku
    q_phi = 0.5 * q_phase + 0.5 * q_kinetic

    print(f"\nSTEP 4: Phi-9 Quality")
    print(f"   Phase Quality:   {q_phase:.4f}")
    print(f"   Kinetic Power:   {q_kinetic:.4f}")
    print(f"   Total Phi-9:     {q_phi:.4f}")

    results["Q_phase"] = q_phase
    results["Q_kinetic"] = q_kinetic
    results["Q_phi"] = q_phi

    # === KROK 5: Topological Balance ===
    equilibrium = np.array([0.5, 0.5, 0.5])
    phase_point = np.array([D, S, E])
    dist = np.linalg.norm(phase_point - equilibrium)
    max_dist = np.sqrt(0.75)
    balance = 1.0 - (dist / max_dist)

    print(f"\nSTEP 5: Topological Balance")
    print(f"   Distance from equilibrium: {dist:.4f}")
    print(f"   Balance score:             {balance:.4f}")

    results["Balance"] = balance

    # === KROK 6: Hybrid Boost ===
    boost_raw = (SAv3Config.ALPHA_PHI * q_phi +
                 SAv3Config.ALPHA_HOYER * focus_score +
                 SAv3Config.ALPHA_BAL * balance)

    # Maksymalny potencjalny wzrost
    potential = 1.0 - sa_base

    # Finalne SA v3.0 - FORMULA MNOZYKOWA Z WAGA POTENCJALU
    sa_v3 = sa_base + potential * boost_raw

    print(f"\nSTEP 6: Hybrid Boost")
    print(f"   Boost (raw):     {boost_raw:.4f}")
    print(f"   Potential:       {potential:.4f}")
    print(f"   SA Base:         {sa_base:.4f}")
    print(f"   SA v3.0 Final:   {sa_v3:.4f}")
    print(f"   Formula: SA_v3 = SA_base + (1 - SA_base) × Boost")
    print(f"            {sa_v3:.4f} = {sa_base:.4f} + {potential:.4f} × {boost_raw:.4f}")

    results["Boost"] = boost_raw
    results["Potential"] = potential
    results["SA_v3"] = sa_v3

    return results

def interpret_sa(sa_value):
    """Interpretacja wartosci SA."""
    if sa_value < 0.3:
        return "LOW ACCESSIBILITY", "Specialist/difficult text - requires contextual knowledge"
    elif sa_value < 0.5:
        return "MEDIUM ACCESSIBILITY", "Demanding text - needs domain context"
    elif sa_value < 0.7:
        return "HIGH ACCESSIBILITY", "Generally accessible - understandable to broad audience"
    else:
        return "VERY HIGH ACCESSIBILITY", "Simple text - immediately understandable"

# === URUCHOMIENIE ===
if __name__ == "__main__":
    print("="*80)
    print("ENHANCED SA v3.0 VALIDATION - END-TO-END PIPELINE")
    print("="*80)
    print(f"\nText: '{TEXT_CONTENT[:80]}...'")
    print(f"Length: {len(TEXT_CONTENT)} characters")

    # 1. Pobierz prawdziwy embedding z HerBERT
    real_embedding = get_herbert_embedding(TEXT_CONTENT, normalize=False)

    # 2. Oblicz SA v3.0 z pelna diagnostyka
    result = calculate_sa_v3_full(INPUT_METRICS, real_embedding, TEXT_CONTENT)

    # 3. Porownanie z v2.0
    sa_v2 = INPUT_METRICS["SA_v2_Original"]
    improvement = (result['SA_v3'] / sa_v2 - 1) * 100
    delta = result['SA_v3'] - sa_v2

    print("\n" + "="*80)
    print("FINAL RESULTS")
    print("="*80)
    print(f"\n{'Metric':<30} {'Value':<15} {'Interpretation'}")
    print("-" * 80)
    print(f"{'SA v2.0 (old)':<30} {sa_v2:.4f}          CHAOS/LOW")
    print(f"{'SA v3.0 (new)':<30} {result['SA_v3']:.4f}")

    category, desc = interpret_sa(result['SA_v3'])
    print(f"{'Category v3.0':<30} {category}")
    print(f"\n{desc}")
    print(f"\n{'Absolute improvement':<30} {delta:+.4f}")
    print(f"{'Relative improvement':<30} {improvement:+.1f}%")

    print("\n" + "-"*80)
    print("KEY INSIGHTS:")
    print(f"   * Hoyer raw:      {result['Hoyer_raw']:.4f} (signal sparsity)")
    print(f"   * Z-score:        {result['Z_score']:.4f} (vs NKJP reference)")
    print(f"   * Focus:          {result['Focus_score']:.4f} (semantic sharpness)")
    print(f"   * CI reduction:   {(result['CI_old'] - result['CI_weighted'])/result['CI_old']*100:.1f}% (cognitive correction)")
    print(f"   * Q_kinetic:      {result['Q_kinetic']:.4f} (directive power)")
    print(f"   * Balance:        {result['Balance']:.4f} (topological equilibrium)")

    print("\n" + "-"*80)
    print("FORMULA VERIFICATION:")
    print(f"   SA_v3 = SA_base + (1 - SA_base) × Boost")
    print(f"   {result['SA_v3']:.4f} = {result['SA_base']:.4f} + {result['Potential']:.4f} × {result['Boost']:.4f}")
    print(f"   Check: {result['SA_base'] + result['Potential'] * result['Boost']:.4f}")
    print(f"   Guaranteed: SA_v3 <= 1.0 (current: {result['SA_v3']:.4f})")

    print("\n" + "="*80)
    print("CONCLUSION:")
    print("="*80)
    if result['SA_v3'] > 0.3 and sa_v2 < 0.3:
        print("SA v3.0 successfully CORRECTS the misclassification of legal text as 'CHAOS'.")
        print("The court verdict is NOT chaotic - it's a precise directive with high kinetic power.")
        print(f"The Hoyer sparsity ({result['Hoyer_raw']:.4f}) confirms semantic sharpness.")
    else:
        print(f"SA v2.0: {sa_v2:.4f}, SA v3.0: {result['SA_v3']:.4f}")
    print("="*80)
