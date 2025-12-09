#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Demo: Użycie wyników z selektywnymi obserwatorami (bez ekstremalnych)
"""

import sys
import io
if sys.platform == 'win32':
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

import json
import numpy as np
from pathlib import Path
from gtmo_adelic_layer import AdelicSemanticLayer, Observer

# Ścieżka do pliku
RESULTS_FILE = Path(r"D:\GTMO_MORPHOSYNTAX\gtmo_results\analysis_25112025_no1_dataset_test\full_document.json")

def create_moderate_observers():
    """
    Tworzy zestaw umiarkowanych obserwatorów (bez ekstremalnych jak hallucination, nonsense).
    """
    return [
        Observer(
            id="O_formal",
            interpretation_bias=np.array([0.10, 0.08, -0.06]),
            coherence_threshold=0.85,
            topology_metric='euclidean',
            register='formal'
        ),
        Observer(
            id="O_legal",
            interpretation_bias=np.array([0.15, 0.12, -0.08]),
            coherence_threshold=0.88,
            topology_metric='euclidean',
            register='legal'
        ),
        Observer(
            id="O_casual",
            interpretation_bias=np.array([0.05, 0.00, 0.05]),
            coherence_threshold=0.70,
            topology_metric='euclidean',
            register='casual'
        ),
        Observer(
            id="O_journalistic",
            interpretation_bias=np.array([0.08, 0.05, 0.08]),
            coherence_threshold=0.75,
            topology_metric='euclidean',
            register='journalistic'
        ),
        Observer(
            id="O_technical",
            interpretation_bias=np.array([0.20, 0.18, -0.12]),
            coherence_threshold=0.92,
            topology_metric='euclidean',
            register='technical'
        )
    ]

def main():
    # Inicjalizuj z umiarkowanymi obserwatorami i V_Comm thresholds
    moderate_observers = create_moderate_observers()
    adelic = AdelicSemanticLayer(
        default_observers=moderate_observers,
        use_energy_threshold=True,
        energy_threshold_emerged=80.0,   # Niższe progi dla mniejszej liczby obserwatorów
        energy_threshold_borderline=120.0
    )

    print(f"✅ Adelic Layer (SELECTIVE) with {len(adelic.observers)} moderate observers")
    print(f"   V_Comm thresholds: EMERGED<80, BORDERLINE<120, ALIENATED≥120")
    print()

    if not RESULTS_FILE.exists():
        print(f"❌ File not found: {RESULTS_FILE}")
        return

    with open(RESULTS_FILE, 'r', encoding='utf-8') as f:
        full_doc = json.load(f)

    # Wyciągnij analizy
    analyses = []
    if 'sentences' in full_doc:
        analyses = full_doc['sentences']
    elif 'articles' in full_doc:
        for article in full_doc['articles']:
            if 'sentences' in article:
                analyses.extend(article['sentences'])

    if not analyses:
        print("❌ No analyses found")
        return

    print(f"📄 Processing {len(analyses)} sentences\n")

    emergences = 0
    total = 0

    for idx, data in enumerate(analyses, 1):
        text = data.get('text', '')
        coords = data.get('coordinates', {})

        if not coords:
            continue

        total += 1

        base_coords = np.array([
            coords['determination'],
            coords['stability'],
            coords['entropy']
        ])

        # Analiza adeliczna
        result = adelic.analyze_with_observers(
            text=text,
            base_coords=base_coords,
            metric='phi9'
        )

        emerged = result['emerged']
        status = result['status']
        if emerged:
            emergences += 1

        # Ikony statusu
        if status == 'emerged':
            status_icon = "✨"
        elif status == 'borderline':
            status_icon = "🟡"
        else:
            status_icon = "⚠️"

        print(f"{status_icon} Sentence {idx}: {text[:60]}...")
        print(f"   D-S-E: [{base_coords[0]:.3f}, {base_coords[1]:.3f}, {base_coords[2]:.3f}]")
        print(f"   Status: {status.upper()}, V_Comm Energy: {result['synchronization_energy']:.1f}")

        if emerged:
            gv = result['global_value']
            print(f"   Global φ_∞: [{gv[0]:.3f}, {gv[1]:.3f}, {gv[2]:.3f}]")

        print()

    print(f"📊 Summary:")
    print(f"   Total: {total}")
    print(f"   Emerged: {emergences} ({100*emergences/total if total > 0 else 0:.1f}%)")
    print(f"   Alienated: {total - emergences} ({100*(total-emergences)/total if total > 0 else 0:.1f}%)")

if __name__ == "__main__":
    main()
