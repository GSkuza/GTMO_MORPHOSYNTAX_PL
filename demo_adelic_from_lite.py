#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Demo: Użycie wyników z --lite lub --standard w gtmo_adelic_layer
"""

import sys
import io
# Fix Windows console encoding
if sys.platform == 'win32':
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

import json
import numpy as np
from pathlib import Path
from gtmo_adelic_layer import AdelicSemanticLayer

# Ścieżka do pliku full_document.json
# WAŻNE: Użyj r"..." (raw string) dla Windows paths aby uniknąć escape sequences!
RESULTS_FILE = Path(r"D:\GTMO_MORPHOSYNTAX\gtmo_results\analysis_25112025_no1_dataset_test\full_document.json")

def main():
    # Inicjalizuj warstwę adeliczną z V_Comm thresholds (skalibrowane na podstawie analizy)
    # EMERGED: V_Comm < 100, BORDERLINE: 100-150, ALIENATED: ≥ 150
    adelic = AdelicSemanticLayer(
        use_energy_threshold=True,
        energy_threshold_emerged=100.0,
        energy_threshold_borderline=150.0
    )
    print(f"✅ Adelic Layer initialized with {len(adelic.observers)} observers")
    print(f"   V_Comm thresholds: EMERGED<100, BORDERLINE<150, ALIENATED≥150\n")

    # Sprawdź czy plik istnieje
    if not RESULTS_FILE.exists():
        print(f"❌ File not found: {RESULTS_FILE}")
        print(f"   Please check the path and make sure the file exists.")
        return

    # Wczytaj full_document.json
    print(f"📂 Loading: {RESULTS_FILE.name}")
    with open(RESULTS_FILE, 'r', encoding='utf-8') as f:
        full_doc = json.load(f)

    # Wyciągnij analizy zdań z dokumentu
    analyses = []

    if 'sentences' in full_doc:
        # Dla sentence-based analysis (bezpośrednia lista zdań)
        analyses = full_doc['sentences']
        print(f"📄 Found {len(analyses)} sentence analyses (sentence-based)\n")
    elif 'articles' in full_doc:
        # Dla article-based analysis - spłaszcz zdania z artykułów
        for article in full_doc['articles']:
            if 'sentences' in article:
                analyses.extend(article['sentences'])
        print(f"📄 Found {len(analyses)} sentence analyses from articles\n")
    else:
        print("❌ No sentences or articles found in document")
        print(f"   Available keys: {list(full_doc.keys())}")
        return

    if not analyses:
        print("❌ No analyses to process")
        return

    # Analizuj każde zdanie z warstwą adeliczną
    for idx, data in enumerate(analyses, 1):
        # Wyciągnij potrzebne informacje
        text = data.get('text', '')
        coords = data.get('coordinates', {})

        if not coords:
            print(f"⚠️  Skipping sentence {idx} - no coordinates")
            continue

        # Konwertuj współrzędne do np.ndarray
        base_coords = np.array([
            coords['determination'],
            coords['stability'],
            coords['entropy']
        ])

        print(f"📝 Sentence {idx}: {text[:80]}{'...' if len(text) > 80 else ''}")
        print(f"   Base D-S-E: [{base_coords[0]:.4f}, {base_coords[1]:.4f}, {base_coords[2]:.4f}]")

        # Wyświetl dodatkowe metryki jeśli są dostępne (tryb standard)
        if 'ambiguity' in data:
            print(f"   Ambiguity: {data['ambiguity']:.4f}")
        if 'depth' in data:
            print(f"   Depth: {data['depth']}")
        if 'geometric_balance' in data:
            print(f"   Geometric Balance: {data['geometric_balance']:.6f}")
        if 'geometric_tension' in data:
            print(f"   Geometric Tension: {data['geometric_tension']:.6f}")

        # Analiza adeliczna
        result = adelic.analyze_with_observers(
            text=text,
            base_coords=base_coords,
            metric='phi9'
        )

        # Wyświetl wyniki adeliczne
        status = result['status']
        if status == 'emerged':
            status_icon = "✨"
        elif status == 'borderline':
            status_icon = "🟡"
        else:
            status_icon = "⚠️"

        print(f"   🔮 Adelic Analysis:")
        print(f"      Status: {status_icon} {status.upper()}")
        print(f"      V_Comm Energy: {result['synchronization_energy']:.1f}")
        print(f"      Observers: {result['n_observers']}")

        if result['emerged']:
            global_val = result['global_value']
            print(f"      Global φ_∞: [{global_val[0]:.4f}, {global_val[1]:.4f}, {global_val[2]:.4f}]")
        elif status == 'borderline':
            print(f"      ⚠️ Borderline - uncertain semantic convergence")
        else:
            print(f"      ❌ Alienated - no semantic consensus")

        print()

    print(f"✅ Processed {len(analyses)} sentences")

if __name__ == "__main__":
    main()
