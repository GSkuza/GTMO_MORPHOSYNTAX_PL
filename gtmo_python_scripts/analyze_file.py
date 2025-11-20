#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Skrypt do analizy plików tekstowych/markdown z użyciem GTMØ + Stanza
Generuje JSON w formacie zgodnym z example_stanza_output.json
"""

import sys
import json
import os
from pathlib import Path

# Dodaj ścieżkę do podmodułu GTMO_MORPHOSYNTAX_PL
script_dir = Path(__file__).parent
gtmo_module_path = script_dir / "GTMO_MORPHOSYNTAX_PL"
sys.path.insert(0, str(gtmo_module_path))

from gtmo_morphosyntax import EnhancedGTMOProcessor

def load_text_file(file_path: str) -> str:
    """Wczytuje plik tekstowy (txt, md, itp.)"""
    with open(file_path, 'r', encoding='utf-8') as f:
        return f.read()

def analyze_file(input_path: str, output_path: str = None):
    """
    Analizuje plik i zapisuje wynik do JSON.

    Args:
        input_path: Ścieżka do pliku wejściowego (.txt, .md)
        output_path: Ścieżka do pliku wyjściowego JSON (opcjonalnie)
    """
    # Sprawdź czy plik istnieje
    if not os.path.exists(input_path):
        print(f"❌ Plik nie istnieje: {input_path}")
        return

    print(f"📖 Wczytuję plik: {input_path}")
    text = load_text_file(input_path)

    print(f"📏 Długość tekstu: {len(text)} znaków")
    print(f"🔧 Inicjalizuję EnhancedGTMOProcessor...")

    processor = EnhancedGTMOProcessor()

    print(f"⚙️  Analizuję tekst (to może potrwać kilka sekund)...")
    result = processor.analyze_legal_text(text)

    # Jeśli nie podano ścieżki wyjściowej, utwórz ją na podstawie wejściowej
    if output_path is None:
        input_file = Path(input_path)
        output_path = input_file.parent / f"{input_file.stem}_stanza_analysis.json"

    print(f"💾 Zapisuję wynik do: {output_path}")
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(result, f, ensure_ascii=False, indent=2)

    # Pokaż krótkie podsumowanie
    print("\n" + "="*60)
    print("📊 PODSUMOWANIE ANALIZY")
    print("="*60)

    gtmo_coords = result.get('gtmo_coordinates', {})
    print(f"\n🎯 Współrzędne GTMØ:")
    print(f"   Determination: {gtmo_coords.get('determination', 0):.3f}")
    print(f"   Stability:     {gtmo_coords.get('stability', 0):.3f}")
    print(f"   Entropy:       {gtmo_coords.get('entropy', 0):.3f}")

    stanza = result.get('stanza_analysis', {})
    if stanza:
        smoking_guns = stanza.get('smoking_guns', [])
        print(f"\n🔫 Smoking Guns: {len(smoking_guns)}")
        for i, gun in enumerate(smoking_guns[:3], 1):  # Pokaż max 3
            print(f"   {i}. {gun.get('type', 'unknown')}: severity {gun.get('severity', 0):.2f}")
            if 'details' in gun and 'conflict' in gun['details']:
                print(f"      → {gun['details']['conflict']}")

        if len(smoking_guns) > 3:
            print(f"   ... i {len(smoking_guns) - 3} więcej")

    legal = result.get('legal_assessment', {})
    if legal:
        print(f"\n⚖️  Ocena Prawna:")
        print(f"   Quality: {legal.get('quality', 'unknown')}")
        print(f"   Legal Coherence: {legal.get('legal_coherence_score', 0):.3f}")
        print(f"   Smoking Gun Count: {legal.get('smoking_gun_count', 0)}")

        issues = legal.get('issues', [])
        if issues:
            print(f"   Issues: {len(issues)}")
            for issue in issues[:2]:  # Pokaż max 2
                print(f"      - {issue.get('category', 'unknown')}: {issue.get('severity', 'unknown')}")

    causality = result.get('causality_analysis', {})
    if causality:
        causal_strength = causality.get('causal_strength', 0)
        print(f"\n🔗 Analiza Rozumowania:")
        print(f"   Causal Strength (Reasoning Quality): {causal_strength:.3f}")

        broken_chains = causality.get('broken_chains', [])
        if broken_chains:
            print(f"   Broken Causal Chains: {len(broken_chains)}")

        circular = causality.get('circular_reasoning', [])
        if circular:
            print(f"   Circular Reasoning: {len(circular)} detected")

    singularity = result.get('singularity_warning', {})
    if singularity and singularity.get('active'):
        print(f"\n⚠️  OSTRZEŻENIE SINGULARNOŚCI!")
        print(f"   Severity: {singularity.get('severity', 'unknown')}")
        print(f"   Message: {singularity.get('message', '')}")

    print("\n" + "="*60)
    print(f"✅ Analiza zakończona. Wynik zapisany w: {output_path}")
    print("="*60)

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Użycie:")
        print(f"  python {sys.argv[0]} <ścieżka_do_pliku> [ścieżka_wyjściowa.json]")
        print()
        print("Przykład:")
        print(f'  python {sys.argv[0]} "C:\\Users\\grzeg\\Desktop\\projekt_poselski_edited.md"')
        print(f'  python {sys.argv[0]} "plik.txt" "wynik.json"')
        sys.exit(1)

    input_file = sys.argv[1]
    output_file = sys.argv[2] if len(sys.argv) > 2 else None

    analyze_file(input_file, output_file)
