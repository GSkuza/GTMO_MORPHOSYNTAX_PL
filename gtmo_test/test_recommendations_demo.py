#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Demo systemu rekomendacji w języku naturalnym
Pokazuje jak działa dla różnych typów problemów
"""

import sys
import os
sys.path.insert(0, 'gtmo_results_analyse')

from gtmo_verdict_analyzer import NaturalLanguageRecommendations

# Przykładowe złe zdania z różnymi problemami
test_sentences = [
    {
        'name': 'PROBLEM MORFOLOGICZNY (skomplikowane słownictwo)',
        'text': 'Organ właściwy w rozumieniu przepisów ustawy z dnia 5 czerwca 1998 r. o samorządzie wojewódzkim w przypadku gdy wnioskodawca spełnia przesłanki określone w przepisach wydanych na podstawie delegacji ustawowej zawartej w art. 5 ust. 2 pkt 3 rozporządzenia wykonawczego...',
        'SA': 0.23,
        'CI_morph_pct': 55,
        'CI_synt_pct': 25,
        'CI_sem_pct': 20,
        'ambiguity': 3.8,
        'depth': 9,
        'classification': 'CHAOTIC_STRUCTURE'
    },
    {
        'name': 'PROBLEM SKŁADNIOWY (za długie zdanie)',
        'text': 'W przypadku, gdy podmiot, o którym mowa w ust. 1, działając w ramach swoich kompetencji określonych w przepisach ustawy z dnia 15 grudnia 2016 r., w terminie 30 dni od dnia złożenia wniosku, o którym mowa w art. 10 ust. 2, stwierdzi, że zostały spełnione wszystkie przesłanki, które zostały wymienione w rozporządzeniu Ministra, o którym mowa w ust. 3, wydaje decyzję administracyjną.',
        'SA': 0.18,
        'CI_morph_pct': 25,
        'CI_synt_pct': 60,
        'CI_sem_pct': 15,
        'ambiguity': 4.5,
        'depth': 15,
        'classification': 'CHAOTIC_STRUCTURE'
    },
    {
        'name': 'PROBLEM SEMANTYCZNY (niejasne znaczenia)',
        'text': 'Właściwy organ może wydać stosowną decyzję w odpowiednim terminie, jeżeli zachodzą okoliczności uzasadniające takie działanie zgodnie z obowiązującymi przepisami.',
        'SA': 0.25,
        'CI_morph_pct': 20,
        'CI_synt_pct': 20,
        'CI_sem_pct': 60,
        'ambiguity': 5.2,
        'depth': 8,
        'classification': 'CHAOTIC_STRUCTURE'
    }
]

def main():
    print("=" * 80)
    print("DEMO: SYSTEM REKOMENDACJI W JĘZYKU NATURALNYM")
    print("=" * 80)
    print()

    # Inicjalizuj z LLM
    api_key = os.getenv('ANTHROPIC_API_KEY')
    use_llm = bool(api_key)

    if use_llm:
        print("✅ LLM włączony (Claude API)")
    else:
        print("⚠️  LLM wyłączony (brak API key) - używam tylko szablonów")

    print()

    recommender = NaturalLanguageRecommendations(use_llm=use_llm, api_key=api_key)

    for i, sentence in enumerate(test_sentences, 1):
        print("=" * 80)
        print(f"PRZYKŁAD {i}: {sentence['name']}")
        print("=" * 80)
        print()

        print("ORYGINALNY TEKST:")
        print(f'"{sentence["text"]}"')
        print()

        print(f"METRYKI (dla kontekstu - nie pokazywane userowi):")
        print(f"  SA: {sentence['SA']*100:.1f}%")
        print(f"  CI decomposition: morph={sentence['CI_morph_pct']:.0f}%, "
              f"synt={sentence['CI_synt_pct']:.0f}%, sem={sentence['CI_sem_pct']:.0f}%")
        print(f"  Ambiguity: {sentence['ambiguity']:.1f}")
        print(f"  Depth: {sentence['depth']}")
        print()

        # Generuj rekomendacje
        rec = recommender.generate_recommendations(sentence)

        print("─" * 80)
        print("REKOMENDACJE (JĘZYK NATURALNY - BEZ ŻARGONU):")
        print("─" * 80)
        print()

        print(f"PROBLEM:")
        print(f"Ten przepis jest {rec['severity']}.")
        print(f"Główny problem: {rec['main_problem_detailed']}")
        print()

        print(f"CO ZROBIĆ TERAZ (proste poprawki):")
        for j, fix in enumerate(rec['quick_fixes'], 1):
            print(f"  {j}. {fix}")
        print()

        print(f"CO ZROBIĆ DŁUGOTERMINOWO:")
        for j, fix in enumerate(rec['long_term_fixes'], 1):
            print(f"  {j}. {fix}")
        print()

        print(f"PRZYKŁAD LEPSZEJ WERSJI:")
        print(rec['example_better_version'])
        print()

        print(f"RYZYKO PRAWNE:")
        print(rec['legal_risks'])
        print()
        print()

    print("=" * 80)
    print("✅ DEMO ZAKOŃCZONE")
    print("=" * 80)

if __name__ == "__main__":
    main()
