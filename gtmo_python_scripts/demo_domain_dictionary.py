#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
DEMO: GTMØ Domain Dictionary Usage
===================================
Przykład wykorzystania słowników domenowych z poprzednich analiz
do poprawy jakości analizy morfosyntaktycznej nowych tekstów.
"""

import sys
if sys.platform == 'win32':
    import io
    if not isinstance(sys.stdout, io.TextIOWrapper) or sys.stdout.encoding != 'utf-8':
        sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
    if not isinstance(sys.stderr, io.TextIOWrapper) or sys.stderr.encoding != 'utf-8':
        sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8')

from gtmo_domain_dictionary import DomainDictionary
from gtmo_morphosyntax import QuantumMorphosyntaxEngine


def demo_basic_dictionary_loading():
    """Demo 1: Podstawowe ładowanie słownika z wyników analiz."""
    print("=" * 80)
    print("DEMO 1: Ładowanie słownika domenowego z analiz Konstytucji RP")
    print("=" * 80)

    # Inicjalizacja słownika domenowego
    dd = DomainDictionary(results_dir="gtmo_results")

    # Załaduj słownik z analiz Konstytucji
    stats = dd.load_domain_from_results(
        pattern="KONSTYTUCJA_RP_sentence_*.json",
        domain_name="constitutional_law"
    )

    print(f"\n✓ Słownik załadowany:")
    print(f"  - Domena: {stats['domain_name']}")
    print(f"  - Plików przeanalizowanych: {stats['total_files']}")
    print(f"  - Unikalnych lemm: {stats['unique_lemmas']}")
    print(f"  - Całkowita liczba tokenów: {stats['total_tokens']}")

    # Pokaż najczęstsze terminy
    print("\n" + "=" * 80)
    print("Top 30 terminów domenowych (prawo konstytucyjne):")
    print("=" * 80)

    for i, (term, freq) in enumerate(dd.get_most_common_terms(30), 1):
        info = dd.get_term_info(term)
        pos_tags = ', '.join(sorted(info['pos_tags'])) if info else 'N/A'
        print(f"{i:2}. {term:25} freq={freq:4}  POS: {pos_tags}")

    return dd


def demo_term_analysis(dd: DomainDictionary):
    """Demo 2: Szczegółowa analiza wybranych terminów."""
    print("\n" + "=" * 80)
    print("DEMO 2: Szczegółowa analiza terminów domenowych")
    print("=" * 80)

    # Przeanalizuj kilka kluczowych terminów
    key_terms = ["RZECZYPOSPOLITEJ", "KONSTYTUCJA", "USTAWA", "SEJM", "OBYWATEL"]

    for term in key_terms:
        info = dd.get_term_info(term)
        if info:
            print(f"\n--- {term} ---")
            print(f"  Częstość występowania: {info['frequency']}")
            print(f"  POS tags: {', '.join(sorted(info['pos_tags']))}")
            print(f"  Case tags: {', '.join(sorted(info['case_tags']))}")
            print(f"  Przykładowe pełne tagi:")
            for tag in info['full_tags'][:3]:
                print(f"    • {tag}")

            if info['contexts']:
                print(f"  Przykładowy kontekst:")
                print(f"    \"{info['contexts'][0][:80]}...\"")


def demo_domain_filtering():
    """Demo 3: Filtrowanie analiz morfologicznych używając słownika."""
    print("\n" + "=" * 80)
    print("DEMO 3: Filtrowanie analiz morfologicznych z użyciem słownika domenowego")
    print("=" * 80)

    # Załaduj słownik
    dd = DomainDictionary(results_dir="gtmo_results")
    dd.load_domain_from_results(
        pattern="KONSTYTUCJA_RP_sentence_*.json",
        domain_name="constitutional_law"
    )

    # Inicjalizuj silnik z słownikiem domenowym
    print("\n✓ Inicjalizacja silnika GTMØ ze słownikiem domenowym...")
    engine = QuantumMorphosyntaxEngine(domain_dictionary=dd)

    # Testowy tekst z domeny prawa konstytucyjnego
    test_text = "Rzeczpospolita Polska jest demokratycznym państwem prawnym."

    print(f"\nTestowy tekst: \"{test_text}\"")
    print("\nAnaliza morfosyntaktyczna z wykorzystaniem słownika domenowego:")
    print("-" * 80)

    try:
        # Analiza z filtrowaniem domenowym
        morph_coords, morph_meta, quantum_states = engine.analyze_morphology_quantum(test_text)

        print(f"\n✓ Analiza zakończona:")
        print(f"  - Współrzędne morfologiczne: D={morph_coords[0]:.4f}, S={morph_coords[1]:.4f}, E={morph_coords[2]:.4f}")
        print(f"  - Liczba analiz: {morph_meta['total_analyses']}")
        print(f"  - Ambiguity: {morph_meta['ambiguity']:.2f}")
        print(f"  - Słowa kwantowe: {morph_meta['quantum_words']}")

        print(f"\n  Wykryte przypadki:")
        for case, count in sorted(morph_meta['cases'].items(), key=lambda x: x[1], reverse=True):
            print(f"    • {case}: {count}")

        print(f"\n  Tagi (próbka):")
        for tag in morph_meta['debug_tags'][:10]:
            print(f"    • {tag}")

    except Exception as e:
        print(f"✗ Błąd podczas analizy: {e}")


def demo_export_dictionary():
    """Demo 4: Eksport słownika do pliku JSON."""
    print("\n" + "=" * 80)
    print("DEMO 4: Eksport słownika domenowego do pliku")
    print("=" * 80)

    dd = DomainDictionary(results_dir="gtmo_results")
    dd.load_domain_from_results(
        pattern="KONSTYTUCJA_RP_sentence_*.json",
        domain_name="constitutional_law"
    )

    output_file = "domain_dict_constitutional_law.json"
    dd.export_dictionary(output_file)

    print(f"\n✓ Słownik wyeksportowany do: {output_file}")
    print(f"  Ten plik może być używany jako:")
    print(f"  - Referencja terminologii domenowej")
    print(f"  - Dane treningowe dla modeli ML")
    print(f"  - Baza wiedzy dla systemu eksperckiego")


def demo_multi_domain():
    """Demo 5: Praca z wieloma domenami."""
    print("\n" + "=" * 80)
    print("DEMO 5: Obsługa wielu domen jednocześnie")
    print("=" * 80)

    # Słownik 1: Konstytucja (prawo konstytucyjne)
    dd_constitutional = DomainDictionary(results_dir="gtmo_results")
    stats1 = dd_constitutional.load_domain_from_results(
        pattern="KONSTYTUCJA_RP_sentence_*.json",
        domain_name="constitutional_law"
    )

    # Możliwość załadowania innych domen (jeśli istnieją)
    # dd_vat = DomainDictionary(results_dir="gtmo_results")
    # dd_vat.load_domain_from_results(pattern="VAT_*.json", domain_name="tax_law")

    print("\n✓ Załadowane domeny:")
    print(f"  1. {stats1['domain_name']}: {stats1['unique_lemmas']} lemm")

    print("\nMożliwości zastosowania:")
    print("  - Automatyczna detekcja domeny tekstu")
    print("  - Przełączanie kontekstów analizy")
    print("  - Budowanie ontologii międzydomenowych")


def demo_domain_term_detection():
    """Demo 6: Detekcja terminów domenowych w nowym tekście."""
    print("\n" + "=" * 80)
    print("DEMO 6: Detekcja terminów domenowych w nowym tekście")
    print("=" * 80)

    dd = DomainDictionary(results_dir="gtmo_results")
    dd.load_domain_from_results(
        pattern="KONSTYTUCJA_RP_sentence_*.json",
        domain_name="constitutional_law"
    )

    # Nowy tekst do analizy
    new_text = """
    Sejm Rzeczypospolitej Polskiej uchwala ustawy. Konstytucja określa podstawowe
    zasady ustrojowe państwa. Obywatele mają prawo do sprawiedliwego procesu.
    """

    words = new_text.split()

    print(f"\nNowy tekst do analizy:")
    print(new_text.strip())

    print("\n" + "-" * 80)
    print("Wykryte terminy domenowe:")
    print("-" * 80)

    domain_terms_found = []
    for word in words:
        cleaned_word = word.strip('.,!?;:')
        if dd.is_domain_term(cleaned_word, min_frequency=2):
            term_info = dd.get_term_info(cleaned_word.upper())
            if term_info:
                domain_terms_found.append((cleaned_word, term_info['frequency']))
                print(f"  ✓ {cleaned_word:20} (freq={term_info['frequency']}, "
                      f"POS={', '.join(list(term_info['pos_tags'])[:2])})")

    print(f"\n✓ Znaleziono {len(domain_terms_found)} terminów domenowych w tekście")
    print(f"  Wskaźnik specjalizacji: {len(domain_terms_found)/len(words)*100:.1f}%")


def main():
    """Główna funkcja demonstracyjna."""
    print("\n")
    print("╔" + "=" * 78 + "╗")
    print("║" + " " * 15 + "GTMØ DOMAIN DICTIONARY - DEMONSTRACJA" + " " * 24 + "║")
    print("╚" + "=" * 78 + "╝")

    try:
        # Demo 1: Podstawowe ładowanie
        dd = demo_basic_dictionary_loading()

        # Demo 2: Analiza terminów
        demo_term_analysis(dd)

        # Demo 3: Filtrowanie morfologiczne
        demo_domain_filtering()

        # Demo 4: Eksport
        demo_export_dictionary()

        # Demo 5: Wiele domen
        demo_multi_domain()

        # Demo 6: Detekcja terminów
        demo_domain_term_detection()

        print("\n" + "=" * 80)
        print("PODSUMOWANIE")
        print("=" * 80)
        print("""
✓ Słowniki domenowe z gtmo_results są teraz aktywne!

Jak używać w swoich analizach:

1. Załaduj słownik domenowy:
   ```python
   from gtmo_domain_dictionary import DomainDictionary
   dd = DomainDictionary()
   dd.load_domain_from_results(pattern="TWOJA_DOMENA_*.json")
   ```

2. Użyj z silnikiem GTMØ:
   ```python
   from gtmo_morphosyntax import QuantumMorphosyntaxEngine
   engine = QuantumMorphosyntaxEngine(domain_dictionary=dd)
   coords, meta, states = engine.analyze_morphology_quantum(text)
   ```

3. Korzyści:
   - Redukcja ambiguity morfosyntaktycznej
   - Priorytetyzacja interpretacji domenowych
   - Poprawa jakości analizy specjalistycznych tekstów
   - Budowa korpusów specjalistycznych
        """)

    except FileNotFoundError as e:
        print(f"\n✗ Błąd: {e}")
        print("Upewnij się, że folder 'gtmo_results' zawiera pliki analiz JSON.")
    except Exception as e:
        print(f"\n✗ Nieoczekiwany błąd: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
