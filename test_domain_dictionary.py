#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Quick test for domain dictionary functionality"""

import sys
if sys.platform == 'win32':
    import io
    if not isinstance(sys.stdout, io.TextIOWrapper) or sys.stdout.encoding != 'utf-8':
        sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

from gtmo_domain_dictionary import DomainDictionary

print("=" * 70)
print("Test słownika domenowego GTMØ")
print("=" * 70)

# Test 1: Załaduj słownik
print("\n[1/3] Ładowanie słownika z analiz Konstytucji...")
dd = DomainDictionary(results_dir="gtmo_results")
stats = dd.load_domain_from_results(
    pattern="KONSTYTUCJA_RP_sentence_[1-9].json",  # Tylko pierwsze 9 dla szybkiego testu
    domain_name="constitutional_law_test"
)

print(f"\nStatystyki:")
print(f"  Domena: {stats['domain_name']}")
print(f"  Plików: {stats['total_files']}")
print(f"  Lemm: {stats['unique_lemmas']}")

# Test 2: Top terminy
print("\n[2/3] Top 10 terminów:")
for i, (term, freq) in enumerate(dd.get_most_common_terms(10), 1):
    print(f"  {i:2}. {term:20} freq={freq}")

# Test 3: Analiza wybranego terminu
print("\n[3/3] Szczegóły dla 'RZECZYPOSPOLITEJ':")
info = dd.get_term_info("RZECZYPOSPOLITEJ")
if info:
    print(f"  Częstość: {info['frequency']}")
    print(f"  POS: {', '.join(info['pos_tags'])}")
    print(f"  Cases: {', '.join(info['case_tags'])}")
    print(f"  Przykładowy tag: {info['full_tags'][0] if info['full_tags'] else 'N/A'}")

print("\n" + "=" * 70)
print("✓ Test zakończony pomyślnie!")
print("=" * 70)
