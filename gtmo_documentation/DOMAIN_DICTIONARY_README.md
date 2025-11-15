# GTMØ Domain Dictionary - Instrukcja Użycia

## Wprowadzenie

System słowników domenowych GTMØ pozwala na wykorzystanie poprzednich analiz morfosyntaktycznych jako bazy wiedzy dla nowych analiz. Pliki JSON w folderze `gtmo_results` automatycznie stają się słownikami specjalistycznymi, które:

- **Redukują ambiguity** morfosyntaktyczną poprzez priorytetyzację interpretacji domenowych
- **Poprawiają jakość** analizy tekstów specjalistycznych
- **Budują korpusy** terminologii specjalistycznej
- **Umożliwiają transfer wiedzy** między analizami

---

## Szybki Start

### 1. Podstawowe użycie

```python
from gtmo_domain_dictionary import DomainDictionary

# Załaduj słownik z analiz Konstytucji RP
dd = DomainDictionary(results_dir="gtmo_results")
dd.load_domain_from_results(
    pattern="KONSTYTUCJA_RP_*.json",
    domain_name="constitutional_law"
)

# Wyświetl statystyki
print(f"Załadowano {dd.domain_stats['unique_lemmas']} unikalnych terminów")
```

### 2. Integracja z silnikiem GTMØ

```python
from gtmo_morphosyntax import QuantumMorphosyntaxEngine
from gtmo_domain_dictionary import DomainDictionary

# Załaduj słownik
dd = DomainDictionary()
dd.load_domain_from_results(pattern="KONSTYTUCJA_RP_*.json")

# Utwórz silnik ze słownikiem domenowym
engine = QuantumMorphosyntaxEngine(domain_dictionary=dd)

# Analizuj tekst z użyciem wiedzy domenowej
text = "Rzeczpospolita Polska jest demokratycznym państwem prawnym."
coords, metadata, states = engine.analyze_morphology_quantum(text)
```

### 3. Szybki test

```bash
python test_domain_dictionary.py
```

---

## Funkcje Główne

### Ładowanie Słownika

```python
dd = DomainDictionary(results_dir="gtmo_results")

# Załaduj wszystkie pliki pasujące do wzorca
stats = dd.load_domain_from_results(
    pattern="KONSTYTUCJA_RP_*.json",  # Wzorzec glob
    domain_name="constitutional_law"  # Nazwa domeny
)

# Statystyki
print(stats)
# {
#   'total_files': 882,
#   'unique_lemmas': 1298,
#   'total_tokens': 34685,
#   'domain_name': 'constitutional_law'
# }
```

### Analiza Terminów Domenowych

```python
# Pobierz najczęstsze terminy
top_terms = dd.get_most_common_terms(n=50)
for lemma, frequency in top_terms:
    print(f"{lemma}: {frequency}")

# Szczegółowe informacje o terminie
info = dd.get_term_info("RZECZYPOSPOLITEJ")
print(f"POS tags: {info['pos_tags']}")        # {'subst'}
print(f"Case tags: {info['case_tags']}")      # {'gen', 'dat', 'loc'}
print(f"Frequency: {info['frequency']}")      # 171
print(f"Sample tags: {info['full_tags'][:3]}")
```

### Detekcja Terminów Domenowych

```python
# Sprawdź czy słowo jest terminem domenowym
if dd.is_domain_term("Konstytucja", min_frequency=2):
    print("To jest termin domenowy!")

# Pobierz preferowany POS tag dla słowa
preferred_pos = dd.get_preferred_pos("Rzeczypospolitej")
print(f"Preferowany POS: {preferred_pos}")  # 'subst'

# Pobierz wszystkie tagi domenowe dla słowa
tags = dd.get_domain_tags_for_word("Sejm")
print(tags)  # ['SEJM:subst:sg:nom:m3', 'SEJM:subst:sg:dat:m3', ...]
```

### Eksport Słownika

```python
# Eksportuj do JSON
dd.export_dictionary("my_domain_dict.json")

# Format pliku:
# {
#   "domain_stats": {...},
#   "domain_terms": {
#     "LEMMA": {
#       "pos_tags": ["subst"],
#       "case_tags": ["nom", "gen"],
#       "frequency": 42,
#       "sample_contexts": ["kontekst 1", "kontekst 2", ...]
#     },
#     ...
#   },
#   "top_tags": [...],
#   "loaded_files": [...]
# }
```

---

## Jak to Działa

### 1. Ekstrakcja z Analiz

Moduł przetwarza pliki JSON z `gtmo_results` i ekstrahuje:

- **Lemmy** - formy podstawowe słów
- **POS tags** - kategorie gramatyczne (subst, adj, verb, ...)
- **Case tags** - przypadki gramatyczne (nom, gen, dat, ...)
- **Częstości** - jak często terminy występują
- **Konteksty** - fragmenty tekstu wokół terminów

### 2. Filtrowanie Analiz Morfeusza

Gdy silnik GTMØ analizuje tekst z włączonym słownikiem domenowym:

```python
# Dla każdego słowa:
# 1. Morfeusz zwraca wiele możliwych interpretacji
# 2. Słownik domenowy ocenia każdą interpretację:
#    - +10 punktów: pełne dopasowanie tagu
#    - +5 punktów: dopasowanie POS
#    - +3 punktów: znana lemma
#    - +0-5 punktów: bonus za częstość
# 3. Wybierane są interpretacje z najwyższym score
```

### 3. Przykład Filtrowania

```python
Słowo: "Konstytucji"
Morfeusz zwraca 3 interpretacje:
  A. konstytucja:subst:sg:gen:f     (score: 18 - znana z domeny)
  B. konstytucja:subst:sg:dat:f     (score: 3  - nieznana forma)
  C. konstytucja:adj:sg:gen:f       (score: 0  - błędny POS)

Wybrana: A (najwyższy score)
```

---

## Wzorce Użycia

### Pattern 1: Analiza Jednej Domeny

```python
# Załaduj słownik prawniczy
dd_law = DomainDictionary()
dd_law.load_domain_from_results(pattern="*USTAWA*.json")

# Używaj dla wszystkich tekstów prawniczych
engine = QuantumMorphosyntaxEngine(domain_dictionary=dd_law)
```

### Pattern 2: Wiele Domen

```python
# Różne słowniki dla różnych typów tekstów
dd_constitutional = DomainDictionary()
dd_constitutional.load_domain_from_results(pattern="KONSTYTUCJA_*.json")

dd_tax = DomainDictionary()
dd_tax.load_domain_from_results(pattern="VAT_*.json")

# Automatyczna detekcja domeny (do zaimplementowania)
def select_domain(text):
    if "konstytucja" in text.lower() or "rzeczpospolita" in text.lower():
        return dd_constitutional
    elif "vat" in text.lower() or "podatek" in text.lower():
        return dd_tax
    return None

# Użycie
domain = select_domain(my_text)
engine = QuantumMorphosyntaxEngine(domain_dictionary=domain)
```

### Pattern 3: Budowanie Słownika Inkrementalnie

```python
# Dzień 1: Analizuj pierwszy zestaw
from gtmo_file_loader import GTMOFileLoader
loader = GTMOFileLoader()
loader.analyze_file("doc1.txt")  # -> gtmo_results/doc1_*.json

# Dzień 2: Zbuduj słownik z dnia 1
dd = DomainDictionary()
dd.load_domain_from_results(pattern="doc1_*.json")

# Dzień 3: Użyj słownika dla nowych analiz
engine = QuantumMorphosyntaxEngine(domain_dictionary=dd)
loader.analyze_file("doc2.txt", engine=engine)  # Lepsza jakość!
```

---

## API Reference

### Klasa `DomainDictionary`

#### `__init__(results_dir: str = "gtmo_results")`
Inicjalizuje słownik domenowy.

#### `load_domain_from_results(pattern: str, domain_name: Optional[str] = None) -> Dict`
Ładuje słownik z plików JSON.

**Parametry:**
- `pattern`: Wzorzec glob (np. `"KONSTYTUCJA_*.json"`)
- `domain_name`: Nazwa domeny (opcjonalnie)

**Zwraca:** Statystyki słownika

#### `get_most_common_terms(n: int = 50) -> List[Tuple[str, int]]`
Zwraca n najczęstszych terminów.

#### `get_term_info(lemma: str) -> Optional[Dict]`
Zwraca pełne informacje o terminie.

#### `is_domain_term(word: str, min_frequency: int = 2) -> bool`
Sprawdza czy słowo jest terminem domenowym.

#### `get_preferred_pos(word: str) -> Optional[str]`
Zwraca najbardziej prawdopodobny POS tag.

#### `get_domain_tags_for_word(word: str) -> List[str]`
Zwraca wszystkie tagi domenowe dla słowa.

#### `export_dictionary(output_path: str)`
Eksportuje słownik do pliku JSON.

---

## Struktura Plików

```
gtmo_results/
├── KONSTYTUCJA_RP_sentence_1.json    # Analiza GTMØ
├── KONSTYTUCJA_RP_sentence_2.json
├── ...
└── KONSTYTUCJA_RP_sentence_N.json

domain_dict_constitutional_law.json   # Wyeksportowany słownik
```

### Format Pliku Analizy (wejście)

```json
{
  "additional_metrics": {
    "debug_tags": [
      "KONSTYTUCJA:subst:sg:nom:f",
      "RZECZYPOSPOLITEJ:subst:sg:gen:f",
      ...
    ],
    "total_analyses": 529
  },
  "content": {
    "text": "KONSTYTUCJA RZECZYPOSPOLITEJ POLSKIEJ..."
  }
}
```

### Format Słownika (wyjście)

```json
{
  "domain_stats": {
    "total_files": 882,
    "unique_lemmas": 1298,
    "domain_name": "constitutional_law"
  },
  "domain_terms": {
    "RZECZYPOSPOLITEJ": {
      "pos_tags": ["subst"],
      "case_tags": ["gen", "dat", "loc"],
      "frequency": 171,
      "sample_contexts": ["..."]
    }
  }
}
```

---

## Najlepsze Praktyki

### ✓ DO

1. **Buduj słowniki z homogenicznych korpusów** - lepsze wyniki dla tekstów tej samej domeny
2. **Regularnie aktualizuj słowniki** - dodawaj nowe analizy do puli
3. **Używaj opisowych nazw domen** - `constitutional_law` zamiast `dict1`
4. **Eksportuj słowniki** - zachowuj kopie zapasowe
5. **Monitoruj statystyki** - sprawdzaj `domain_stats` po załadowaniu

### ✗ NIE

1. **Nie mieszaj domen** - nie łącz tekstów prawniczych z literackimi w jednym słowniku
2. **Nie używaj zbyt małych korpusów** - minimum 10-20 plików dla stabilności
3. **Nie ignoruj częstości** - terminy z freq=1 mogą być szumem
4. **Nie nadpisuj stdout** wielokrotnie - problem z kodowaniem Windows

---

## Rozwiązywanie Problemów

### Problem: `FileNotFoundError: Katalog gtmo_results nie istnieje`

**Rozwiązanie:** Upewnij się, że folder `gtmo_results` istnieje i zawiera pliki JSON analiz.

```bash
mkdir gtmo_results
python gtmo_file_loader.py --file dokument.txt
```

### Problem: `Nie znaleziono plików pasujących do wzorca`

**Rozwiązanie:** Sprawdź wzorzec glob:

```python
import glob
files = glob.glob("gtmo_results/TWOJ_WZORZEC*.json")
print(files)  # Powinien pokazać pasujące pliki
```

### Problem: `UnicodeEncodeError` na Windows

**Rozwiązanie:** Moduł automatycznie naprawia kodowanie, ale jeśli problem persystuje:

```python
import sys, io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
```

### Problem: Niska jakość filtrowania

**Rozwiązanie:**
- Zwiększ rozmiar korpusu (więcej plików)
- Sprawdź czy pliki są z tej samej domeny
- Użyj wyższego `min_frequency` w `is_domain_term()`

---

## Przykłady Zastosowań

### 1. Analiza Prawnicza

```python
# Zbuduj słownik z ustaw
dd = DomainDictionary()
dd.load_domain_from_results(pattern="*USTAWA*.json", domain_name="law")

# Analizuj nową ustawę
engine = QuantumMorphosyntaxEngine(domain_dictionary=dd)
result = engine.analyze_morphology_quantum("Treść nowej ustawy...")
```

### 2. Analiza Literacka

```python
# Słownik stylu danego autora
dd = DomainDictionary()
dd.load_domain_from_results(pattern="MICKIEWICZ_*.json", domain_name="romantic_poetry")

# Detekcja pastiszu/plagiatu
suspicious_text = "..."
if dd.is_domain_term("słowo1") and dd.is_domain_term("słowo2"):
    print("Tekst przypomina styl Mickiewicza")
```

### 3. Analiza Techniczna

```python
# Słownik dokumentacji technicznej
dd = DomainDictionary()
dd.load_domain_from_results(pattern="TECH_DOC_*.json")

# Pomoc w pisaniu nowej dokumentacji
for word in new_doc.split():
    if not dd.is_domain_term(word):
        print(f"Uwaga: '{word}' to nietypowy termin dla tej domeny")
```

---

## Rozwój i Wkład

Moduł `gtmo_domain_dictionary.py` jest częścią GTMØ Morphosyntax Engine.

**Autor:** GTMØ Team
**Licencja:** [Twoja licencja]
**Repozytorium:** [Link do repo]

---

## Kontakt i Wsparcie

W razie problemów lub pytań:
- Sprawdź [dokumentację GTMØ](README.md)
- Uruchom testy: `python test_domain_dictionary.py`
- Sprawdź przykłady: `python demo_domain_dictionary.py`

---

**Wersja:** 1.0
**Data:** 2025-11-04
**Status:** Stabilny
