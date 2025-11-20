"""
GTMØ Domain Dictionary Manager
===============================
Moduł do ładowania i wykorzystywania słowników domenowych z poprzednich analiz.
Ekstrahuje terminologię, wzorce morfosyntaktyczne i konteksty semantyczne
z plików JSON w folderze gtmo_results.
"""

import sys
import json
import os
from pathlib import Path
from typing import Dict, List, Set, Tuple, Optional
from collections import defaultdict, Counter
import re

# Fix Windows console encoding
if sys.platform == 'win32':
    import io
    if not isinstance(sys.stdout, io.TextIOWrapper) or sys.stdout.encoding != 'utf-8':
        sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
    if not isinstance(sys.stderr, io.TextIOWrapper) or sys.stderr.encoding != 'utf-8':
        sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8')


class DomainDictionary:
    """Zarządza słownikami domenowymi zbudowanymi z poprzednich analiz GTMØ."""

    def __init__(self, results_dir: str = "gtmo_results"):
        self.results_dir = Path(results_dir)
        self.domain_terms: Dict[str, Dict] = {}  # lemma -> {pos, cases, contexts, frequency}
        self.morphological_patterns: Dict[str, List] = defaultdict(list)
        self.semantic_contexts: Dict[str, Set] = defaultdict(set)
        self.domain_tags: Dict[str, int] = Counter()  # tag -> frequency
        self.loaded_files: List[str] = []

        # Statystyki domeny
        self.domain_stats = {
            'total_tokens': 0,
            'unique_lemmas': 0,
            'total_files': 0,
            'domain_name': None
        }

    def load_domain_from_results(self, pattern: str = "*.json",
                                domain_name: Optional[str] = None) -> Dict:
        """
        Ładuje słownik domenowy z plików JSON w folderze results.

        Args:
            pattern: Wzorzec nazw plików (np. "KONSTYTUCJA_RP_*.json")
            domain_name: Nazwa domeny (np. "constitutional_law")

        Returns:
            Statystyki załadowanego słownika
        """
        if not self.results_dir.exists():
            raise FileNotFoundError(f"Katalog {self.results_dir} nie istnieje")

        json_files = list(self.results_dir.glob(pattern))

        if not json_files:
            print(f"Nie znaleziono plików pasujących do wzorca: {pattern}")
            return self.domain_stats

        print(f"Ładowanie słownika domenowego z {len(json_files)} plików...")

        for json_file in json_files:
            try:
                self._process_analysis_file(json_file)
                self.loaded_files.append(str(json_file))
            except Exception as e:
                print(f"Błąd podczas przetwarzania {json_file.name}: {e}")

        # Aktualizacja statystyk
        self.domain_stats['total_files'] = len(self.loaded_files)
        self.domain_stats['unique_lemmas'] = len(self.domain_terms)
        self.domain_stats['domain_name'] = domain_name or self._infer_domain_name(pattern)

        print(f"✓ Załadowano słownik domenowy '{self.domain_stats['domain_name']}':")
        print(f"  - Plików: {self.domain_stats['total_files']}")
        print(f"  - Unikalnych lemm: {self.domain_stats['unique_lemmas']}")
        print(f"  - Całkowita liczba tokenów: {self.domain_stats['total_tokens']}")

        return self.domain_stats

    def _process_analysis_file(self, file_path: Path):
        """Przetwarza pojedynczy plik analizy JSON."""
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)

        # Ekstrakcja debug_tags (zawierają informacje morfosyntaktyczne)
        if 'additional_metrics' in data and 'debug_tags' in data['additional_metrics']:
            debug_tags = data['additional_metrics']['debug_tags']
            self._extract_from_tags(debug_tags)

        # Ekstrakcja tekstu dla kontekstów semantycznych
        if 'content' in data and 'text' in data['content']:
            text = data['content']['text']
            self._extract_semantic_contexts(text, debug_tags if 'debug_tags' in locals() else [])

        # Aktualizacja statystyk
        if 'additional_metrics' in data and 'total_analyses' in data['additional_metrics']:
            self.domain_stats['total_tokens'] += data['additional_metrics']['total_analyses']

    def _extract_from_tags(self, debug_tags: List[str]):
        """
        Ekstrahuje lemmy, POS i case tags z debug_tags.
        Format: "LEMMA:pos:details..."
        """
        for tag in debug_tags:
            parts = tag.split(':')
            if len(parts) < 2:
                continue

            lemma = parts[0]
            pos = parts[1]

            # Ignoruj interpunkcję i nierozpoznane tokeny
            if pos in ['interp', 'ign'] or not lemma:
                continue

            # Inicjalizuj wpis dla lemmy
            if lemma not in self.domain_terms:
                self.domain_terms[lemma] = {
                    'pos_tags': set(),
                    'case_tags': set(),
                    'full_tags': [],
                    'frequency': 0,
                    'contexts': []
                }

            # Aktualizuj informacje
            self.domain_terms[lemma]['pos_tags'].add(pos)
            self.domain_terms[lemma]['full_tags'].append(tag)
            self.domain_terms[lemma]['frequency'] += 1

            # Ekstrahuj case jeśli dostępny
            if len(parts) >= 4 and parts[1] == 'subst':  # rzeczowniki
                case_info = parts[3]
                for case in ['nom', 'gen', 'dat', 'acc', 'inst', 'loc', 'voc']:
                    if case in case_info:
                        self.domain_terms[lemma]['case_tags'].add(case)

            # Dodaj do wzorców morfologicznych
            pattern_key = f"{pos}:{parts[2] if len(parts) > 2 else ''}"
            self.morphological_patterns[pattern_key].append(lemma)

            # Aktualizuj statystyki tagów
            self.domain_tags[tag] += 1

    def _extract_semantic_contexts(self, text: str, tags: List[str]):
        """Ekstrahuje konteksty semantyczne dla terminów."""
        # Wyciągnij lemmy z tagów
        lemmas = [tag.split(':')[0] for tag in tags if ':' in tag and tag.split(':')[1] not in ['interp', 'ign']]

        # Dla każdej lemmy zapisz fragment tekstu jako kontekst
        words = text.split()
        for i, word in enumerate(words):
            # Dopasuj słowo do lemmy (uproszczone)
            for lemma in lemmas:
                if lemma.lower() in word.lower():
                    # Kontekst: 3 słowa przed i 3 po
                    start = max(0, i - 3)
                    end = min(len(words), i + 4)
                    context = ' '.join(words[start:end])

                    if lemma in self.domain_terms:
                        self.domain_terms[lemma]['contexts'].append(context)
                    self.semantic_contexts[lemma].add(context)

    def _infer_domain_name(self, pattern: str) -> str:
        """Wnioskuje nazwę domeny z wzorca plików."""
        # Wyciągnij wspólny prefiks z nazw plików
        if self.loaded_files:
            first_file = Path(self.loaded_files[0]).stem
            # Usuń część numeryczną
            domain = re.sub(r'_sentence_\d+|_\d+', '', first_file)
            return domain
        return "unknown_domain"

    def get_domain_tags_for_word(self, word: str) -> List[str]:
        """
        Zwraca wszystkie możliwe tagi domenowe dla danego słowa.
        Używane do priorytetyzacji interpretacji morfosyntaktycznych.
        """
        word_upper = word.upper()

        if word_upper in self.domain_terms:
            return self.domain_terms[word_upper]['full_tags']

        # Próba dopasowania przez podobieństwo
        similar_tags = []
        for lemma, data in self.domain_terms.items():
            if lemma.lower() == word.lower():
                similar_tags.extend(data['full_tags'])

        return similar_tags

    def get_preferred_pos(self, word: str) -> Optional[str]:
        """
        Zwraca najbardziej prawdopodobny POS tag dla słowa w tej domenie.
        """
        word_upper = word.upper()

        if word_upper in self.domain_terms:
            pos_tags = list(self.domain_terms[word_upper]['pos_tags'])
            if pos_tags:
                # Zwróć najczęstszy POS
                pos_counts = Counter()
                for tag in self.domain_terms[word_upper]['full_tags']:
                    pos = tag.split(':')[1] if ':' in tag else None
                    if pos:
                        pos_counts[pos] += 1
                return pos_counts.most_common(1)[0][0]

        return None

    def get_term_info(self, lemma: str) -> Optional[Dict]:
        """Zwraca pełne informacje o terminie z domeny."""
        return self.domain_terms.get(lemma.upper())

    def is_domain_term(self, word: str, min_frequency: int = 2) -> bool:
        """
        Sprawdza, czy słowo jest terminem domenowym.
        """
        word_upper = word.upper()
        if word_upper in self.domain_terms:
            return self.domain_terms[word_upper]['frequency'] >= min_frequency
        return False

    def get_most_common_terms(self, n: int = 50) -> List[Tuple[str, int]]:
        """Zwraca n najczęstszych terminów domenowych."""
        terms_with_freq = [(lemma, data['frequency'])
                          for lemma, data in self.domain_terms.items()]
        return sorted(terms_with_freq, key=lambda x: x[1], reverse=True)[:n]

    def export_dictionary(self, output_path: str):
        """Eksportuje słownik domenowy do pliku JSON."""
        export_data = {
            'domain_stats': self.domain_stats,
            'domain_terms': {
                lemma: {
                    'pos_tags': list(data['pos_tags']),
                    'case_tags': list(data['case_tags']),
                    'frequency': data['frequency'],
                    'sample_contexts': data['contexts'][:5]  # Maksymalnie 5 kontekstów
                }
                for lemma, data in self.domain_terms.items()
            },
            'top_tags': self.domain_tags.most_common(100),
            'loaded_files': self.loaded_files
        }

        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(export_data, f, ensure_ascii=False, indent=2)

        print(f"✓ Słownik domenowy wyeksportowany do: {output_path}")

    def filter_morphological_analyses(self, word: str, morfeusz_analyses: List) -> List:
        """
        Filtruje analizy morfologiczne Morfeusza używając wiedzy domenowej.
        Priorytetyzuje interpretacje zgodne ze słownikiem domenowym.

        Args:
            word: Analizowane słowo
            morfeusz_analyses: Lista analiz z Morfeusza [(start, end, (lemma, tag, interp))]

        Returns:
            Przefiltrowana i posortowana lista analiz (najlepsze na początku)
        """
        if not morfeusz_analyses:
            return []

        # Pobierz preferencje domenowe
        domain_tags = self.get_domain_tags_for_word(word)
        preferred_pos = self.get_preferred_pos(word)

        scored_analyses = []

        for analysis in morfeusz_analyses:
            _, _, (lemma, tag, _) = analysis
            score = 0

            # Pełne dopasowanie tagu
            if tag in domain_tags:
                score += 10

            # Dopasowanie POS
            if preferred_pos and tag.startswith(preferred_pos + ':'):
                score += 5

            # Dopasowanie lemmy
            if lemma.upper() in self.domain_terms:
                score += 3
                # Dodatkowe punkty za wysoką częstość
                freq = self.domain_terms[lemma.upper()]['frequency']
                score += min(freq / 10, 5)  # Maksymalnie +5 punktów

            scored_analyses.append((score, analysis))

        # Sortuj według score (malejąco)
        scored_analyses.sort(key=lambda x: x[0], reverse=True)

        return [analysis for _, analysis in scored_analyses]


def main():
    """Przykład użycia: budowanie słownika z analiz Konstytucji RP."""

    # Inicjalizacja
    dd = DomainDictionary()

    # Załaduj słownik z analiz Konstytucji
    stats = dd.load_domain_from_results(
        pattern="KONSTYTUCJA_RP_*.json",
        domain_name="constitutional_law"
    )

    # Wyświetl najczęstsze terminy
    print("\n=== Top 20 Terminów Domenowych ===")
    for i, (term, freq) in enumerate(dd.get_most_common_terms(20), 1):
        info = dd.get_term_info(term)
        pos_tags = ', '.join(info['pos_tags']) if info else 'N/A'
        print(f"{i:2}. {term:20} (freq={freq:3}) POS: {pos_tags}")

    # Przykład: analiza słowa "Konstytucja"
    print("\n=== Analiza słowa 'Konstytucja' ===")
    term_info = dd.get_term_info("KONSTYTUCJA")
    if term_info:
        print(f"POS tags: {term_info['pos_tags']}")
        print(f"Case tags: {term_info['case_tags']}")
        print(f"Frequency: {term_info['frequency']}")
        print(f"Sample tags: {term_info['full_tags'][:3]}")

    # Eksport słownika
    dd.export_dictionary("domain_dict_constitutional_law.json")

    print("\n✓ Słownik domenowy gotowy do użycia!")


if __name__ == "__main__":
    main()
