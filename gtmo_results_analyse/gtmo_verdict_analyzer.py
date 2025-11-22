#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
GTMØ Verdict Analyzer - Analizator Wyroków Sądowych
====================================================

Kompleksowa analiza wyników GTMØ dla wyroków sądowych i dokumentów prawnych.
Wizualizuje kluczowe metryki i identyfikuje "smoking guns" - najsłabsze logicznie fragmenty tekstu.

Funkcjonalności:
- "EKG" Wyroku - przebieg Dostępności Semantycznej (SA)
- Identyfikacja bloków krytycznych (SA < 10%)
- Mapa cieplna źródeł chaosu (CI decomposition)
- Macierz korelacji metryk GTMØ
- Rozkład SA w dokumencie
- Eksport wyników do CSV/JSON

Użycie:
  python gtmo_verdict_analyzer.py   <path_to_analysis.json> [opcje]

Przykład:
    python gtmo_verdict_analyzer.py ../gtmo_results/analysis_xyz/full_document.json --all
"""

import sys
import io

# Fix Windows console encoding for Unicode characters
if sys.platform == 'win32':
    try:
        if not sys.stdout.closed:
            sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
        if not sys.stderr.closed:
            sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8')
    except (ValueError, AttributeError):
        pass

import os
import json
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from pathlib import Path
import argparse
import numpy as np
from typing import Dict, List, Tuple, Optional
import warnings
warnings.filterwarnings('ignore')

# Ustawienia wizualne
sns.set_theme(style="whitegrid")
plt.rcParams['figure.figsize'] = (20, 6)
plt.rcParams['axes.titlesize'] = 16
plt.rcParams['axes.labelsize'] = 12
plt.rcParams['font.size'] = 10

# Progi krytyczne
CRITICAL_THRESHOLD = 0.10   # SA < 10% = krytyczne
WARNING_THRESHOLD = 0.30    # SA < 30% = ostrzeżenie


class NaturalLanguageRecommendations:
    """
    Generuje rekomendacje w języku naturalnym BEZ żargonu technicznego.

    Tłumaczy metryki GTMØ (SA, CI, ambiguity) na praktyczne porady
    dla prawników i legislatorów.
    """

    def __init__(self, use_llm: bool = True, api_key: str = None):
        """
        Args:
            use_llm: Czy używać LLM (Claude) dla generowania przykładów
            api_key: Klucz API do Anthropic Claude (opcjonalny, z env)
        """
        self.use_llm = use_llm
        self.api_key = api_key or os.getenv('ANTHROPIC_API_KEY')

        if self.use_llm and not self.api_key:
            print("⚠️  Brak klucza API - LLM wyłączony. Ustaw ANTHROPIC_API_KEY w .env")
            self.use_llm = False


    def translate_severity(self, sa: float) -> str:
        """Tłumaczy SA na poziom problemu w języku naturalnym"""
        if sa < 0.10:
            return "ekstremalnie trudny do zrozumienia"
        elif sa < 0.20:
            return "bardzo trudny do zrozumienia"
        elif sa < 0.30:
            return "trudny do zrozumienia"
        elif sa < 0.40:
            return "średnio czytelny, wymaga poprawy"
        elif sa < 0.50:
            return "akceptowalny, ale można uprościć"
        else:
            return "czytelny"


    def identify_main_problem(self, ci_morph_pct: float, ci_synt_pct: float,
                             ci_sem_pct: float) -> tuple[str, str]:
        """
        Identyfikuje główny problem na podstawie dekompozycji CI

        Returns:
            (problem_short, problem_detailed)
        """
        problems = {
            'morphological': (ci_morph_pct,
                            "zbyt skomplikowane słownictwo",
                            "Za dużo trudnych wyrazów prawniczych i skomplikowanych form gramatycznych"),
            'syntactic': (ci_synt_pct,
                         "zdania za długie i zagmatwane",
                         "Zdanie ma zbyt wiele zagnieżdżeń, podrzędnych części i wtrąceń"),
            'semantic': (ci_sem_pct,
                        "niejasne znaczenie słów",
                        "Wyrazy mają wieloznaczne znaczenie lub nie są jasno zdefiniowane")
        }

        # Znajdź dominujący problem
        max_problem = max(problems.items(), key=lambda x: x[1][0])
        _problem_type, (_pct, short, detailed) = max_problem

        return short, detailed


    def generate_quick_fixes(self, sentence_data: dict) -> list[str]:
        """
        Generuje listę szybkich poprawek (bez żargonu)

        Args:
            sentence_data: Dict z metrykami (SA, CI_decomp, ambiguity, depth, etc.)
        """
        fixes = []

        ci_morph = sentence_data.get('CI_morph_pct', 0)
        ci_synt = sentence_data.get('CI_synt_pct', 0)
        ci_sem = sentence_data.get('CI_sem_pct', 0)
        ambiguity = sentence_data.get('ambiguity', 0)
        depth = sentence_data.get('depth', 0)

        # Morfologia
        if ci_morph > 40:
            fixes.append("Zamień trudne wyrazy prawnicze na prostsze, codzienne słowa")
            fixes.append(f"Ogranicz użycie skomplikowanych form wyrazów (obecnie: {ci_morph:.0f}% problemu)")
        elif ci_morph > 30:
            fixes.append("Uprość niektóre wyrazy - użyj prostszych synonimów")

        # Składnia
        if ci_synt > 40:
            if depth > 12:
                fixes.append(f"Rozbij zdanie na 3-4 krótsze (obecna głębokość składniowa: {depth:.0f} poziomów)")
            elif depth > 8:
                fixes.append(f"Rozbij zdanie na 2 krótsze (obecna głębokość: {depth:.0f} poziomów)")
            fixes.append("Usuń niepotrzebne wtrącenia i nawiasy")
        elif ci_synt > 30:
            fixes.append("Rozbij na 2 krótsze zdania")

        # Semantyka
        if ci_sem > 40:
            fixes.append("Dodaj definicje dla niejasnych terminów prawnych")
            fixes.append("Precyzuj znaczenie wieloznacznych wyrażeń")
        elif ci_sem > 30:
            fixes.append("Wyjaśnij lub zdefiniuj kluczowe pojęcia")

        # Wieloznaczność
        if ambiguity > 4:
            fixes.append(f"Usuń wieloznaczności - zdanie można zrozumieć na {ambiguity:.1f} różnych sposobów")
        elif ambiguity > 3.5:
            fixes.append("Precyzuj sformułowania, aby ograniczyć możliwe interpretacje")

        # Jeśli brak konkretnych problemów, ogólne rady
        if not fixes:
            fixes.append("Uprość strukturę zdania")
            fixes.append("Użyj prostszego języka")

        return fixes


    def generate_long_term_fixes(self, sentence_data: dict) -> list[str]:
        """Generuje listę głębszych zmian długoterminowych"""
        fixes = []

        sa = sentence_data.get('SA', 0)

        if sa < 0.20:
            fixes.append("Przepisz cały artykuł od podstaw, prostszym językiem")
            fixes.append("Rozważ podział na kilka krótszych artykułów")
            fixes.append("Dodaj sekcję z przykładami praktycznymi")
        elif sa < 0.30:
            fixes.append("Przepisz artykuł używając listy punktowanej zamiast długiego zdania")
            fixes.append("Dodaj glosariusz z definicjami kluczowych pojęć")
        else:
            fixes.append("Wprowadź przykłady zastosowania w komentarzu")
            fixes.append("Ujednolicaj terminologię w całej ustawie")

        return fixes


    def generate_legal_risks(self, sentence_data: dict) -> str:
        """
        Generuje opis ryzyk prawnych w naturalnym języku

        Bazuje na classification, ambiguity, SA
        """
        classification = sentence_data.get('classification', 'UNKNOWN')
        ambiguity = sentence_data.get('ambiguity', 0)
        sa = sentence_data.get('SA', 0)

        risks = []

        # Klasyfikacja
        if classification == 'CHAOTIC_STRUCTURE':
            risks.append("🔴 WYSOKIE RYZYKO: Przepis chaotyczny i nieprzewidywalny")
        elif classification == 'BALANCED_NORM':
            risks.append("🟡 ŚREDNIE RYZYKO: Przepis wyważony, ale wymaga poprawy")

        # Wieloznaczność
        if ambiguity > 5:
            risks.append(f"⚖️ Ekstremalna wieloznaczność ({ambiguity:.1f} interpretacji) - może być zakwestionowany jako niekonstytucyjny (naruszenie zasady lex certa)")
        elif ambiguity > 4:
            risks.append("⚖️ Wysoka wieloznaczność - różne sądy mogą interpretować na różne sposoby, co prowadzi do chaosu orzeczniczego")
        elif ambiguity > 3:
            risks.append("⚖️ Możliwe spory interpretacyjne między stronami postępowania")

        # Niska dostępność semantyczna
        if sa < 0.15:
            risks.append("📋 Organy stosujące prawo nie będą wiedzieć jak wykonać przepis - problemy w egzekucji")
        elif sa < 0.25:
            risks.append("📋 Trudności w praktycznym stosowaniu - urzędnicy będą potrzebować szczegółowych wytycznych")

        # Ryzyko sporów
        if ambiguity > 3.5 and sa < 0.30:
            risks.append("🔍 Wysokie ryzyko długotrwałych sporów sądowych o interpretację")

        if not risks:
            risks.append("⚠️ Średnie ryzyko problemów z interpretacją")

        return " ".join(risks)


    def generate_recommendations(self, sentence_data: dict) -> dict:
        """
        Główna metoda - generuje pełny zestaw rekomendacji

        Args:
            sentence_data: Dict z metrykami zdania

        Returns:
            Dict z rekomendacjami w języku naturalnym
        """
        sa = sentence_data.get('SA', 0)
        text = sentence_data.get('text', '')

        # 1. Poziom problemu
        severity = self.translate_severity(sa)

        # 2. Główny problem
        main_problem_short, main_problem_detailed = self.identify_main_problem(
            sentence_data.get('CI_morph_pct', 0),
            sentence_data.get('CI_synt_pct', 0),
            sentence_data.get('CI_sem_pct', 0)
        )

        # 3. Szybkie poprawki
        quick_fixes = self.generate_quick_fixes(sentence_data)

        # 4. Długoterminowe zmiany
        long_term = self.generate_long_term_fixes(sentence_data)

        # 5. Ryzyka prawne
        legal_risks = self.generate_legal_risks(sentence_data)

        # 6. Przykład lepszej wersji (LLM lub szablon)
        if self.use_llm and len(text) > 50:
            example = self._generate_example_with_llm(sentence_data)
        else:
            example = self._generate_example_template(sentence_data)

        return {
            'severity': severity,
            'main_problem_short': main_problem_short,
            'main_problem_detailed': main_problem_detailed,
            'quick_fixes': quick_fixes,
            'long_term_fixes': long_term,
            'legal_risks': legal_risks,
            'example_better_version': example
        }


    def _generate_example_template(self, sentence_data: dict) -> str:
        """Generuje przykład poprawki używając szablonu (fallback)"""
        text = sentence_data.get('text', '')
        ci_synt = sentence_data.get('CI_synt_pct', 0)

        if ci_synt > 40:
            return f"""Było: "{text[:100]}..."

Może być (przykład uproszczenia):
"[Przepis należy rozbić na 2-3 krótsze zdania. Każde zdanie powinno mieć jeden główny cel.]"

Uwaga: To szablon - dla precyzyjnego przykładu użyj trybu z LLM."""
        else:
            return f"""Było: "{text[:100]}..."

Może być: "[Uprość słownictwo i strukturę zdania]"

Uwaga: Dla konkretnego przykładu włącz tryb LLM."""


    def _generate_example_with_llm(self, sentence_data: dict) -> str:
        """Generuje szczegółową analizę i rekomendacje używając Claude API"""
        try:
            import anthropic

            client = anthropic.Anthropic(api_key=self.api_key)

            text = sentence_data.get('text', '')
            sa = sentence_data.get('SA', 0)
            ci_morph_pct = sentence_data.get('CI_morph_pct', 0)
            ci_synt_pct = sentence_data.get('CI_synt_pct', 0)
            ci_sem_pct = sentence_data.get('CI_sem_pct', 0)
            ambiguity = sentence_data.get('ambiguity', 0)
            depth = sentence_data.get('depth', 0)

            prompt = f"""Jesteś ekspertem od przystępności języka prawnego i analizy konstytucyjnej.

ZADANIE: Przeanalizuj poniższy przepis i wygeneruj SZCZEGÓŁOWĄ rekomendację w języku naturalnym (bez żargonu technicznego).

═══════════════════════════════════════════════════════════════════════════════
ORYGINALNY PRZEPIS:
"{text}"

METRYKI GTMØ (dla kontekstu - NIE pokazuj ich użytkownikowi w formie liczb):
• Dostępność Semantyczna (SA): {sa*100:.1f}%
• Dekompozycja Niedefinitywności (CI):
  - Morfologiczna (trudne słowa): {ci_morph_pct:.0f}%
  - Składniowa (długie zdania): {ci_synt_pct:.0f}%
  - Semantyczna (niejasne znaczenie): {ci_sem_pct:.0f}%
• Wieloznaczność: {ambiguity:.1f} możliwych interpretacji
• Głębokość składniowa: {depth:.0f} poziomów zagnieżdżenia

═══════════════════════════════════════════════════════════════════════════════

INSTRUKCJE FORMATOWANIA:

1. **Użyj wizualnych ramek** z Unicode:
   - Główne sekcje: ╔══════╗, ║, ╚══════╝
   - Podsekcje: ┏━━━┓, ┃, ┗━━━┛
   - Listy: ├─, │, └─

2. **Struktura odpowiedzi** (OBOWIĄZKOWA):

   ╔═══════════════════════════════════════════════════════════════════════════╗
   ║ 📋 ANALIZA SEMANTYCZNA                                                    ║
   ╚═══════════════════════════════════════════════════════════════════════════╝

   [Wyjaśnij JĘZYKIEM NATURALNYM dlaczego ten przepis jest trudny do zrozumienia.
    Bazuj na dekompozycji CI, ale NIE używaj żargonu - wyjaśnij konkretnie co jest nie tak:
    - Jeśli CI_morph dominuje: "Za dużo skomplikowanych wyrazów prawniczych"
    - Jeśli CI_synt dominuje: "Zdanie zbyt długie, za dużo wtrąceń i podrzędników"
    - Jeśli CI_sem dominuje: "Wyrazy mają niejasne lub wieloznaczne znaczenie"]

   ╔═══════════════════════════════════════════════════════════════════════════╗
   ║ 🔍 MOŻLIWE INTERPRETACJE (problem wieloznaczności)                        ║
   ╚═══════════════════════════════════════════════════════════════════════════╝

   [Opisz 3-7 różnych sposobów, w jakie ten przepis może być zrozumiany.
    Ponumeruj je: 1️⃣, 2️⃣, 3️⃣, itd.
    Dla każdej interpretacji:
    - Wyjaśnij JAK można to zrozumieć
    - Napisz JAKI byłby skutek prawny tej interpretacji
    - Wskaż KTÓRY fragment przepisu powoduje tę wieloznaczność]

   Przykład:
   1️⃣ Pierwsza interpretacja: [opis]
      → Skutek prawny: [co by to oznaczało?]
      → Źródło problemu: [który fragment?]

   2️⃣ Druga interpretacja: [opis]
      → Skutek prawny: [...]
      → Źródło problemu: [...]

   ╔═══════════════════════════════════════════════════════════════════════════╗
   ║ 🛠️ PROPOZYCJE NAPRAWY                                                     ║
   ╚═══════════════════════════════════════════════════════════════════════════╝

   [Zaproponuj 2-3 WARIANTY poprawki. Każdy wariant to KONKRETNY przepisany tekst.]

   ┏━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┓
   ┃ WARIANT A: [nazwa, np. "Podział na krótsze zdania"]                      ┃
   ┗━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┛

   📝 Przepisany tekst:
   [Konkretny przepisany tekst - pełna wersja artykułu po poprawce]

   ✅ Zalety:
   ├─ [zaleta 1]
   ├─ [zaleta 2]
   └─ [zaleta 3]

   ⚠️ Wady:
   ├─ [wada 1 lub "Brak istotnych wad"]
   └─ [wada 2]

   📈 Szacowany wzrost SA: [np. "z 23% → ~55%"] (zwiększenie czytelności o ~140%)

   🎯 Rekomendacja: [Czy zalecasz ten wariant? Dla kogo jest najlepszy?]

   ┏━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┓
   ┃ WARIANT B: [nazwa, np. "Lista punktowana + definicje"]                   ┃
   ┗━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┛

   [Powtórz strukturę jak w WARIANT A]

   ╔═══════════════════════════════════════════════════════════════════════════╗
   ║ ⚖️ RYZYKO PRAWNE                                                          ║
   ╚═══════════════════════════════════════════════════════════════════════════╝

   [Opisz KONKRETNE ryzyka:
    - Jeśli SA < 15%: "Organy stosujące prawo nie będą wiedziały jak wykonać przepis"
    - Jeśli wieloznaczność > 4: "Różne sądy mogą interpretować na różne sposoby - chaos orzeczniczy"
    - Jeśli CI_sem dominuje: "Ryzyko zakwestionowania jako niekonstytucyjny (naruszenie lex certa)"

    NIE używaj metrycznych wartości - pisz JĘZYKIEM NATURALNYM o skutkach prawnych]

   ╔═══════════════════════════════════════════════════════════════════════════╗
   ║ 🎯 RANKING PRIORYTETÓW                                                    ║
   ╚═══════════════════════════════════════════════════════════════════════════╝

   1. [Najważniejsze działanie - co zrobić TERAZ]
   2. [Drugie w kolejności]
   3. [Trzecie w kolejności]

   ⏱️ Szacowany czas wdrożenia: [np. "2-3 dni robocze dla legislatora"]
   💰 Koszty: [np. "Niskie - wymaga tylko przeformułowania"]

WAŻNE ZASADY:
- NIE używaj żargonu technicznego (SA, CI, D-S-E, entropia, itp.) w odpowiedzi
- Pisz tak, żeby zrozumiał prawnik-praktyk lub urzędnik
- Wyjaśnij DLACZEGO coś jest problemem, a nie JAKI ma wskaźnik
- Wszystkie propozycje poprawek muszą zachować treść normatywną
- Używaj emojis do wizualizacji (📋, 🔍, 🛠️, ✅, ⚠️, 📈, 🎯, ⏱️, 💰)
- Ramki Unicode muszą być DOKŁADNIE wyrównane

TERAZ WYGENERUJ PEŁNĄ ANALIZĘ:"""

            message = client.messages.create(
                model="claude-sonnet-4-20250514",
                max_tokens=3000,
                temperature=0.4,
                messages=[{"role": "user", "content": prompt}]
            )

            return message.content[0].text

        except Exception as e:
            print(f"⚠️  LLM error: {e}")
            return self._generate_example_template(sentence_data)


class GTMOVerdictAnalyzer:
    """Analizator wyroków GTMØ"""

    def __init__(self, json_path: str):
        """
        Inicjalizacja analizatora

        Args:
            json_path: Ścieżka do pliku JSON z analizą GTMØ
        """
        self.json_path = Path(json_path)
        self.output_dir = self.json_path.parent / "verdict_analysis_output"
        self.output_dir.mkdir(exist_ok=True)

        print(f"📂 Wczytywanie: {self.json_path.name}")

        # Wczytaj i przetworz dane
        self.raw_data = self._load_json()
        self.analyses = self._extract_analyses()
        self.df = self._create_dataframe()

        print(f"✓ Załadowano {len(self.df)} bloków tekstowych")
        print(f"💾 Wyniki będą zapisane w: {self.output_dir}")


    def _load_json(self) -> dict:
        """Wczytuje plik JSON"""
        with open(self.json_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        return data


    def _extract_analyses(self) -> List[dict]:
        """
        Wyciąga analizy z różnych możliwych struktur JSON

        Returns:
            Lista analiz bloków tekstowych
        """
        data = self.raw_data
        analyses = []

        print(f"\n🔍 Analiza struktury JSON...")
        print(f"   Typ głównego obiektu: {type(data)}")

        if isinstance(data, list):
            # Format: lista analiz
            print(f"   ✓ Wykryto LISTĘ z {len(data)} analizami")
            analyses = data

        elif isinstance(data, dict):
            print(f"   Dostępne klucze: {list(data.keys())}")

            # Sprawdź różne możliwe klucze
            if 'sentences' in data:
                print(f"   ✓ Wykryto strukturę z kluczem 'sentences'")
                analyses = data['sentences']
            elif 'analyses' in data:
                print(f"   ✓ Wykryto strukturę z kluczem 'analyses'")
                analyses = data['analyses']
            elif 'results' in data:
                print(f"   ✓ Wykryto strukturę z kluczem 'results'")
                analyses = data['results']
            elif 'articles' in data:
                print(f"   ✓ Wykryto strukturę z kluczem 'articles'")
                # Wyciągnij zdania z każdego artykułu
                articles = data['articles']
                sentences = []
                for article in articles:
                    if isinstance(article, dict) and 'sentences' in article:
                        sentences.extend(article['sentences'])
                if sentences:
                    print(f"   ✓ Wyekstrahowano {len(sentences)} zdań z {len(articles)} artykułów")
                    analyses = sentences
                else:
                    # Fallback: jeśli articles nie mają sentences, użyj articles jako bloków
                    analyses = articles
            elif '_original_data' in data and isinstance(data['_original_data'], dict):
                if 'articles' in data['_original_data']:
                    print(f"   ✓ Wykryto strukturę '_original_data/articles'")
                    analyses = data['_original_data']['articles']
            elif 'stanza_analysis' in data and isinstance(data['stanza_analysis'], dict):
                if 'sentences' in data['stanza_analysis']:
                    print(f"   ✓ Wykryto strukturę 'stanza_analysis/sentences'")
                    analyses = data['stanza_analysis']['sentences']
            elif 'content' in data and 'coordinates' in data:
                # Pojedyncza analiza
                print(f"   ✓ Wykryto POJEDYNCZĄ analizę")
                analyses = [data]
            else:
                # Szukaj list wewnątrz
                for key, value in data.items():
                    if isinstance(value, list) and len(value) > 0:
                        if isinstance(value[0], dict) and 'content' in value[0]:
                            print(f"   ✓ Wykryto analizy pod kluczem '{key}'")
                            analyses = value
                            break

        if not analyses:
            raise ValueError(
                "❌ Nie znaleziono analiz w pliku JSON!\n"
                "Oczekiwano:\n"
                "- listy analiz, lub\n"
                "- obiektu z kluczem 'sentences'/'analyses'/'results', lub\n"
                "- obiektu z '_original_data/articles', lub\n"
                "- obiektu z 'stanza_analysis/sentences'"
            )

        return analyses


    def _safe_get(self, obj, *keys, default=0):
        """Bezpiecznie wyciąga wartość z zagnieżdżonej struktury"""
        for key in keys:
            if isinstance(obj, dict) and key in obj:
                obj = obj[key]
            else:
                return default
        return obj if obj is not None else default


    def _create_dataframe(self) -> pd.DataFrame:
        """
        Przetwarza analizy na pandas DataFrame

        Returns:
            DataFrame z przetworzonymi danymi
        """
        print("\n🔄 Przetwarzanie danych...")

        processed_data = []
        errors = []

        for i, analysis in enumerate(self.analyses):
            try:
                # Wyciągnij dane z różnych lokalizacji
                content = self._safe_get(analysis, 'content', default={})
                coords = self._safe_get(analysis, 'coordinates', default={})
                const_metrics = self._safe_get(analysis, 'constitutional_metrics', default={})
                additional_metrics = self._safe_get(analysis, 'additional_metrics', default={})
                depth_metrics = self._safe_get(analysis, 'depth_metrics', default={})
                rhetorical = self._safe_get(analysis, 'rhetorical_analysis', default={})

                # Sprawdź też _original_data
                original_data = self._safe_get(analysis, '_original_data', default={})
                original_coords = self._safe_get(original_data, 'coordinates', default={})
                original_const_metrics = self._safe_get(original_data, 'constitutional_metrics', default={})

                # Tekst
                text = self._safe_get(analysis, 'text') or self._safe_get(content, 'text', default='')
                text_preview = (text[:150] + '...') if len(text) > 150 else text

                # Współrzędne D-S-E
                D = self._safe_get(coords, 'determination') or self._safe_get(original_coords, 'determination') or self._safe_get(analysis, 'gtmo_coordinates', 'determination')
                S = self._safe_get(coords, 'stability') or self._safe_get(original_coords, 'stability') or self._safe_get(analysis, 'gtmo_coordinates', 'stability')
                E = self._safe_get(coords, 'entropy') or self._safe_get(original_coords, 'entropy') or self._safe_get(analysis, 'gtmo_coordinates', 'entropy')

                # Metryki konstytucyjne
                SA_obj = self._safe_get(const_metrics, 'semantic_accessibility', default={}) or self._safe_get(original_const_metrics, 'semantic_accessibility', default={})
                # FIXED: Obsługa nowego formatu z v2/v3 oraz starego formatu z 'value'
                if isinstance(SA_obj, dict):
                    # Preferuj v3 (nowsze), potem v2, potem stary format
                    SA = (self._safe_get(SA_obj, 'v3', 'value') or
                          self._safe_get(SA_obj, 'v2', 'value') or
                          self._safe_get(SA_obj, 'value'))
                else:
                    SA = SA_obj

                CD_obj = self._safe_get(const_metrics, 'definiteness', default={}) or self._safe_get(original_const_metrics, 'definiteness', default={})
                CD = self._safe_get(CD_obj, 'value') if isinstance(CD_obj, dict) else CD_obj

                CI_obj = self._safe_get(const_metrics, 'indefiniteness', default={}) or self._safe_get(original_const_metrics, 'indefiniteness', default={})
                CI = self._safe_get(CI_obj, 'value') if isinstance(CI_obj, dict) else CI_obj

                # Dekompozycja CI
                decomp = self._safe_get(CI_obj, 'decomposition', default={}) if isinstance(CI_obj, dict) else {}
                CI_morph_pct = self._safe_get(decomp, 'morphological', 'percentage')
                CI_synt_pct = self._safe_get(decomp, 'syntactic', 'percentage')
                CI_sem_pct = self._safe_get(decomp, 'semantic', 'percentage')

                # Głębokość i ambiguity
                depth = self._safe_get(depth_metrics, 'max_depth') or self._safe_get(analysis, 'depth') or self._safe_get(original_data, 'depth')
                ambiguity = self._safe_get(additional_metrics, 'ambiguity') or self._safe_get(analysis, 'ambiguity')

                # Klasyfikacja
                classification_obj = self._safe_get(const_metrics, 'classification', default={}) or self._safe_get(original_const_metrics, 'classification', default={})
                classification = self._safe_get(classification_obj, 'type', default='UNKNOWN')

                # Analiza retoryczna
                pos_anomalies = self._safe_get(rhetorical, 'pos_anomalies', default={})
                pos_anomaly_score = self._safe_get(pos_anomalies, 'anomaly_score')

                # Numer zdania
                sentence_num = self._safe_get(analysis, 'sentence_number') or self._safe_get(analysis, 'analysis_metadata', 'sentence_number') or (i + 1)

                block_data = {
                    'block_id': i,
                    'sentence_number': sentence_num,
                    'text': text_preview,
                    'full_text': text,
                    'D': float(D) if D else 0.0,
                    'S': float(S) if S else 0.0,
                    'E': float(E) if E else 0.0,
                    'SA': float(SA) if SA else 0.0,
                    'CD': float(CD) if CD else 0.0,
                    'CI': float(CI) if CI else 0.0,
                    'depth': float(depth) if depth else 0.0,
                    'ambiguity': float(ambiguity) if ambiguity else 0.0,
                    'classification': str(classification),
                    'CI_morph_pct': float(CI_morph_pct) if CI_morph_pct else 0.0,
                    'CI_synt_pct': float(CI_synt_pct) if CI_synt_pct else 0.0,
                    'CI_sem_pct': float(CI_sem_pct) if CI_sem_pct else 0.0,
                    'pos_anomaly_score': float(pos_anomaly_score) if pos_anomaly_score else 0.0
                }

                processed_data.append(block_data)

            except Exception as e:
                errors.append(f"Blok {i}: {str(e)}")
                continue

        if not processed_data:
            raise ValueError(f"❌ Nie udało się przetworzyć żadnych danych! Błędy: {errors[:5]}")

        df = pd.DataFrame(processed_data)

        # Raport
        print(f"   ✓ Przetworzono: {len(df)} bloków")
        if errors:
            print(f"   ⚠ Błędy: {len(errors)} bloków")

        return df


    # ========================================================================
    # STATYSTYKI OGÓLNE
    # ========================================================================

    def print_statistics(self):
        """Wyświetla statystyki ogólne dokumentu"""
        print("\n" + "=" * 70)
        print("STATYSTYKI OGÓLNE DOKUMENTU")
        print("=" * 70)

        # Filtruj wartości niezerowe
        sa_nonzero = self.df[self.df['SA'] > 0]['SA']
        depth_nonzero = self.df[self.df['depth'] > 0]['depth']

        if len(sa_nonzero) > 0:
            print(f"\n📊 Dostępność Semantyczna (SA):")
            print(f"   • Średnia: {sa_nonzero.mean()*100:.2f}%")
            print(f"   • Odchylenie std: {sa_nonzero.std()*100:.2f}%")
            print(f"   • Min (najgorszy): {sa_nonzero.min()*100:.2f}%")
            print(f"   • Max (najlepszy): {sa_nonzero.max()*100:.2f}%")
            print(f"   • Bloków z SA > 0: {len(sa_nonzero)}/{len(self.df)}")

        if len(depth_nonzero) > 0:
            print(f"\n📊 Głębokość Składniowa:")
            print(f"   • Średnia: {depth_nonzero.mean():.1f}")
            print(f"   • Maksymalna: {depth_nonzero.max():.0f}")

        # Współrzędne D-S-E
        d_nonzero = self.df[self.df['D'] > 0]['D']
        s_nonzero = self.df[self.df['S'] > 0]['S']
        e_nonzero = self.df[self.df['E'] > 0]['E']

        if len(d_nonzero) > 0:
            print(f"\n📊 Współrzędne D-S-E:")
            print(f"   • Determination: μ={d_nonzero.mean():.3f}, σ={d_nonzero.std():.3f}")
            print(f"   • Stability: μ={s_nonzero.mean():.3f}, σ={s_nonzero.std():.3f}")
            print(f"   • Entropy: μ={e_nonzero.mean():.3f}, σ={e_nonzero.std():.3f}")

        # Klasyfikacje
        print(f"\n📊 Klasyfikacja strukturalna:")
        for cls, count in self.df['classification'].value_counts().items():
            pct = (count / len(self.df)) * 100
            print(f"   • {cls}: {count} bloków ({pct:.1f}%)")

        # Źródła chaosu
        ci_components = self.df[['CI_morph_pct', 'CI_synt_pct', 'CI_sem_pct']]
        ci_nonzero = ci_components[(ci_components > 0).any(axis=1)]

        if len(ci_nonzero) > 0:
            print(f"\n📊 Średni rozkład źródeł Niedefinitywności (CI):")
            print(f"   • Morfologiczna: {ci_nonzero['CI_morph_pct'].mean():.1f}%")
            print(f"   • Składniowa: {ci_nonzero['CI_synt_pct'].mean():.1f}%")
            print(f"   • Semantyczna: {ci_nonzero['CI_sem_pct'].mean():.1f}%")


    # ========================================================================
    # WIZUALIZACJA 1: "EKG" WYROKU
    # ========================================================================

    def visualize_ekg(self):
        """Wykres liniowy Dostępności Semantycznej - "EKG" wyroku"""
        print(f"\n📈 Generowanie wykresu 'EKG' wyroku...")

        df_plot = self.df[self.df['SA'] > 0].copy()

        if len(df_plot) == 0:
            print("   ❌ Brak danych SA do wizualizacji!")
            return

        df_plot['SA_formatted'] = df_plot['SA'].apply(lambda x: f"{x*100:.2f}%")

        fig = px.line(
            df_plot,
            x='block_id',
            y='SA',
            title=f'Przebieg Dostępności Semantycznej (SA) w dokumencie - "EKG" Wyroku<br><sub>Analiza {len(df_plot)} bloków tekstowych</sub>',
            labels={
                'block_id': 'Numer Bloku Tekstowego',
                'SA': 'Dostępność Semantyczna (SA)'
            },
            hover_data={
                'SA': False,
                'SA_formatted': True,
                'text': True,
                'sentence_number': True,
                'depth': True,
                'classification': True
            }
        )

        # Progi
        fig.add_hline(
            y=CRITICAL_THRESHOLD,
            line_dash="dash",
            line_color="red",
            line_width=2,
            annotation_text=f"Próg Krytyczny ({CRITICAL_THRESHOLD*100}%)",
            annotation_position="right"
        )

        fig.add_hline(
            y=WARNING_THRESHOLD,
            line_dash="dot",
            line_color="orange",
            line_width=1,
            annotation_text=f"Próg Ostrzegawczy ({WARNING_THRESHOLD*100}%)",
            annotation_position="right"
        )

        fig.update_traces(
            mode='lines+markers',
            line=dict(width=2, color='steelblue'),
            marker=dict(size=6)
        )

        fig.update_layout(
            yaxis_tickformat='.0%',
            height=600,
            hovermode='closest'
        )

        output_path = self.output_dir / "ekg_wyroku.html"
        fig.write_html(str(output_path))

        print(f"   ✓ Zapisano: {output_path.name}")


    # ========================================================================
    # WIZUALIZACJA 2: SMOKING GUNS
    # ========================================================================

    def identify_smoking_guns(self):
        """Identyfikuje i raportuje bloki krytyczne"""
        print(f"\n🔍 Identyfikacja 'Smoking Guns'...")

        df_valid = self.df[self.df['SA'] > 0].copy()

        critical_blocks = df_valid[df_valid['SA'] < CRITICAL_THRESHOLD].sort_values('SA')
        warning_blocks = df_valid[
            (df_valid['SA'] >= CRITICAL_THRESHOLD) &
            (df_valid['SA'] < WARNING_THRESHOLD)
        ].sort_values('SA')

        print("\n" + "=" * 70)
        print("🔍 SMOKING GUNS - BLOKI KRYTYCZNE")
        print("=" * 70)

        if len(critical_blocks) == 0:
            print(f"\n✓ Dobra wiadomość: Brak bloków krytycznych (SA < {CRITICAL_THRESHOLD*100}%)")
        else:
            print(f"\n❌ Znaleziono {len(critical_blocks)} bloków krytycznych (SA < {CRITICAL_THRESHOLD*100}%):")
            print(f"   To {(len(critical_blocks)/len(df_valid)*100):.1f}% dokumentu!\n")

            # Top 5
            print(f"🔴 TOP 5 NAJGORSZYCH BLOKÓW:\n")
            for idx, row in critical_blocks.head(5).iterrows():
                print(f"#{row['block_id']} | SA: {row['SA']*100:.2f}% | Głębokość: {row['depth']:.0f}")
                print(f"   {row['text'][:100]}...\n")

            # Zapisz do pliku
            report_path = self.output_dir / "smoking_guns.txt"
            with open(report_path, 'w', encoding='utf-8') as f:
                f.write("SMOKING GUNS - BLOKI KRYTYCZNE\n")
                f.write("=" * 70 + "\n\n")
                f.write(f"Znaleziono: {len(critical_blocks)} bloków krytycznych\n")
                f.write(f"Próg: SA < {CRITICAL_THRESHOLD*100}%\n\n")

                for idx, row in critical_blocks.iterrows():
                    f.write(f"\nBlok #{row['block_id']} (Zdanie #{row['sentence_number']})\n")
                    f.write(f"SA: {row['SA']*100:.2f}% | Głębokość: {row['depth']:.0f} | Chaos składni: {row['CI_synt_pct']:.1f}%\n")
                    f.write(f"Klasyfikacja: {row['classification']}\n")
                    f.write(f"Tekst: {row['full_text']}\n")
                    f.write("-" * 70 + "\n")

            print(f"   💾 Raport zapisany: {report_path.name}")

            # CSV
            csv_path = self.output_dir / "smoking_guns.csv"
            critical_blocks.to_csv(csv_path, index=False, encoding='utf-8')
            print(f"   💾 CSV zapisany: {csv_path.name}")

        if len(warning_blocks) > 0:
            print(f"\n⚠️  Znaleziono {len(warning_blocks)} bloków ostrzegawczych ({CRITICAL_THRESHOLD*100}% ≤ SA < {WARNING_THRESHOLD*100}%)")
            print(f"   To {(len(warning_blocks)/len(df_valid)*100):.1f}% dokumentu")

        return critical_blocks, warning_blocks


    # ========================================================================
    # WIZUALIZACJA 3: MAPA CIEPLNA CHAOSU
    # ========================================================================

    def visualize_chaos_heatmap(self):
        """Mapa cieplna źródeł chaosu (CI decomposition)"""
        print(f"\n🔥 Generowanie mapy cieplnej chaosu...")

        chaos_df = self.df[
            (self.df['CI_morph_pct'] > 0) |
            (self.df['CI_synt_pct'] > 0) |
            (self.df['CI_sem_pct'] > 0)
        ].copy()

        if len(chaos_df) == 0:
            print("   ❌ Brak danych o dekompozycji CI!")
            return

        chaos_components = chaos_df[['CI_morph_pct', 'CI_synt_pct', 'CI_sem_pct']]

        plt.figure(figsize=(22, 5))

        sns.heatmap(
            chaos_components.T,
            cmap='Reds',
            annot=False,
            cbar_kws={'label': 'Udział procentowy (%)'},
            vmin=0,
            vmax=100
        )

        plt.title(
            f'Mapa Cieplna Źródeł Chaosu (Niedefinitywności CI)\n'
            f'Analiza {len(chaos_df)} bloków',
            pad=20
        )
        plt.xlabel('Numer Bloku', fontsize=12)
        plt.ylabel('Komponent Chaosu', fontsize=12)
        plt.yticks(
            ticks=[0.5, 1.5, 2.5],
            labels=['Morfologiczny', 'Składniowy', 'Semantyczny'],
            rotation=0
        )

        plt.tight_layout()

        output_path = self.output_dir / "chaos_heatmap.png"
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()

        print(f"   ✓ Zapisano: {output_path.name}")

        # Statystyki dominacji
        dominant = chaos_components.idxmax(axis=1).value_counts()
        print(f"\n   📊 Dominujące źródło chaosu:")
        source_names = {
            'CI_morph_pct': 'Morfologiczny',
            'CI_synt_pct': 'Składniowy',
            'CI_sem_pct': 'Semantyczny'
        }
        for source, count in dominant.items():
            pct = (count / len(chaos_df)) * 100
            print(f"      • {source_names.get(source, source)}: {count} bloków ({pct:.1f}%)")


    # ========================================================================
    # WIZUALIZACJA 4: MACIERZ KORELACJI
    # ========================================================================

    def visualize_correlation(self):
        """Macierz korelacji metryk GTMØ"""
        print(f"\n🔗 Generowanie macierzy korelacji...")

        corr_cols = [
            'SA', 'D', 'S', 'E',
            'CD', 'CI',
            'depth', 'ambiguity',
            'CI_morph_pct', 'CI_synt_pct', 'CI_sem_pct',
            'pos_anomaly_score'
        ]

        available_cols = [col for col in corr_cols if col in self.df.columns and self.df[col].sum() != 0]

        if len(available_cols) < 3:
            print("   ❌ Za mało danych do obliczenia korelacji!")
            return

        corr_matrix = self.df[available_cols].corr()

        plt.figure(figsize=(14, 12))

        sns.heatmap(
            corr_matrix,
            annot=True,
            cmap='coolwarm',
            fmt='.2f',
            center=0,
            vmin=-1,
            vmax=1,
            square=True,
            linewidths=0.5,
            cbar_kws={'label': 'Współczynnik korelacji Pearsona'}
        )

        plt.title(
            'Macierz Korelacji Kluczowych Metryk GTMØ\n'
            'Wartości bliskie -1 lub +1 oznaczają silny związek',
            pad=20
        )
        plt.xticks(rotation=45, ha='right')
        plt.yticks(rotation=0)

        plt.tight_layout()

        output_path = self.output_dir / "correlation_matrix.png"
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()

        print(f"   ✓ Zapisano: {output_path.name}")

        # Najsilniejsze korelacje z SA
        if 'SA' in corr_matrix.columns:
            sa_corrs = corr_matrix['SA'].drop('SA').sort_values()
            print(f"\n   📊 Najsilniejsze korelacje z SA:")
            print(f"      Negatywne (↑metryka → ↓SA):")
            for metric, corr in sa_corrs.head(3).items():
                print(f"         • {metric}: {corr:.3f}")
            print(f"      Pozytywne (↑metryka → ↑SA):")
            for metric, corr in sa_corrs.tail(3).items():
                print(f"         • {metric}: {corr:.3f}")


    # ========================================================================
    # WIZUALIZACJA 5: ROZKŁAD SA
    # ========================================================================

    def visualize_sa_distribution(self):
        """Rozkład SA w dokumencie"""
        print(f"\n📊 Generowanie rozkładu SA...")

        df_valid = self.df[self.df['SA'] > 0].copy()

        if len(df_valid) == 0:
            print("   ❌ Brak danych SA!")
            return

        fig, axes = plt.subplots(1, 2, figsize=(18, 6))

        # Histogram
        axes[0].hist(df_valid['SA'] * 100, bins=30, edgecolor='black', alpha=0.7, color='steelblue')
        axes[0].axvline(x=CRITICAL_THRESHOLD*100, color='red', linestyle='--', linewidth=2, label=f'Krytyczny ({CRITICAL_THRESHOLD*100}%)')
        axes[0].axvline(x=WARNING_THRESHOLD*100, color='orange', linestyle='--', linewidth=2, label=f'Ostrzegawczy ({WARNING_THRESHOLD*100}%)')
        axes[0].set_xlabel('Dostępność Semantyczna SA (%)', fontsize=12)
        axes[0].set_ylabel('Liczba bloków', fontsize=12)
        axes[0].set_title('Rozkład Dostępności Semantycznej', fontsize=14)
        axes[0].legend()
        axes[0].grid(alpha=0.3)

        # Box plot według klasyfikacji
        classifications = df_valid['classification'].unique()
        classification_data = [
            df_valid[df_valid['classification'] == cls]['SA'] * 100
            for cls in classifications
        ]

        bp = axes[1].boxplot(
            classification_data,
            labels=classifications,
            patch_artist=True,
            widths=0.6
        )

        # Kolory
        for patch in bp['boxes']:
            patch.set_facecolor('lightblue')

        axes[1].axhline(y=CRITICAL_THRESHOLD*100, color='red', linestyle='--', linewidth=2, alpha=0.5)
        axes[1].axhline(y=WARNING_THRESHOLD*100, color='orange', linestyle='--', linewidth=2, alpha=0.5)
        axes[1].set_ylabel('Dostępność Semantyczna SA (%)', fontsize=12)
        axes[1].set_xlabel('Klasyfikacja strukturalna', fontsize=12)
        axes[1].set_title('SA według typu struktury', fontsize=14)
        plt.setp(axes[1].xaxis.get_majorticklabels(), rotation=45, ha='right')
        axes[1].grid(alpha=0.3, axis='y')

        plt.tight_layout()

        output_path = self.output_dir / "sa_distribution.png"
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()

        print(f"   ✓ Zapisano: {output_path.name}")


    # ========================================================================
    # REKOMENDACJE W JĘZYKU NATURALNYM
    # ========================================================================

    def generate_natural_language_recommendations(self, use_llm: bool = True):
        """
        Generuje rekomendacje w języku naturalnym dla przepisów z SA < 30%

        Args:
            use_llm: Czy używać LLM (Claude) dla generowania przykładów poprawek
        """
        print(f"\n📝 Generowanie rekomendacji w języku naturalnym...")

        # Filtruj zdania z SA < 30%
        problematic = self.df[self.df['SA'] < WARNING_THRESHOLD].copy()

        if len(problematic) == 0:
            print(f"   ✓ Świetnie! Brak przepisów wymagających poprawy (SA < {WARNING_THRESHOLD*100}%)")
            return

        print(f"   📊 Znaleziono {len(problematic)} przepisów wymagających poprawy ({len(problematic)/len(self.df)*100:.1f}% dokumentu)")

        # Inicjalizuj generator rekomendacji
        recommender = NaturalLanguageRecommendations(
            use_llm=use_llm,
            api_key=os.getenv('ANTHROPIC_API_KEY')
        )

        # Generuj rekomendacje
        recommendations = []

        for idx, row in problematic.iterrows():
            sentence_data = {
                'text': row['full_text'],
                'SA': row['SA'],
                'CI_morph_pct': row.get('CI_morph_pct', 0),
                'CI_synt_pct': row.get('CI_synt_pct', 0),
                'CI_sem_pct': row.get('CI_sem_pct', 0),
                'ambiguity': row.get('ambiguity', 0),
                'depth': row.get('depth', 0),
                'classification': row.get('classification', 'UNKNOWN')
            }

            # Generuj rekomendacje
            rec = recommender.generate_recommendations(sentence_data)

            recommendations.append({
                'sentence_id': row['block_id'],
                'sentence_number': row.get('sentence_number', idx),
                'text_preview': row['text'][:100],
                'full_text': row['full_text'],
                **rec
            })

            # Progress
            if (len(recommendations) % 20 == 0):
                print(f"   🔄 Przetworzono {len(recommendations)}/{len(problematic)} przepisów...")

        print(f"   ✓ Wygenerowano {len(recommendations)} rekomendacji")

        # Zapisz raport
        self._save_recommendations_report(recommendations)

        return recommendations


    def _save_recommendations_report(self, recommendations: List[dict]):
        """Zapisuje raport rekomendacji w czytelnym formacie"""

        report_path = self.output_dir / "recommendations_natural_language.txt"

        with open(report_path, 'w', encoding='utf-8') as f:
            f.write("=" * 80 + "\n")
            f.write("RAPORT: PRZEPISY WYMAGAJĄCE POPRAWY\n")
            f.write("Wnioski w języku naturalnym (bez żargonu technicznego)\n")
            f.write("=" * 80 + "\n\n")

            f.write(f"Dokument: {self.json_path.parent.name}\n")
            f.write(f"Problemów znalezionych: {len(recommendations)} przepisów\n\n")

            for i, rec in enumerate(recommendations, 1):
                f.write("-" * 80 + "\n")
                f.write(f"PRZEPIS #{i} (Zdanie #{rec['sentence_number']})\n")
                f.write("-" * 80 + "\n\n")

                f.write("TEKST:\n")
                f.write(f'"{rec["full_text"]}"\n\n')

                f.write("PROBLEM:\n")
                f.write(f"Ten przepis jest {rec['severity']}.\n")
                f.write(f"Główny problem: {rec['main_problem_detailed']}\n\n")

                f.write("CO ZROBIĆ TERAZ (proste poprawki):\n")
                for j, fix in enumerate(rec['quick_fixes'], 1):
                    f.write(f"  {j}. {fix}\n")
                f.write("\n")

                f.write("CO ZROBIĆ DŁUGOTERMINOWO:\n")
                for j, fix in enumerate(rec['long_term_fixes'], 1):
                    f.write(f"  {j}. {fix}\n")
                f.write("\n")

                f.write("PRZYKŁAD LEPSZEJ WERSJI:\n")
                f.write(rec['example_better_version'])
                f.write("\n\n")

                f.write("RYZYKO PRAWNE:\n")
                f.write(rec['legal_risks'])
                f.write("\n\n")

        print(f"   💾 Raport zapisany: {report_path.name}")

        # Zapisz też JSON dla łatwiejszego parsowania
        json_path = self.output_dir / "recommendations_natural_language.json"
        with open(json_path, 'w', encoding='utf-8') as f:
            json.dump(recommendations, f, indent=2, ensure_ascii=False)

        print(f"   💾 JSON zapisany: {json_path.name}")


    # ========================================================================
    # EKSPORT WYNIKÓW
    # ========================================================================

    def export_results(self):
        """Eksportuje wyniki do plików"""
        print(f"\n💾 Eksport wyników...")

        # Pełne dane
        csv_path = self.output_dir / "gtmo_full_analysis.csv"
        self.df.to_csv(csv_path, index=False, encoding='utf-8')
        print(f"   ✓ Pełna analiza: {csv_path.name}")

        # Podsumowanie JSON
        sa_nonzero = self.df[self.df['SA'] > 0]['SA']
        depth_nonzero = self.df[self.df['depth'] > 0]['depth']
        df_valid = self.df[self.df['SA'] > 0]

        critical_blocks = df_valid[df_valid['SA'] < CRITICAL_THRESHOLD]
        warning_blocks = df_valid[
            (df_valid['SA'] >= CRITICAL_THRESHOLD) &
            (df_valid['SA'] < WARNING_THRESHOLD)
        ]

        summary = {
            'document': self.json_path.parent.name,
            'total_blocks': len(self.df),
            'valid_blocks': len(df_valid),
            'critical_blocks': len(critical_blocks),
            'warning_blocks': len(warning_blocks),
            'statistics': {
                'mean_SA': float(sa_nonzero.mean()) if len(sa_nonzero) > 0 else 0,
                'min_SA': float(sa_nonzero.min()) if len(sa_nonzero) > 0 else 0,
                'max_SA': float(sa_nonzero.max()) if len(sa_nonzero) > 0 else 0,
                'std_SA': float(sa_nonzero.std()) if len(sa_nonzero) > 0 else 0,
                'mean_depth': float(depth_nonzero.mean()) if len(depth_nonzero) > 0 else 0,
                'max_depth': float(depth_nonzero.max()) if len(depth_nonzero) > 0 else 0
            }
        }

        json_path = self.output_dir / "gtmo_summary.json"
        with open(json_path, 'w', encoding='utf-8') as f:
            json.dump(summary, f, indent=2, ensure_ascii=False)
        print(f"   ✓ Podsumowanie: {json_path.name}")


    def generate_all(self):
        """Generuje wszystkie analizy i wizualizacje"""
        self.print_statistics()
        self.visualize_ekg()
        self.identify_smoking_guns()
        self.visualize_chaos_heatmap()
        self.visualize_correlation()
        self.visualize_sa_distribution()
        self.export_results()

        print("\n" + "=" * 70)
        print("✅ ANALIZA ZAKOŃCZONA")
        print("=" * 70)
        print(f"\nWszystkie wyniki zapisane w: {self.output_dir}")
        print(f"\nWygenerowane pliki:")
        for file in sorted(self.output_dir.iterdir()):
            print(f"   • {file.name}")


# ============================================================================
# CLI
# ============================================================================

def main():
    parser = argparse.ArgumentParser(
        description='GTMØ Verdict Analyzer - Analizator Wyroków Sądowych',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Przykłady użycia:

  # Pełna analiza
  python gtmo_verdict_analyzer.py path/to/document.json --all

  # Tylko statystyki i smoking guns
  python gtmo_verdict_analyzer.py path/to/document.json --stats --smoking-guns

  # Wszystkie wizualizacje
  python gtmo_verdict_analyzer.py path/to/document.json --visualize
        """
    )

    parser.add_argument('json_file', help='Ścieżka do pliku JSON z analizą GTMØ')

    # Opcje
    parser.add_argument('--all', action='store_true', help='Wszystkie analizy i wizualizacje')
    parser.add_argument('--stats', action='store_true', help='Statystyki ogólne')
    parser.add_argument('--ekg', action='store_true', help='Wykres "EKG" wyroku')
    parser.add_argument('--smoking-guns', action='store_true', help='Identyfikacja bloków krytycznych')
    parser.add_argument('--chaos', action='store_true', help='Mapa cieplna chaosu')
    parser.add_argument('--correlation', action='store_true', help='Macierz korelacji')
    parser.add_argument('--distribution', action='store_true', help='Rozkład SA')
    parser.add_argument('--visualize', action='store_true', help='Wszystkie wizualizacje')
    parser.add_argument('--export', action='store_true', help='Eksport wyników')
    parser.add_argument('--recommendations', action='store_true',
                       help='Generuj rekomendacje w języku naturalnym (z LLM)')
    parser.add_argument('--no-llm', action='store_true',
                       help='Wyłącz LLM (szybsze, ale bez konkretnych przykładów)')

    args = parser.parse_args()

    print("=" * 70)
    print("GTMØ VERDICT ANALYZER")
    print("=" * 70)

    analyzer = GTMOVerdictAnalyzer(args.json_file)

    if args.all:
        analyzer.generate_all()
    else:
        if args.stats:
            analyzer.print_statistics()
        if args.ekg or args.visualize:
            analyzer.visualize_ekg()
        if args.smoking_guns:
            analyzer.identify_smoking_guns()
        if args.chaos or args.visualize:
            analyzer.visualize_chaos_heatmap()
        if args.correlation or args.visualize:
            analyzer.visualize_correlation()
        if args.distribution or args.visualize:
            analyzer.visualize_sa_distribution()
        if args.export:
            analyzer.export_results()
        if args.recommendations:
            use_llm = not args.no_llm  # LLM domyślnie włączony, chyba że --no-llm
            analyzer.generate_natural_language_recommendations(use_llm=use_llm)

        # Jeśli nic nie wybrano, pokaż wszystko
        if not any([args.stats, args.ekg, args.smoking_guns, args.chaos,
                   args.correlation, args.distribution, args.visualize, args.export,
                   args.recommendations]):
            analyzer.generate_all()


if __name__ == "__main__":
    main()
