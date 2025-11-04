# GTMØ Quantum Morphosyntax Engine - Dokumentacja Techniczna

## Przegląd Systemu

GTMØ (Gödel-Tarski-Montague-Ørsted) Quantum Morphosyntax Engine to zaawansowany system analizy morfosyntaktycznej języka polskiego z kwantową semantyką i kompletnym systemem aksjomatów działającym jako system immunologiczny.

### Wersja
- **Wersja modułu:** 2.0
- **Typ analizy:** GTMØ Quantum Morphosyntax
- **Język:** Polski (pl)
- **Ostatnia aktualizacja:** November 4, 2025

## Architektura Systemu

### 1. Główne Komponenty

#### 1.1 Teoretyczne Stałe GTMØ
```python
PHI = (1 + sqrt(5)) / 2                    # Złoty podział
SINGULARITY_COORDS = [1.0, 1.0, 0.0]      # Współrzędne singularności
COGNITIVE_CENTER = [0.5, 0.5, 0.5]        # Centrum kognitywne
ENTROPY_THRESHOLD_SINGULARITY = 0.001      # Próg entropii singularności
BOUNDARY_THICKNESS = 0.02                  # Grubość granicy topologicznej
DECOHERENCE_RATE = 0.02                   # Tempo dekoherencji kwantowej
ENTANGLEMENT_THRESHOLD = 0.7              # Próg splątania kwantowego
```

#### 1.2 Przestrzeń Współrzędnych F³
System operuje w trójwymiarowej przestrzeni semantycznej:
- **D (Determination):** Stopień określoności semantycznej [0,1]
- **S (Stability):** Stabilność strukturalna [0,1] 
- **E (Entropy):** Entropia semantyczna [0,1]

#### 1.3 Mapowanie Przypadków Polskich
```python
CASE_COORDS = {
    'nom': [0.95, 0.92, 0.08],  # Mianownik - wysoka określoność
    'gen': [0.55, 0.25, 0.88],  # Dopełniacz - wysoka entropia
    'dat': [0.72, 0.65, 0.35],  # Celownik - umiarkowane wartości
    'acc': [0.89, 0.85, 0.15],  # Biernik - wysoka stabilność
    'ins': [0.42, 0.18, 0.95],  # Narzędnik - maksymalna entropia
    'loc': [0.78, 0.95, 0.12],  # Miejscownik - maksymalna stabilność
    'voc': [0.65, 0.35, 0.75]   # Wołacz - wysoka entropia
}
```

#### 1.4 Mapowanie Części Mowy
```python
POS_COORDS = {
    'subst': [0.80, 0.85, 0.20],  # Rzeczownik - stabilny
    'adj':   [0.65, 0.68, 0.32],  # Przymiotnik - umiarkowany
    'verb':  [0.70, 0.45, 0.65],  # Czasownik - dynamiczny
    'adv':   [0.52, 0.38, 0.68],  # Przysłówek - entropia
    'num':   [0.95, 0.90, 0.10],  # Liczebnik - maksymalna określoność
    'pron':  [0.68, 0.52, 0.53],  # Zaimek - neutralny
    'prep':  [0.76, 0.75, 0.24],  # Przyimek - strukturalny
    'conj':  [0.65, 0.85, 0.20],  # Spójnik - łączący
    'part':  [0.40, 0.26, 0.84],  # Partykuła - chaotyczna
    'interp':[0.95, 0.95, 0.05]   # Interpunkcja - maksymalna precyzja
}
```

### 2. System Aksjomatów GTMØ

#### 2.1 Klasa GTMOAxiomSystem
System 13 aksjomatów działających jako immunologiczna ochrona przed niespójnościami semantycznymi.

```python
class GTMOAxiomSystem:
    """Complete implementation of GTMØ 13 Executable Axioms."""
```

#### 2.2 Poziomy Aktywacji Aksjomatów
```python
class AxiomActivationLevel(Enum):
    DORMANT = 0.0      # Nieaktywny
    MONITORING = 0.3   # Monitorowanie
    ACTIVE = 0.7       # Aktywny
    CRITICAL = 1.0     # Krytyczny
```

#### 2.3 Lista Aksjomatów
- **AX0:** Systemic Uncertainty - Wprowadza fundamentalną niepewność kwantową
- **AX1:** Ontological Difference - Zapobiega zbliżeniu do singularności
- **AX2:** Translogical Isolation - Blokuje translogiczne ścieżki do Ø
- **AX3:** Epistemic Singularity - Zapobiega epistemicznym roszczeniom
- **AX4:** Non-representability - Blokuje standardową reprezentację Ø
- **AX5:** Topological Boundary - Utrzymuje Ø na granicy topologicznej
- **AX6:** Heuristic Extremum - Wymusza minimalną entropię dla Ø
- **AX7:** Meta-closure - Wywołuje meta-refleksję kognitywną
- **AX8:** Not Limit Point - Zapobiega zbieżności sekwencji do Ø
- **AX9:** Operator Irreducibility - Blokuje standardowe operatory przy Ø
- **AX10:** Meta-operator Definition - Dopuszcza tylko meta-operatory przy Ø
- **AX11:** Adaptive Learning - Uczenie się z napotkanych granic
- **AX12:** Topological Classification - Klasyfikacja przez atraktor topologiczny

### 3. Stany Kwantowe

#### 3.1 Enumeracja Stanów
```python
class QuantumState(Enum):
    SUPERPOSITION = "⟨ψ|"        # Superpozycja kwantowa
    COLLAPSED = "|ψ⟩"            # Stan skolapsowany
    ENTANGLED = "⟨ψ₁ψ₂|"         # Stan splątany
    DECOHERENT = "|ψ_mixed⟩"     # Stan zdekoherentny
```

#### 3.2 Klasa QuantumSemanticState
```python
@dataclass
class QuantumSemanticState:
    amplitudes: Dict[str, complex]    # Amplitudy kwantowe dla przypadków
    phase: float                      # Faza kwantowa
    coherence: float                  # Koherencja kwantowa
    entangled_with: List[str]         # Lista słów splątanych
    measurement_count: int            # Liczba pomiarów
```

### 4. Główny Silnik Analizy

#### 4.1 Klasa QuantumMorphosyntaxEngine
```python
class QuantumMorphosyntaxEngine:
    """Główny silnik analizy morfosyntaktycznej z kwantową semantyką."""
```

#### 4.2 Kluczowe Metody

##### 4.2.1 Adaptacyjne Wagi
```python
def calculate_adaptive_weights(self, text: str, morph_meta: Dict, synt_meta: Dict) -> Tuple[float, float]:
    """
    Oblicza adaptacyjne wagi dla fuzji morfologii i składni.
    
    Args:
        text: Analizowany tekst
        morph_meta: Metadane morfologiczne
        synt_meta: Metadane składniowe
        
    Returns:
        Tuple[morph_weight, synt_weight]: Znormalizowane wagi
    """
```

**Strategia Wagowa:**
- Krótkie fragmenty (1-3 słowa): morph_weight = 0.40
- Średnie zdania (4-15 słów): morph_weight = 0.64
- Długie zdania (16-30 słów): morph_weight = 0.55
- Bardzo długie (30+ słów): morph_weight = 0.45

##### 4.2.2 Analiza Morfologiczna z Kwantową Superpozycją
```python
def analyze_morphology_quantum(self, text: str) -> Tuple[np.ndarray, Dict, Dict[str, QuantumSemanticState]]:
    """
    Analiza morfologiczna z kwantową superpozycją.
    
    Returns:
        - coords: Współrzędne finalne w przestrzeni F³
        - metadata: Metadane analizy (ambiguity, cases, pos)
        - quantum_states: Stany kwantowe słów
    """
```

##### 4.2.3 Analiza Składniowa z Detekcją Splątania
```python
def analyze_syntax_quantum(self, text: str, word_quantum_states: Dict) -> Tuple[np.ndarray, Dict, List[Tuple]]:
    """
    Analiza składniowa z detekcją splątania kwantowego.
    
    Returns:
        - coords: Współrzędne składniowe
        - metadata: Metadane składniowe
        - entanglements: Lista wykrytych splątań
    """
```

##### 4.2.4 Główna Funkcja Analizy
```python
def gtmo_analyze_quantum(self, text: str, source_file: Optional[Dict] = None) -> Dict:
    """
    Główna funkcja analizy GTMØ z integracją aksjomatów.
    
    Args:
        text: Tekst do analizy
        source_file: Informacje o pliku źródłowym
        
    Returns:
        Dict: Kompletny wynik analizy w formacie JSON
    """
```

### 5. Constitutional Metrics - Metryki Konstytucyjne

#### 5.1 Teoretyczne Podstawy
Metryki oparte na Zasadzie Nieoznaczoności Semantycznej:
```
Δ_form · Δ_int ≥ ħ_semantic
```

Projekcja morfosyntaktyczna:
```
CD = (1/Ambiguity) × Depth × √(D×S/E)    # Constitutional Definiteness
CI = Ambiguity × Depth × √(E/(D×S))      # Constitutional Indefiniteness
```

#### 5.2 Dualność Matematyczna
Fundamentalne ograniczenie:
```
CI × CD = Depth²
```

#### 5.3 Constitutional Definiteness (CD)
- **Wzór:** `CD = (1/Ambiguity) × Depth × √(D×S/E)`
- **Interpretacja:** Miara określoności strukturalnej
- **Wysoka CD:** Tekst uporządkowany, jednoznaczny, strukturalny
- **Komponenty:**
  - `inverse_ambiguity`: Odwrotność ambiguity morfologicznej
  - `depth`: Głębokość składniowa
  - `geometric_balance`: Balans geometryczny √(D×S/E)

#### 5.4 Constitutional Indefiniteness (CI)
- **Wzór:** `CI = Ambiguity × Depth × √(E/(D×S))`
- **Interpretacja:** Miara niedefinitywności strukturalnej
- **Wysoka CI:** Tekst chaotyczny, wieloznaczny, nieprzewidywalny
- **Komponenty:**
  - `ambiguity`: Ambiguity morfologiczna
  - `depth`: Głębokość składniowa
  - `geometric_tension`: Napięcie geometryczne √(E/(D×S))

#### 5.5 Dekompozycja CI według Źródeł
```python
CI_total = CI_morphological + CI_syntactic + CI_semantic
```

- **CI_morphological:** Wkład morfologii (fleksja, ambiguity)
- **CI_syntactic:** Wkład składni (głębokość, długość zdań)
- **CI_semantic:** Chaos semantyczny w przestrzeni F³

#### 5.6 Semantic Accessibility (SA)
- **Wzór:** `SA = CD / (CI + CD) = CD / Depth²`
- **Zakres:** [0, 1]
- **Interpretacja:**
  - SA → 1: Tekst bardzo dostępny (maksymalna definiteness)
  - SA → 0: Tekst niedostępny (minimalna definiteness)
- **Kategorie:**
  - SA > 0.7: WYSOKA_DOSTĘPNOŚĆ
  - SA 0.3-0.7: ŚREDNIA_DOSTĘPNOŚĆ  
  - SA < 0.3: NISKA_DOSTĘPNOŚĆ

#### 5.7 Klasyfikacja Strukturalna
Na podstawie ratio CD/CI:
- **CD/CI > 1.0:** ORDERED_STRUCTURE (strukturalny)
- **CD/CI > 0.5:** BALANCED_STRUCTURE (zbalansowany)
- **CD/CI ≤ 0.5:** CHAOTIC_STRUCTURE (chaotyczny)

### 6. Tensor Kwantowy
```python
T_quantum = D × S × (1 - E)
```

Tensor reprezentuje całkowitą "kwantowość" semantyczną tekstu.

### 7. Struktura Wyniku JSON

#### 7.1 Główne Sekcje
```json
{
  "version": "2.0",
  "analysis_type": "GTMØ",
  "timestamp": "ISO 8601",
  "content": {
    "text": "Analizowany tekst",
    "length": 123,
    "word_count": 45
  },
  "coordinates": {
    "determination": 0.789,
    "stability": 0.689,
    "entropy": 0.337
  },
  "constitutional_metrics": { ... },
  "quantum_metrics": { ... },
  "axiom_protection": { ... }
}
```

#### 7.2 Constitutional Metrics
```json
"constitutional_metrics": {
  "definiteness": {
    "value": 1.1976,
    "formula": "(1/3.29) × 3 × √(0.799×0.694/0.322) = 1.20",
    "interpretation": "Wysoka CD = tekst uporządkowany",
    "components": {
      "inverse_ambiguity": 0.3043,
      "depth": 3,
      "geometric_balance": 1.3117
    }
  },
  "indefiniteness": {
    "value": 7.5148,
    "formula": "3.29 × 3 × √(0.322/(0.799×0.694)) = 7.51",
    "interpretation": "Wysoka CI = tekst chaotyczny",
    "decomposition": {
      "morphological": {"value": 2.5049, "percentage": 30.26},
      "syntactic": {"value": 2.2871, "percentage": 27.63},
      "semantic": {"value": 3.4851, "percentage": 42.11}
    }
  },
  "semantic_accessibility": {
    "value": 0.1331,
    "percentage": 13.31,
    "category": "NISKA_DOSTĘPNOŚĆ",
    "formula": "CD / Depth² = 1.20 / 9 = 0.133"
  },
  "duality": {
    "product": 9.0,
    "theoretical": 9,
    "error_percent": 0.0,
    "verification": "PASSED"
  },
  "classification": {
    "type": "CHAOTIC_STRUCTURE",
    "cd_ci_ratio": 0.1594
  }
}
```

### 8. Zależności i Instalacja

#### 8.1 Wymagane Biblioteki
```bash
pip install numpy
pip install morfeusz2
pip install spacy
python -m spacy download pl_core_news_lg
```

#### 8.2 Opcjonalne Biblioteki
```bash
pip install cmath  # Dla operacji na liczbach zespolonych
```

### 9. Użycie Podstawowe

#### 9.1 Import i Inicjalizacja
```python
from gtmo_morphosyntax import analyze_quantum_with_axioms

# Analiza pojedynczego tekstu
result = analyze_quantum_with_axioms("Rzeczpospolita Polska przestrzega prawa.")
```

#### 9.2 Analiza Wsadowa
```python
from gtmo_morphosyntax import batch_analyze_quantum_with_axioms

texts = ["Tekst 1", "Tekst 2", "Tekst 3"]
results = batch_analyze_quantum_with_axioms(texts, "source_file.md")
```

#### 9.3 Bezpośrednie Użycie Silnika
```python
engine = QuantumMorphosyntaxEngine()
result = engine.gtmo_analyze_quantum("Tekst do analizy")
```

### 10. Interpretacja Wyników

#### 10.1 Współrzędne Przestrzeni F³
- **D > 0.7:** Wysoka określoność semantyczna
- **S > 0.7:** Wysoka stabilność strukturalna  
- **E < 0.3:** Niska entropia (porządek)

#### 10.2 Constitutional Metrics
- **Dualność:** CI × CD = Depth² (błąd < 1%)
- **SA < 0.3:** Tekst trudno dostępny
- **CD > CI:** Struktura uporządkowana

#### 10.3 Quantum Metrics
- **Coherence > 0.8:** Silna koherencja kwantowa
- **Entanglements > 0:** Wykryte splątania semantyczne
- **Superposition Type:** Typ superpozycji kwantowej

### 11. Funkcje Utilitarne

#### 11.1 Ładowanie Plików Markdown
```python
def load_markdown_file(file_path: str) -> List[str]:
    """
    Ładuje i parsuje plik markdown na zdania.
    
    Args:
        file_path: Ścieżka do pliku .md
        
    Returns:
        Lista zdań z pliku
    """
```

#### 11.2 Klasyfikacja Interpretacji
```python
def _classify_interpretation(self, coords: Dict[str, float]) -> Dict[str, str]:
    """
    Klasyfikuje interpretację na podstawie współrzędnych.
    
    Returns:
        Dict z poziomami determination/stability/entropy i oceną ogólną
    """
```

### 12. Obsługa Błędów

#### 12.1 Wyjątki Specjalne
```python
class SingularityError(Exception):
    """Błąd związany z singularnością Ø"""

class EpistemicBoundaryError(Exception):
    """Błąd granic epistemicznych"""
```

#### 12.2 Fallback Mechanisms
- Analiza heurystyczna gdy spaCy niedostępne
- Backup coordinates gdy brak Morfeusz2
- Graceful degradation przy błędach

### 13. Wydajność i Optymalizacja

#### 13.1 Adaptive Weights
System automatycznie dostosowuje wagi morfologii/składni na podstawie:
- Długości tekstu
- Liczby słów
- Głębokości składniowej
- Poziomu ambiguity

#### 13.2 Quantum State Management
- Automatyczna dekoherencja w czasie
- Zarządzanie stanami splątanymi
- Optymalizacja pomiarów kwantowych

### 14. Rozszerzalność

#### 14.1 Dodawanie Nowych Aksjomatów
```python
def _ax13_new_axiom(self, state: Dict) -> Dict:
    """Template dla nowego aksjomatu"""
    # Implementacja logiki aksjomatu
    self._activate_axiom("AX13", "Reason for activation")
    return modified_state
```

#### 14.2 Rozszerzanie Metryki
System pozwala na dodawanie nowych metryk konstytucyjnych poprzez:
- Modyfikację sekcji constitutional_metrics
- Dodanie nowych formuł matematycznych
- Rozszerzenie systemu weryfikacji

### 15. Testy i Walidacja

#### 15.1 Test Integration
```python
# W pliku test_basic.py
def test_constitutional_metrics():
    result = analyze_quantum_with_axioms("Test text")
    assert 'constitutional_metrics' in result
    assert result['constitutional_metrics']['duality']['verification'] == 'PASSED'
```

#### 15.2 Benchmark Texts
System testowany na korpusie tekstów prawnych polskich:
- Konstytucja RP
- Ustawy podatkowe
- Orzeczenia sądowe
- Akty prawne UE

---

## Kontakt i Wsparcie

**Autor:** GSkuza  
**Repozytorium:** https://github.com/GSkuza/GTMO_MORPHOSYNTAX_PL  
**Licencja:** Zobacz plik LICENSE  
**Dokumentacja:** Ten plik

---

*GTMØ Quantum Morphosyntax Engine v2.0 - Complete Constitutional Metrics Implementation*