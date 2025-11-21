# GTMØ QUANTUM MORPHOSYNTAX ENGINE - Dokumentacja Techniczna

## Spis treści

1. [Wprowadzenie](#1-wprowadzenie)
2. [Architektura systemu](#2-architektura-systemu)
3. [Fundamenty teoretyczne](#3-fundamenty-teoretyczne)
4. [Implementacja 13 Aksjomatów GTMØ](#4-implementacja-13-aksjomatów-gtmø)
5. [Analiza morfosyntaktyczna](#5-analiza-morfosyntaktyczna)
6. [Metryki kwantowe](#6-metryki-kwantowe)
7. [Analiza retoryczna](#7-analiza-retoryczna)
8. [Metryki konstytucyjne](#8-metryki-konstytucyjne)
9. [Integracja z modelami językowymi](#9-integracja-z-modelami-językowymi)
10. [API i użycie](#10-api-i-użycie)
11. [Formaty wyjściowe](#11-formaty-wyjściowe)
12. [Rozszerzenia i moduły](#12-rozszerzenia-i-moduły)
13. [Optymalizacja i wydajność](#13-optymalizacja-i-wydajność)
14. [Przykłady użycia](#14-przykłady-użycia)

---

## 1. Wprowadzenie

### 1.1 Cel systemu

GTMØ Quantum Morphosyntax Engine to zaawansowany system analizy lingwistycznej dla języka polskiego, łączący:

- **Morfosyntaktykę tradycyjną** (Morfeusz2, spaCy)
- **Semantykę kwantową** (superpozycja stanów, splątanie)
- **Geometrię topologiczną** (przestrzeń fazowa D-S-E)
- **System aksjomatyczny** (13 Aksjomatów GTMØ jako "system immunologiczny")

### 1.2 Główne komponenty

```
gtmo_morphosyntax.py           # Główny silnik
├── GTMOAxiomSystem            # System 13 aksjomatów
├── QuantumMorphosyntaxEngine  # Silnik analizy kwantowej
├── QuantumAmbiguityAnalyzer   # Detekcja niejednoznaczności
└── EnhancedGTMOProcessor      # Procesor z integracją Stanza
```

### 1.3 Zależności

**Wymagane:**
```python
morfeusz2                  # Analiza morfologiczna (PAN)
spacy                      # Analiza syntaktyczna
pl_core_news_lg            # Model języka polskiego dla spaCy
numpy                      # Operacje numeryczne
```

**Opcjonalne:**
```python
transformers               # HerBERT (embeddingi semantyczne)
torch                      # Backend dla HerBERT
sentence-transformers      # Coherence measurement
stanza                     # Zaawansowana analiza składniowa
```

---

## 2. Architektura systemu

### 2.1 Przepływ danych

```
TEXT INPUT
    ↓
[Morphological Analysis] → Morfeusz2 → disambiguation
    ↓
[Syntactic Analysis] → spaCy/Stanza → dependency parsing
    ↓
[Temporal Analysis] → verb tense/aspect extraction
    ↓
[Rhetorical Analysis] → irony/paradox detection
    ↓
[Quantum State] → superposition, entanglement
    ↓
[Axiom System] → 13 axioms validation
    ↓
[Constitutional Metrics] → CD-CI duality
    ↓
[Topological Classification] → attractor analysis
    ↓
JSON OUTPUT (Φ⁹ coordinates + metadata)
```

### 2.2 Przestrzeń fazowa GTMØ

**Współrzędne F³ (Phase Space):**

```
D (Determination)  : [0,1] - precyzja semantyczna
S (Stability)      : [0,1] - stabilność strukturalna
E (Entropy)        : [0,1] - entropia/chaos informacyjny
```

**Transformacja do Φ⁹:**

```python
Φ⁹ = (D, S, E, θ_D, φ_D, ρ_D, θ_S, φ_S, ρ_S)
```

Gdzie:
- `(D, S, E)` - współrzędne fazowe
- `(θ_D, φ_D, ρ_D)` - współrzędne sferyczne dla Determination
- `(θ_S, φ_S, ρ_S)` - współrzędne sferyczne dla Stability

**Ekstakcja przez iterację Julii:**

```python
z_{n+1} = z_n² + c
c = (-0.8 + 0.156i) × (1 + E)
```

### 2.3 Singularność GTMØ (Ø)

```
Ø = (1.0, 1.0, 0.0)  # "Punkt niemożliwy"
```

Właściwości:
- **Nieosiągalna** bezpośrednio przez operacje standardowe
- **Graniczna** topologicznie (∂Ψ)
- **Chronienna** przez system aksjomatów

---

## 3. Fundamenty teoretyczne

### 3.1 Korelacja D-S-E

**KRYTYCZNE:** Wysokie D MUSI korelować z wysokim S!

```
Jeśli D > 0.7 → S ≥ D - 0.1
```

**Uzasadnienie:**
- Precyzyjne znaczenie (wysokie D) jest z natury **stabilne** (wysokie S)
- Język formalny/prawniczy: wysokie D ∧ wysokie S ∧ niskie E
- Język poetycki: średnie D ∧ niskie S ∧ wysokie E

### 3.2 Entropia semantyczna (E)

**Składowe entropii:**

```python
E = 0.4 × polysemy + 0.4 × syntactic_ambiguity + 0.2 × incoherence
```

1. **Polysemy** - wieloznaczność leksykalna (w kontekście!)
2. **Syntactic ambiguity** - niejednoznaczność składniowa (PP-attachment, etc.)
3. **Incoherence** - 1 - spójność między zdaniami

**FIX:** Terminy domenowe są **monosemiczne w kontekście**!

Przykład:
- "powód" w kontekście prawnym → **jedno** znaczenie (plaintiff)
- "powód" w kontekście ogólnym → **wiele** znaczeń (cause, reason, plaintiff)

### 3.3 Quantum superposition semantics

**Kiedy stosować superpozycję?**

```python
if ambiguity_score > 2.0:
    use_quantum_superposition = True
```

Wskaźniki ambiguity:
- Markery lingwistyczne: "może", "chyba", "prawdopodobnie"
- Znaki zapytania: `?`
- Wielokrotność interpretacji morfologicznych

**Reprezentacja:**

```python
|ψ⟩ = Σ_i √p_i × e^(iφ_i) |interpretation_i⟩
```

- `p_i` - prawdopodobieństwo interpretacji
- `φ_i` - faza kwantowa (koherencja)

**Koherencja kwantowa:**

```
coherence = |Σ_i amplitude_i|² / Σ_i |amplitude_i|²
```

- `coherence = 1.0` → **KOHERENTNY** (fazy wyrównane)
- `coherence = 0.0` → **DEKOHERENTNY** (fazy losowe)

---

## 4. Implementacja 13 Aksjomatów GTMØ

### 4.1 Przegląd systemu aksjomatycznego

System aksjomatów działa jako **"system immunologiczny"** - chroni integralność przestrzeni semantycznej.

```python
class GTMOAxiomSystem:
    def execute_all_axioms(self, system_state: Dict) -> Dict:
        """Wykonaj wszystkie 13 aksjomatów sekwencyjnie"""
```

### 4.2 Lista aksjomatów

| ID | Nazwa | Funkcja |
|----|-------|---------|
| AX0 | Systemic Uncertainty | Wprowadź kwantową niepewność |
| AX1 | Ontological Difference | Zapobiegaj zbliżeniu do Ø |
| AX2 | Translogical Isolation | Blokuj ścieżki obliczeniowe do Ø |
| AX3 | Epistemic Singularity | Zapobiegaj twierdzeniom o "znajomości" Ø |
| AX4 | Non-Representability | Blokuj standardową reprezentację Ø |
| AX5 | Topological Boundary | Utrzymuj Ø na brzegu ∂Ψ |
| AX6 | Heuristic Extremum | Wymuszaj minimalną entropię przy Ø |
| AX7 | Meta-Closure | Wyzwalaj meta-refleksję przy Ø |
| AX8 | Not Limit Point | Zapobiegaj zbieżności trajektorii do Ø |
| AX9 | Operator Irreducibility | Blokuj operatory standardowe przy Ø |
| AX10 | Meta-Operator Definition | Pozwalaj tylko meta-operatory przy Ø |
| AX11 | Adaptive Learning | Ucz się z napotkanych granic |
| AX12 | Topological Classification | Klasyfikuj przez atraktory topologiczne |

### 4.3 Przykłady działania aksjomatów

#### AX1: Ontological Difference

```python
def _ax1_ontological_difference(self, state: Dict) -> Dict:
    coords = self._extract_coordinates(state)
    distance_to_singularity = np.linalg.norm(coords - SINGULARITY_COORDS)
    
    if distance_to_singularity < 0.05:  # Za blisko Ø!
        direction = coords - SINGULARITY_COORDS
        push_direction = direction / np.linalg.norm(direction)
        new_coords = coords + push_direction * 0.02  # Odepchnij
        return self._update_coordinates(state, new_coords)
```

**Efekt:** Jeśli analiza zbliża się do Ø (1.0, 1.0, 0.0), system **automatycznie ją odsuwa**.

#### AX7: Meta-Closure

```python
def _ax7_meta_closure(self, state: Dict) -> Dict:
    distance_to_singularity = np.linalg.norm(coords - SINGULARITY_COORDS)
    
    if distance_to_singularity < 0.1:
        # Wyzwól meta-refleksję
        meta_uncertainty = min(0.1, (0.1 - distance_to_singularity) * 2)
        new_coords = coords + np.array([-meta_uncertainty, -meta_uncertainty, meta_uncertainty])
        
        state['meta_reflection'] = {
            'triggered_by': 'AX7_proximity_to_singularity',
            'uncertainty_applied': meta_uncertainty
        }
```

**Efekt:** Przy zbliżeniu do Ø system **zwiększa niepewność** (meta-cognitive doubt).

#### AX12: Topological Classification

**Atraktory topologiczne:**

```python
self.attractors = {
    'Ψᴷ': np.array([0.85, 0.85, 0.15]),  # Wiedza naukowa
    'Ψʰ': np.array([0.15, 0.15, 0.85]),  # Chaos/poetry
    'Ψᴺ': np.array([0.50, 0.30, 0.90]),  # Negacja
    'Ø':  np.array([1.00, 1.00, 0.00]),  # Singularność
    'Ψ~': np.array([0.50, 0.50, 0.80]),  # Superpozycja
}
```

**Baseny przyciągania:**

```python
self.attractor_basins = {
    'Ψᴷ': 0.15, 'Ψʰ': 0.20, 'Ψᴺ': 0.25, 
    'Ø': 0.10, 'Ψ~': 0.18
}
```

**Klasyfikacja:**

```python
if distance_to_attractor <= basin_radius:
    # Stan jest w basenie przyciągania
    direction = (attractor_coords - coords) / distance
    pull_strength = 0.01 * (1 - distance / basin_radius)
    new_coords = coords + direction * pull_strength
```

---

## 5. Analiza morfosyntaktyczna

### 5.1 Analiza morfologiczna (Morfeusz2)

**Zadanie:** Dezambiguacja form fleksyjnych polskiego.

```python
def analyze_morphology_quantum(self, text: str) -> Tuple[np.ndarray, Dict, Dict]:
    analyses = morfeusz.analyse(text)
    
    # KRYTYCZNE: Wybierz JEDNĄ interpretację, nie wszystkie!
    best_analysis = self.disambiguate_morfeusz(word, analyses_list, sentence)
```

**Strategia dezambiguacji:**

1. **spaCy POS tagging** - mapowanie POS → Morfeusz
2. **Domain dictionary** - terminologia dziedzinowa
3. **Heurystyka częstości** - preferencja: `subst > verb > adj > adv`

**Dekompozycja morfonologiczna:**

```python
def extract_true_morphemes(self, analysis: Tuple) -> Dict[str, np.ndarray]:
    # Wydobądź morfemy: root, suffix, inflection
    morpheme_coords = {
        'root': np.array([0.6, 0.65, 0.4]),      # Polisemiczny ale stabilny
        'suffix': np.array([0.90, 0.92, 0.10]),  # Precyzyjny i stabilny!
    }
```

**Kompozycja tensorowa (NIE uśrednianie!):**

```python
def tensor_product_composition(self, morpheme_coords: List[np.ndarray]) -> np.ndarray:
    # Średnia geometryczna (tensor product proxy)
    composed = np.ones(3)
    for coords in morpheme_coords:
        composed *= coords
    composed = np.power(composed, 1.0 / len(morpheme_coords))
```

**Dlaczego geometryczna?**

- Zachowuje proporcje (D/S/E ratios)
- Morfologia **zwiększa precyzję** (suffix dodaje determination)
- Arytmetyczna niszczy informację

### 5.2 Analiza syntaktyczna (spaCy/Stanza)

**Zadanie:** Ekstakcja struktury zależnościowej.

```python
def analyze_syntax_quantum(self, text: str) -> Tuple[np.ndarray, Dict, List]:
    doc = nlp(text)
    
    # Liczenie typów zależności
    dep_counts = {}
    for token in doc:
        dep_counts[token.dep_] = dep_counts.get(token.dep_, 0) + 1
```

**Calculation D-S-E:**

```python
# DETERMINATION
core_deps = ['ROOT', 'nsubj', 'obj', 'iobj', 'csubj']
core_count = sum(dep_counts.get(dep, 0) for dep in core_deps)
determination = 0.4 + (core_count / total_tokens) * 1.5

# STABILITY
num_dep_types = len(dep_counts)
if num_dep_types < 4:
    stability = 0.75
elif num_dep_types < 8:
    stability = 0.6

# ENTROPY
entropy = 0.25  # base
if max_depth > 2:
    entropy += min((max_depth - 2) * 0.08, 0.25)
```

**Fallback heuristic** (bez spaCy):

```python
def fallback_syntax_analysis(text: str) -> Tuple[np.ndarray, Dict]:
    avg_sent_len = word_count / sent_count
    
    if avg_sent_len < 7:
        determination = 0.75  # Proste zdania
    elif avg_sent_len < 15:
        determination = 0.6
    else:
        determination = 0.45  # Złożone okresy
```

### 5.3 Adaptacyjne wagi fuzji

**Problem:** Jak łączyć morph_coords i synt_coords?

```python
def calculate_adaptive_weights(self, text: str, morph_meta: Dict, synt_meta: Dict):
    word_count = len(text.split())
    
    if word_count <= 3:
        morph_weight = 0.40  # Krótkie → więcej składni
    elif word_count <= 15:
        morph_weight = 0.64  # Standardowa
    elif word_count <= 30:
        morph_weight = 0.55
    else:
        morph_weight = 0.45  # Długie → składnia dominuje
    
    synt_weight = 1.0 - morph_weight
```

**Korekta kwantowa:**

```python
morph_weight += 0.1 * total_quantum_coherence
```

**Finalna fuzja:**

```python
final_coords = morph_weight * morph_coords + synt_weight * synt_coords
```

---

## 6. Metryki kwantowe

### 6.1 Superpozycja stanów

**Reprezentacja kwantowa:**

```python
class QuantumSemanticState:
    amplitudes: Dict[str, complex]  # case → amplitude
    phase: float                    # Faza globalna
    coherence: float                # [0,1]: 1=koherentny, 0=dekoherentny
    entangled_with: List[str]       # Splątane słowa
    measurement_count: int          # Liczba kolapsów
```

**Tworzenie superpozycji:**

```python
def create_superposition_state(self, word: str, case_frequencies: Dict):
    amplitudes = {}
    for case, freq in case_frequencies.items():
        probability = freq / total_observations
        phase_variation = np.random.uniform(-0.1, 0.1)  # Mała wariancja!
        amplitude = np.sqrt(probability) * cmath.exp(1j * phase_variation)
        amplitudes[case] = amplitude
```

**FIX:** Po dezambiguacji fazy są **WYRÓWNANE** (koherentne)!

### 6.2 Kolaps superpozycji

**Geometryczna kompozycja (NIE arytmetyczna!):**

```python
def collapse_superposition(self, word: str, observed_cases: List[str]) -> np.ndarray:
    composed = np.ones(3)
    total_weight = 0
    
    for coords, prob in case_coords_with_probs:
        composed *= np.power(coords, prob)  # Geometryczna waga
        total_weight += prob
    
    collapsed_coords = np.power(composed, 1.0 / total_weight)
```

### 6.3 Splątanie kwantowe

**Detekcja splątania:**

```python
def measure_quantum_interference(self, word1: str, word2: str) -> float:
    combined = state1.amplitude + state2.amplitude
    individual_sum = |state1.amplitude|² + |state2.amplitude|²
    combined_prob = |combined|²
    
    interference = |combined_prob - individual_sum|
```

**Tworzenie splątania:**

```python
if interference > ENTANGLEMENT_THRESHOLD:  # 0.7
    self.create_entanglement(word1, word2, coupling_strength=0.5)
```

**Efekt splątania:**

```python
for case in state1.amplitudes:
    coupled_amp = coupling_strength * (state1.amplitudes[case] + state2.amplitudes[case])
    state1.amplitudes[case] = coupled_amp
    state2.amplitudes[case] = coupled_amp.conjugate()
```

### 6.4 Decoherence (dekoherencja)

**Spadek koherencji w czasie:**

```python
def _apply_decoherence(self, word: str):
    if measurement_count < 2:
        decoherence_rate = DECOHERENCE_RATE * 0.1  # Minimalna po dezambiguacji
    else:
        decoherence_rate = DECOHERENCE_RATE  # 0.02
    
    quantum_state.coherence *= (1 - decoherence_rate)  # MALEJE!
    quantum_state.coherence = max(coherence, 0.5)      # Minimum
```

---

## 7. Analiza retoryczna

### 7.1 Detekcja ironii/sarkazmu

**Zasada:** Ironia = **INWERSJA** współrzędnych D-S-E!

```python
def detect_enhanced_rhetorical_mode(text, base_coords, morph_metadata):
    # Markery ironii
    irony_markers = ['oczywiście', 'jakże', 'ależ', 'no pewnie', 'super']
    
    # Wzorce sarkazmu
    sarcasm_patterns = [
        r'\b(super|świetny|genialny)\b.*\b(ale|tylko|że)\b',
        r'\".*\"',  # Cudzysłów często sarkastyczny
        r'!{2,}'    # Wielokrotne wykrzykniki
    ]
    
    # Kontekst negatywny + słowa pozytywne = IRONIA
    if has_positive and has_negative_context:
        irony_score += 2
```

**Transformacja dla ironii:**

```python
if total_irony_sarcasm > 1.5:  # Próg obniżony dla lepszej detekcji
    inverted_coords = np.array([
        1.0 - base_coords[0],  # D: pewność → niepewność
        1.0 - base_coords[1],  # S: stabilność → niestabilność
        1.0 - base_coords[2]   # E: entropia → neg-entropia
    ])
    return inverted_coords, 'irony', metadata
```

### 7.2 Detekcja paradoksu

**Zasada:** Paradoks = **ZACHOWANIE** współrzędnych + wzrost entropii!

```python
elif paradox_score > 1:
    paradox_coords = base_coords.copy()
    paradox_coords[2] = min(paradox_coords[2] * 1.5, 1.0)  # Zwiększ entropię
    return paradox_coords, 'paradox', metadata
```

**Markery paradoksu:**

```python
paradox_markers = [
    'jednocześnie', 'zarazem', 'a jednak', 'mimo to',
    'wbrew', 'choć', 'aczkolwiek', 'paradoksalnie'
]
```

### 7.3 Moduł GTMORhetoricalAnalyzer

**Zaawansowana analiza (opcjonalna):**

```python
if self.rhetorical_analyzer:
    transformed_coords, rhetorical_mode, rhetorical_metadata = \
        self.rhetorical_analyzer.analyze_rhetorical_mode(
            text=text,
            morph_coords=final_coords,
            syntax_coords=syntax_coords
        )
```

**Dodatkowe metryki:**

- `structural_divergence` - rozbieżność morph/syntax
- `pos_anomalies` - anomalie POS (adj_ratio, verb_ratio)
- `irony_indicators` - szczegółowe wskaźniki ironii

---

## 8. Metryki konstytucyjne

### 8.1 Dualność CD-CI

**Constitutional Definiteness (CD):**

```
CD = D × S × (1 / Ambiguity)
```

**Constitutional Indefiniteness (CI):**

```
CI = E × Depth × Ambiguity
```

**Własność dualności:**

```
CI × CD ≈ Depth²
```

### 8.2 Dekompozycja CI

```python
CI_morphological = (morphological_ambiguity - 1) × 0.4
CI_syntactic = (depth - 1) × 0.4
CI_semantic = entropy × 0.2

CI = CI_morphological + CI_syntactic + CI_semantic
```

### 8.3 Semantic Accessibility (SA)

**SA v2.0:**

```python
SA = (CD / (CD + CI)) × kinetic_power
```

**SA v3.0 (z HerBERT):**

```python
if HERBERT_AVAILABLE:
    embedding = get_herbert_embedding(text)
    complexity = np.linalg.norm(embedding)
    SA_v3 = SA_v2 × (1 - complexity_penalty)
```

**Kategorie SA:**

```python
class SACategory(Enum):
    VERY_HIGH = "very_high"      # SA > 0.8
    HIGH = "high"                # 0.6 < SA ≤ 0.8
    MODERATE = "moderate"        # 0.4 < SA ≤ 0.6
    LOW = "low"                  # 0.2 < SA ≤ 0.4
    VERY_LOW = "very_low"        # SA ≤ 0.2
```

### 8.4 Klasyfikacja strukturalna

```python
if cd_ci_ratio > 2.0:
    return StructureClassification.HIGHLY_DETERMINED
elif cd_ci_ratio > 1.0:
    return StructureClassification.DETERMINED
elif cd_ci_ratio > 0.5:
    return StructureClassification.BALANCED
elif cd_ci_ratio > 0.2:
    return StructureClassification.INDEFINITE
else:
    return StructureClassification.HIGHLY_INDEFINITE
```

### 8.5 Geometric balance & tension

**Geometric balance:**

```
GB = √(CD × CI)  # Średnia geometryczna
```

**Geometric tension:**

```
GT = |CD - CI| / (CD + CI)  # Różnica względna
```

---

## 9. Integracja z modelami językowymi

### 9.1 HerBERT (embeddingi semantyczne)

**Ładowanie modelu:**

```python
from transformers import AutoTokenizer, AutoModel
herbert_tokenizer = AutoTokenizer.from_pretrained("allegro/herbert-base-cased")
herbert_model = AutoModel.from_pretrained("allegro/herbert-base-cased")
herbert_model.eval()
```

**Ekstakcja embeddingu:**

```python
with torch.no_grad():
    inputs = herbert_tokenizer(text, return_tensors="pt", truncation=True, max_length=512)
    outputs = herbert_model(**inputs)
    embedding = outputs.last_hidden_state[:, 0, :].squeeze().numpy()
```

**Zastosowania:**

1. **Polysemy measurement** - porównanie kontekstowe vs ogólne
2. **SA v3.0** - penalty za complexity
3. **Semantic flow** - analiza koherencji między zdaniami

### 9.2 Sentence-BERT (koherencja)

**Model:**

```python
from sentence_transformers import SentenceTransformer
sentence_model = SentenceTransformer('paraphrase-multilingual-mpnet-base-v2')
```

**Pomiar koherencji:**

```python
embeddings = sentence_model.encode(sentence_texts)

for i in range(len(embeddings) - 1):
    cos_sim = np.dot(emb1, emb2) / (norm1 * norm2)
    similarities.append(cos_sim)

coherence = np.mean(similarities)
```

### 9.3 Stanza (zaawansowana składnia)

**Pipeline:**

```python
import stanza
stanza_nlp = stanza.Pipeline('pl', processors='tokenize,pos,lemma,depparse')
```

**Detekcja "smoking guns" (prawniczy):**

```python
# Wykryj: "nie popełnił" → "skazuje" (SPRZECZNOŚĆ!)
if negated_verbs and affirming_verbs:
    smoking_guns.append({
        'type': 'negation_consequence_conflict',
        'severity': 0.95,
        'conflict': f"'{neg['text']}' → '{aff['text']}'"
    })
```

---

## 10. API i użycie

### 10.1 Główne funkcje

**Pojedyncza analiza:**

```python
def analyze_quantum_with_axioms(text: str, source_file: str = "unknown") -> Dict:
    """
    Główna funkcja analizy z aksjomatami GTMØ
    
    Args:
        text: Tekst do analizy (zdanie lub dłuższy fragment)
        source_file: Nazwa pliku źródłowego (dla metadanych)
    
    Returns:
        Dict z pełną analizą (współrzędne, metryki, metadata)
    """
    engine = _get_global_engine()
    return engine.gtmo_analyze_quantum(text, source_file)
```

**Analiza wsadowa:**

```python
def batch_analyze_quantum_with_axioms(texts: List[str], source_file: str = "batch") -> List[Dict]:
    """
    Analiza wsadowa z aksjomatami GTMØ
    
    Args:
        texts: Lista tekstów do analizy
        source_file: Nazwa pliku źródłowego
    
    Returns:
        Lista słowników z wynikami
    """
```

### 10.2 Klasa QuantumMorphosyntaxEngine

**Inicjalizacja:**

```python
engine = QuantumMorphosyntaxEngine(
    domain_dictionary=None,              # Opcjonalny słownik dziedzinowy
    herbert_tokenizer=GLOBAL_HERBERT_TOKENIZER,
    herbert_model=GLOBAL_HERBERT_MODEL
)
```

**Główna metoda:**

```python
result = engine.gtmo_analyze_quantum(
    text="Tekst do analizy",
    source_file={'name': 'document.txt', 'path': '/path/to/file'}
)
```

### 10.3 Klasa EnhancedGTMOProcessor

**Analiza tekstu prawniczego:**

```python
processor = EnhancedGTMOProcessor()
result = processor.analyze_legal_text("Sąd uznał, że oskarżony nie popełnił czynu. Jednak go skazuje.")
```

**Wynik zawiera:**

```python
{
    'gtmo_coordinates': {'determination': 0.65, 'stability': 0.70, 'entropy': 0.35},
    'stanza_analysis': {
        'sentences': [...],
        'smoking_guns': [...]  # Wykryte sprzeczności
    },
    'legal_assessment': {
        'quality': 'poor',
        'legal_coherence_score': 0.45,
        'smoking_gun_count': 1
    }
}
```

---

## 11. Formaty wyjściowe

### 11.1 Główna struktura JSON

```json
{
  "version": "2.0",
  "analysis_type": "GTMØ",
  "timestamp": "2025-11-21T15:12:10.123456",
  
  "content": {
    "text": "Tekst analizowany...",
    "length": 150,
    "word_count": 25
  },
  
  "coordinates": {
    "determination": 0.752341,
    "stability": 0.698234,
    "entropy": 0.287456
  },
  
  "phi9": [0.752, 0.698, 0.287, 1.234, 0.456, 0.123, 2.345, 0.789, 0.234],
  
  "quantum_tensor": {
    "value": 0.382456,
    "formula": "0.752 × 0.698 × 0.713 = 0.382"
  }
}
```

### 11.2 Metryki morfosyntaktyczne

```json
"additional_metrics": {
  "total_analyses": 45,
  "cases": {"nom": 8, "gen": 5, "acc": 7, "ins": 2},
  "pos": {"subst": 12, "verb": 8, "adj": 5},
  "ambiguity": 1.8,
  "debug_tags": ["sprawa:subst", "sąd:subst", "orzec:verb"]
}
```

### 11.3 Metryki kwantowe

```json
"quantum_metrics": {
  "total_coherence": 0.8234,
  "quantum_words": 15,
  "entanglements": 3,
  "superposition_type": "COHERENT_SUPERPOSITION"
},

"quantum_enhanced": {
  "num_quantum_states": 15,
  "coherence_detailed": {
    "phase_coherence": 0.823,
    "amplitude_coherence": 0.856,
    "overall_coherence": 0.839
  },
  "entanglement": {
    "mean_entanglement": 0.234,
    "max_entanglement": 0.678,
    "entangled_pairs": 3
  },
  "quantum_classification": "WEAKLY_ENTANGLED"
}
```

### 11.4 Metryki konstytucyjne

```json
"constitutional_metrics": {
  "CD": 0.4523,
  "CI": 0.8234,
  "cd_ci_ratio": 0.5493,
  "duality_product": 3.7234,
  "duality_theoretical": 4.0,
  "duality_error": 0.0692,
  
  "SA": 0.3545,
  "SA_v3": 0.3123,
  "sa_category": "low",
  
  "CI_morphological": 0.32,
  "CI_syntactic": 1.20,
  "CI_semantic": 0.06,
  
  "structure_classification": "indefinite",
  "geometric_balance": 0.6234,
  "geometric_tension": 0.2901
}
```

### 11.5 Analiza retoryczna

```json
"rhetorical_analysis": {
  "mode": "irony",
  "irony_score": 4.5,
  "paradox_score": 0.0,
  "structural_divergence": 0.234,
  "coordinate_inversion": {
    "original": {"determination": 0.75, "stability": 0.70, "entropy": 0.28},
    "inverted": {"determination": 0.25, "stability": 0.30, "entropy": 0.72}
  },
  "irony_indicators": ["oczywiście", "jak zwykle"]
}
```

### 11.6 Atraktory topologiczne

```json
"topological_attractors": {
  "nearest_attractor": {
    "attractor_name": "Ψᴷ",
    "coordinates": [0.85, 0.85, 0.15],
    "distance": 0.123,
    "in_basin": true,
    "basin_radius": 0.15
  },
  "basin_analysis": {
    "in_any_basin": true,
    "basin_probabilities": {
      "Ψᴷ": 0.85,
      "Ψʰ": 0.05,
      "Ψᴺ": 0.10
    }
  }
}
```

### 11.7 Ochrona aksjomatyczna

```json
"axiom_protection": {
  "axioms_activated": 3,
  "violations_prevented": 1,
  "meta_reflections": 0
}
```

---

## 12. Rozszerzenia i moduły

### 12.1 gtmo_pure_rhetoric.py

**Zaawansowana analiza retoryczna:**

- Detekcja ironii przez rozbieżność morph/syntax
- Analiza anomalii POS (adj_ratio, verb_ratio)
- Pattern matching dla sarkazmu

### 12.2 gtmo_domain_dictionary.py

**Słownik terminologii dziedzinowej:**

```python
domain_dict = DomainDictionary()
domain_dict.add_domain_term("powód", ["subst:sg:nom:m1"], domain="legal")

if domain_dict.is_domain_term("powód"):
    # Traktuj jako monosemiczny w kontekście prawnym
    polysemy_score = 0.1
```

### 12.3 gtmo_constitutional_duality.py

**Kalkulator metryk konstytucyjnych:**

```python
calculator = ConstitutionalDualityCalculator(use_sa_v3=True)
metrics = calculator.calculate_metrics(
    ambiguity=1.8,
    depth=4,
    D=0.75, S=0.70, E=0.28,
    text=text
)
```

### 12.4 gtmo_topological_attractors.py

**Analiza atraktorów i ewolucji:**

```python
analyzer = TopologicalAttractorAnalyzer()
nearest_name, distance, meta = analyzer.find_nearest_attractor(coords)

temporal_evolution = TemporalEvolutionAnalyzer(history_size=100)
temporal_evolution.add_state(coords, timestamp=time.time())
evolution_summary = temporal_evolution.get_evolution_summary()
```

### 12.5 gtmo_quantum_enhanced.py

**Zaawansowane metryki kwantowe:**

```python
from gtmo_quantum_enhanced import analyze_quantum_enhanced

result = analyze_quantum_enhanced(
    text=text,
    words=words,
    coords_per_word=coords_list,
    base_coherence=0.85
)
```

### 12.6 gtmo_file_loader.py

**Ładowanie plików Markdown:**

```python
from gtmo_file_loader import load_markdown_file

articles = load_markdown_file("document.md")
# Returns: List[str] - lista artykułów/paragrafów
```

### 12.7 gtmo_json_saver.py

**Zoptymalizowany zapis JSON:**

```python
from gtmo_json_saver import GTMOOptimizedSaver

saver = GTMOOptimizedSaver()
analysis_folder = saver.create_analysis_folder("document.md")

# Zapisz pojedyncze zdanie
saver.save_sentence_analysis(result, sentence_text, sentence_number)

# Zapisz artykuł
saver.save_article_analysis(result, article_text, article_number)

# Zapisz pełny dokument
saver.save_full_document_analysis(source_file, articles, article_analyses)

# Finalizuj embeddingi i macierze
embeddings_file = saver.finalize_embeddings()
matrices_file = saver.finalize_matrices()
herbert_analysis = saver.create_herbert_analysis()
```

---

## 13. Optymalizacja i wydajność

### 13.1 Lazy loading modeli

**Problem:** HerBERT (768 MB) + spaCy (500 MB) = długi czas ładowania

**Rozwiązanie:**

```python
# Global instance (loaded once at import)
GLOBAL_HERBERT_TOKENIZER = herbert_tokenizer
GLOBAL_HERBERT_MODEL = herbert_model

def _get_global_engine():
    global _GLOBAL_ENGINE
    if _GLOBAL_ENGINE is None:
        _GLOBAL_ENGINE = QuantumMorphosyntaxEngine(
            herbert_tokenizer=GLOBAL_HERBERT_TOKENIZER,
            herbert_model=GLOBAL_HERBERT_MODEL
        )
    return _GLOBAL_ENGINE
```

### 13.2 Optymalizacja embeddingów

**Problem:** Embeddingi (768 float64) × 1000 zdań = ~6 MB × JSON overhead

**Rozwiązanie:**

```python
# Osobny plik binary dla embeddingów
embeddings_storage = {
    'sentence_1': embedding_array,  # numpy array
    'sentence_2': embedding_array
}

# W głównym JSON tylko referencja
result['herbert_embedding_ref'] = 'sentence_1'
```

### 13.3 Batch processing

**Zalecenia:**

- Dziel długie dokumenty na artykuły/paragrafy
- Zapisuj każde zdanie osobno (unikaj utraty danych przy crash)
- Finalizuj embeddingi na koniec (`saver.finalize_embeddings()`)

### 13.4 Memory management

**Dla długich dokumentów (>1000 zdań):**

```python
# Opróżnij cache co 100 zdań
if sentence_counter % 100 == 0:
    saver.flush_embeddings()
    gc.collect()
```

### 13.5 Skip logic dla złożonych struktur

**Problem:** Paragrafy >20 zdań mogą powodować hang w constitutional calculator

**Rozwiązanie:**

```python
if len(sentences) > 20 or len(paragraph) > 5000:
    print(f"Skipping paragraph-level analysis (too large)")
    para_result = {
        'skipped': True,
        'skip_reason': f'Too large: {len(sentences)} sentences'
    }
```

---

## 14. Przykłady użycia

### 14.1 Analiza pojedynczego zdania

```python
from gtmo_morphosyntax import analyze_quantum_with_axioms

text = "Rzeczpospolita Polska przestrzega wiążącego ją prawa międzynarodowego."
result = analyze_quantum_with_axioms(text, source_file="constitution.txt")

print(f"Coordinates: D={result['coordinates']['determination']:.3f}, "
      f"S={result['coordinates']['stability']:.3f}, "
      f"E={result['coordinates']['entropy']:.3f}")

print(f"CD={result['constitutional_metrics']['CD']:.3f}, "
      f"CI={result['constitutional_metrics']['CI']:.3f}")

print(f"SA={result['constitutional_metrics']['SA']:.3f} ({result['constitutional_metrics']['sa_category']})")
```

**Output:**

```
Coordinates: D=0.823, S=0.791, E=0.156
CD=0.651, CI=0.624
SA=0.511 (moderate)
```

### 14.2 Detekcja ironii

```python
text = "Super, znowu się spóźniłeś. Oczywiście nic się nie stało."
result = analyze_quantum_with_axioms(text)

if result['rhetorical_analysis']['mode'] == 'irony':
    print("IRONIA WYKRYTA!")
    print(f"Irony score: {result['rhetorical_analysis']['irony_score']}")
    print(f"Original D={result['rhetorical_analysis']['coordinate_inversion']['original']['determination']:.3f}")
    print(f"Inverted D={result['rhetorical_analysis']['coordinate_inversion']['inverted']['determination']:.3f}")
```

**Output:**

```
IRONIA WYKRYTA!
Irony score: 4.5
Original D=0.752
Inverted D=0.248
```

### 14.3 Analiza dokumentu Markdown

```python
from gtmo_file_loader import load_markdown_file
from gtmo_morphosyntax import analyze_quantum_with_axioms
from gtmo_json_saver import GTMOOptimizedSaver

# Load document
articles = load_markdown_file("constitution.md")
saver = GTMOOptimizedSaver()
analysis_folder = saver.create_analysis_folder("constitution.md")

# Analyze each article
for i, article in enumerate(articles, 1):
    result = analyze_quantum_with_axioms(article, source_file="constitution.md")
    result['article_number'] = i
    
    # Save
    saved_file = saver.save_article_analysis(result, article, i)
    print(f"Article {i} → {saved_file}")

# Finalize
embeddings_file = saver.finalize_embeddings()
print(f"Embeddings saved to {embeddings_file}")
```

### 14.4 Analiza prawnicza z Stanza

```python
from gtmo_morphosyntax import EnhancedGTMOProcessor

processor = EnhancedGTMOProcessor()
text = """
Sąd uznał, że oskarżony nie popełnił zarzucanego mu czynu.
W związku z powyższym Sąd skazuje oskarżonego na karę 2 lat pozbawienia wolności.
"""

result = processor.analyze_legal_text(text)

print(f"Quality: {result['legal_assessment']['quality']}")
print(f"Coherence: {result['legal_assessment']['legal_coherence_score']:.3f}")

if result['stanza_analysis']['smoking_guns']:
    print("SMOKING GUNS:")
    for gun in result['stanza_analysis']['smoking_guns']:
        print(f"  - {gun['type']}: {gun['details']['conflict']}")
```

**Output:**

```
Quality: poor
Coherence: 0.345

SMOKING GUNS:
  - negation_consequence_conflict: 'nie popełnił' → 'skazuje'
```

### 14.5 Batch processing z progress tracking

```python
from tqdm import tqdm
from gtmo_morphosyntax import analyze_quantum_with_axioms

sentences = [
    "Zdanie pierwsze.",
    "Zdanie drugie.",
    # ... 1000 zdań
]

results = []
for i, sentence in enumerate(tqdm(sentences), 1):
    try:
        result = analyze_quantum_with_axioms(sentence, source_file=f"batch_{i}")
        results.append(result)
    except Exception as e:
        print(f"Error at sentence {i}: {e}")
        continue

print(f"Analyzed {len(results)}/{len(sentences)} sentences successfully")
```

### 14.6 Export do CSV (basic metrics)

```python
import json
import csv

# Load all sentence JSONs
sentence_files = glob.glob("analyses/*/sentence_*.json")

with open("results.csv", "w", newline='', encoding='utf-8') as f:
    writer = csv.writer(f)
    writer.writerow(['Sentence', 'D', 'S', 'E', 'CD', 'CI', 'SA', 'Depth', 'Ambiguity'])
    
    for file in sentence_files:
        with open(file, 'r', encoding='utf-8') as jf:
            data = json.load(jf)
            writer.writerow([
                data['content']['text'][:50],
                data['coordinates']['determination'],
                data['coordinates']['stability'],
                data['coordinates']['entropy'],
                data['constitutional_metrics']['CD'],
                data['constitutional_metrics']['CI'],
                data['constitutional_metrics']['SA'],
                data.get('depth', 0),
                data.get('ambiguity', 1.0)
            ])
```

---

## Dodatek A: Stałe i parametry

```python
# Fundamental constants
PHI = 1.618033988749895                    # Golden ratio
SQRT_2_INV = 0.7071067811865476            # 1/√2
SINGULARITY_COORDS = [1.0, 1.0, 0.0]       # Ø coordinates

# Cognitive reference points
COGNITIVE_CENTER = [0.5, 0.5, 0.5]         # Center of phase space

# Thresholds
ENTROPY_THRESHOLD_SINGULARITY = 0.001      # Max entropy at Ø
BOUNDARY_THICKNESS = 0.02                  # Topological boundary width
META_REFLECTION_THRESHOLD = 0.95           # Meta-cognition trigger
DECOHERENCE_RATE = 0.02                    # Quantum decoherence per measurement
ENTANGLEMENT_THRESHOLD = 0.7               # Quantum entanglement detection

# Adaptive fusion weights (sentence length dependent)
# Short (1-3 words):   morph=0.40, synt=0.60
# Medium (4-15):       morph=0.64, synt=0.36
# Long (16-30):        morph=0.55, synt=0.45
# Very long (30+):     morph=0.45, synt=0.55

# Julia set parameters (for Φ⁹)
JULIA_MAX_ITER = 1000
JULIA_R_ESCAPE = 2.0
JULIA_C_BASE = (-0.8, 0.156)
```

---

## Dodatek B: Case i POS coordinates

### Polish cases (GTMØ coordinates)

```python
CASE_COORDS = {
    'nom': [0.849, 0.271, 0.455],  # Nominative (mianownik)
    'gen': [0.787, 0.270, 0.456],  # Genitive (dopełniacz)
    'dat': [0.773, 0.357, 0.456],  # Dative (celownik)
    'acc': [0.836, 0.336, 0.450],  # Accusative (biernik)
    'ins': [0.708, 0.354, 0.468],  # Instrumental (narzędnik)
    'loc': [0.728, 0.282, 0.456],  # Locative (miejscownik)
    'voc': [0.683, 0.368, 0.458]   # Vocative (wołacz)
}
```

### Polish POS tags

```python
POS_COORDS = {
    'subst': [0.804, 0.477, 0.483],  # Noun
    'adj':   [0.747, 0.342, 0.477],  # Adjective
    'verb':  [0.763, 0.351, 0.478],  # Verb
    'adv':   [0.732, 0.383, 0.481],  # Adverb
    'num':   [0.835, 0.422, 0.486],  # Numeral
    'pron':  [0.712, 0.453, 0.484],  # Pronoun
    'prep':  [0.76,  0.75,  0.24 ],  # Preposition
    'conj':  [0.65,  0.85,  0.20 ],  # Conjunction
    'part':  [0.40,  0.26,  0.84 ],  # Particle
    'interp':[0.95,  0.95,  0.05 ]   # Punctuation
}
```

### Temporal coordinates

```python
TEMPORAL_COORDS = {
    'past':        [0.85, 0.90, 0.15],  # High determination
    'present':     [0.70, 0.60, 0.40],  # Medium
    'future':      [0.40, 0.30, 0.85],  # High entropy
    'conditional': [0.35, 0.25, 0.90],  # Max entropy
    'imperative':  [0.95, 0.50, 0.20],  # High intention
    'aorist':      [0.90, 0.95, 0.10],  # Closed event
    'iterative':   [0.60, 0.80, 0.35]   # Cyclical
}
```

---

## Dodatek C: Error handling

### Graceful degradation

```python
# If spaCy not available → fallback to heuristics
if not nlp:
    coords, metadata = fallback_syntax_analysis(text)

# If HerBERT not available → skip polysemy calculation
if not HERBERT_AVAILABLE:
    polysemy = _calculate_polysemy_heuristic(doc)

# If Stanza not available → skip smoking gun detection
if not STANZA_AVAILABLE:
    print("Stanza not available - skipping advanced analysis")
```

### Common errors and solutions

**1. Unicode errors on Windows:**

```python
if sys.platform == 'win32':
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
```

**2. Morfeusz2 not installed:**

```bash
pip install morfeusz2
```

**3. spaCy model not downloaded:**

```bash
python -m spacy download pl_core_news_lg
```

**4. HerBERT out of memory:**

```python
# Use smaller batch sizes
inputs = herbert_tokenizer(text, return_tensors="pt", 
                          truncation=True, max_length=128)  # Reduce from 512
```

**5. Stanza download:**

```python
import stanza
stanza.download('pl')
```

---

## Dodatek D: Glossary

| Term | Definition |
|------|------------|
| **D-S-E** | Determination-Stability-Entropy coordinates |
| **Φ⁹** | 9-dimensional extended phase space |
| **Ø** | Singularity point (1.0, 1.0, 0.0) |
| **CD** | Constitutional Definiteness |
| **CI** | Constitutional Indefiniteness |
| **SA** | Semantic Accessibility |
| **Ψᴷ** | Scientific knowledge attractor |
| **Ψʰ** | Chaos/poetry attractor |
| **Ψᴺ** | Negation attractor |
| **Ψ~** | Superposition attractor |
| **Quantum coherence** | 1.0 = fully coherent, 0.0 = decoherent |
| **Entanglement** | Quantum correlation between words |
| **Decoherence** | Loss of quantum properties over time |
| **Morpheme composition** | Geometric (tensor product) composition |
| **Rhetorical inversion** | D→1-D, S→1-S, E→1-E for irony |
| **Smoking gun** | Logical contradiction detected by Stanza |

---

## Dodatek E: References

1. **GTMØ Theory:** Grzegorz Skuza, "Geometria Topologii Metafizyki Ontologicznej"
2. **Morfeusz2:** Marcin Woliński et al., IPI PAN
3. **spaCy:** Explosion AI, Industrial-strength NLP
4. **HerBERT:** Allegro.pl, Polish BERT
5. **Stanza:** Stanford NLP Group
6. **Constitutional Metrics:** Original GTMØ extension
7. **Quantum Semantics:** Inspired by quantum cognition literature

---

## Changelog

**v2.0 (2025-11-21):**
- ✅ Complete 13 Axioms integration
- ✅ Fixed D-S correlation enforcement
- ✅ Fixed entropy calculation (polysemy, syntactic ambiguity, coherence)
- ✅ Fixed quantum coherence interpretation (1.0=coherent, 0.0=decoherent)
- ✅ Added geometric morpheme composition (tensor product)
- ✅ Added adaptive fusion weights
- ✅ Added constitutional metrics (CD-CI duality, SA v2/v3)
- ✅ Added topological attractors
- ✅ Added enhanced rhetorical analysis (irony/paradox)
- ✅ Added Stanza integration (smoking guns)
- ✅ Optimized HerBERT loading (global instance)
- ✅ Added GTMOOptimizedSaver for large documents

**v1.0 (Initial):**
- Basic morphosyntax analysis
- Quantum superposition states
- Axiom system prototype

---

## Contact & Support

**Author:** Grzegorz Skuza  
**Email:** grzegorzskuza@gmail.com
**GitHub:** GSkuza/GTMO_MORPHOSYNTAX_PL  


---

**END OF DOCUMENTATION**
