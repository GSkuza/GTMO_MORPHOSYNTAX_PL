# GTMØ Adelic Layer - Usprawnienia Teoretyczne

**Data:** 2024-11-24
**Kontekst:** Analiza wyników testów `demo_special_observers.py` i dostosowanie implementacji do pełnej teorii GTMØ

---

## 1. Wprowadzenie

Po przeprowadzeniu testów warstwy adelicznej z obserwatorami specjalnymi (halucynacje, propaganda, nonsens) zidentyfikowano trzy obszary wymagające doprecyzowania teoretycznego:

1. **Symetria metryki** - Czy Φ⁹ poprawnie modeluje asymetrię transformacji semantycznych?
2. **Kalibracja progu emergencji** - Czy ε = 0.15 ma uzasadnienie teoretyczne?
3. **Diagnostyka niepowodzeń** - Która oś (D, S, E) powoduje desynchronizację?

Dokument opisuje wprowadzone usprawnienia i ich uzasadnienie w ramach teorii GTMØ.

---

## 2. Usprawnienie 1: Pseudo-metryka Minkowskiego

### 2.1. Problem

Początkowa implementacja używała **metryki Φ⁹** (Riemannowskiej):

```
d_Φ⁹(φ₁, φ₂) = Σᵢ φⁱ · |φ₁ᵢ - φ₂ᵢ|ⁱ
```

**Własności:**
- Symetryczna: `d(A,B) = d(B,A)`
- Spełnia nierówność trójkąta
- Wszystkie osie (D, S, E) równoprawne topologicznie

**Problem teoretyczny:**
Pełna teoria GTMØ wymaga **pseudo-metryki z sygnaturą (-,+,+)**, gdzie oś S (stabilność) ma charakter temporalny (timelike), a osie D i E mają charakter przestrzenny (spacelike).

### 2.2. Rozwiązanie: Pseudo-metryka Minkowskiego

Dodano alternatywną metrykę:

```
ds² = -κ² dS² + dχ² + dΦ²

gdzie:
χ = D - E  (oś chaotyczności)
Φ = √(D² + E²)  (norma w płaszczyźnie D-E)
κ = parametr skalujący oś S (domyślnie 1.0)
```

**Implementacja:**

```python
def minkowski_distance(
    coords1: np.ndarray,
    coords2: np.ndarray,
    kappa: float = 1.0
) -> float:
    """
    Pseudo-metryka Minkowskiego z sygnaturą (-,+,+).

    UWAGA: Może zwracać wartości ujemne (interwały timelike).
    """
    D1, S1, E1 = coords1
    D2, S2, E2 = coords2

    dS = S2 - S1
    dD = D2 - D1
    dE = E2 - E1

    # Transformacja do układu χ-Φ
    chi1 = D1 - E1
    chi2 = D2 - E2
    dchi = chi2 - chi1

    phi1 = np.sqrt(D1**2 + E1**2)
    phi2 = np.sqrt(D2**2 + E2**2)
    dphi = phi2 - phi1

    # Pseudo-metryka: ds² = -κ² dS² + dχ² + dΦ²
    ds_squared = -(kappa**2) * (dS**2) + (dchi**2) + (dphi**2)

    # Zwróć pierwiastek ze znakiem
    if ds_squared >= 0:
        return np.sqrt(ds_squared)  # Spacelike
    else:
        return -np.sqrt(-ds_squared)  # Timelike (ujemny)
```

### 2.3. Interpretacja teoretyczna

**Sygnatura (-,+,+):**
- **S (timelike):** Reprezentuje "czas semantyczny" - kauzalność transformacji
  - Duża zmiana S → interwał timelike (ujemny)
  - S mierzy trwałość/stabilność znaczenia w "czasie"

- **D, E (spacelike):** Reprezentują "przestrzeń semantyczną" - konfigurację stanu
  - Duża zmiana D lub E → interwał spacelike (dodatni)
  - D, E mierzą pozycję w przestrzeni znaczeń

**Kauzalność semantyczna:**
- Interwały timelike: Zmiany stabilności dominują (ewolucja temporalna)
- Interwały spacelike: Zmiany D/E dominują (reorganizacja struktury)
- Lightcone semantyczny: Możliwe transformacje semantyczne

### 2.4. Wyniki testów

```
[Test 6] Pseudo-metryka Minkowskiego
  Timelike: d_M([0.8 0.3 0.2], [0.8 0.9 0.2]) = -0.600
  Jest ujemna (timelike): True

  Spacelike: d_M([0.3 0.8 0.2], [0.9 0.8 0.2]) = 0.822
  Jest dodatnia (spacelike): True
```

**Porównanie z Φ⁹:**
```
Energy ratio (Minkowski/Φ⁹): 1.968x
→ Minkowski penalizuje zmiany S (stabilność) silniej
→ Φ⁹ penalizuje zmiany E (entropia) silniej
```

### 2.5. Kiedy używać której metryki?

| Metryka | Kiedy używać | Zastosowania |
|---------|-------------|--------------|
| **Φ⁹** | Analiza synchroniczna (snapshot) | Detekcja halucynacji, ambiguity, konsensus obserwatorów |
| **Minkowski** | Analiza diachroniczna (ewolucja) | Tracking dryfu semantycznego, kauzalność transformacji, analiza stabilności |

---

## 3. Usprawnienie 2: Adaptacyjny próg emergencji

### 3.1. Problem

Początkowa implementacja używała **stałego progu ε = 0.15** dla wszystkich kontekstów.

**Problem:**
Czy ten sam próg powinien obowiązywać dla:
- Dokumentu prawniczego (wymagana precyzja, niska entropia)?
- Wiersza poetyckiego (tolerancja na wieloznaczność, wysoka entropia)?

### 3.2. Rozwiązanie: Adaptacyjny epsilon

Wprowadzono próg zależny od kontekstu:

```
ε_adaptive = ε₀ · (1 + γ · H_context) · f_register

gdzie:
ε₀ = bazowy próg (np. 0.15)
H_context = średnia entropia kontekstu E ∈ [0, 1]
γ = czułość na entropię (domyślnie 0.3)
f_register = modulator dla rejestru językowego
```

**Modulatory rejestru:**

```python
register_modifiers = {
    'legal': 0.7,        # Najbardziej rygorystyczny
    'formal': 0.8,       # Bardziej rygorystyczny
    'technical': 0.85,
    'journalistic': 0.9,
    'philosophical': 1.0, # Neutralny
    'sarcastic': 1.1,
    'casual': 1.2,       # Bardziej tolerancyjny
    'poetic': 1.3,       # Bardzo tolerancyjny
}
```

**Implementacja:**

```python
def compute_adaptive_epsilon(
    base_epsilon: float,
    context_entropy: float,
    register: str,
    gamma: float = 0.3
) -> float:
    """
    Oblicza adaptacyjny próg emergencji.

    Kontekst wysokoentropijny → większe ε (tolerancja)
    Kontekst niskoentropijny → mniejsze ε (rygoryzm)
    """
    f_register = register_modifiers.get(register, 1.0)
    entropy_factor = 1.0 + gamma * context_entropy

    epsilon_adaptive = base_epsilon * entropy_factor * f_register

    # Ogranicz do [0.05, 0.5]
    return np.clip(epsilon_adaptive, 0.05, 0.5)
```

### 3.3. Uzasadnienie teoretyczne

**Entropia kontekstowa jako miara niepewności:**

1. **Kontekst niskoentropijny** (E ≈ 0.1):
   - Język precyzyjny, jednoznaczny
   - Wymagany ścisły konsensus obserwatorów
   - ε zmniejszone (bardziej rygorystyczny próg)

2. **Kontekst wysokoentropijny** (E ≈ 0.7):
   - Język wieloznaczny, metaforyczny
   - Naturalna niepewność interpretacji
   - ε zwiększone (bardziej tolerancyjny próg)

**Rejestr językowy jako modulator:**
- Legal/formal: Wymagana precyzja → f < 1.0
- Casual/poetic: Tolerancja na rozbieżności → f > 1.0

### 3.4. Wyniki testów

```
[Test 7] Adaptacyjny próg ε
  Formal + niska entropia (0.1): ε = 0.124
  Casual + wysoka entropia (0.7): ε = 0.218
  Casual/formal ratio: 1.76x
```

**Praktyczne zastosowanie:**

```
📊 EMERGENCJA w różnych kontekstach:
  Kontekst                   ε_adapt    Emerged?
  ------------------------- ---------- ----------
  Dokument prawny            0.105      ✗ NIE
  Dokument formalny          0.150      ✓ TAK
  Artykuł prasowy            0.166      ✓ TAK
  Dyskusja casualowa         0.218      ✓ TAK
```

**Wniosek:**
Ta sama rozbieżność obserwatorów może:
- Blokować emergencję w kontekście prawniczym (ε=0.105)
- Pozwalać na emergencję w kontekście casualowym (ε=0.218)

→ **Adaptacyjny ε uwzględnia "naturalną niepewność" rejestru!**

### 3.5. Użycie w kodzie

```python
# Stały epsilon (domyślnie)
result = layer.analyze_with_observers(
    text="...",
    base_coords=coords,
    observers=observers,
    metric='phi9'  # Używa ε = 0.15
)

# Adaptacyjny epsilon
from gtmo_adelic_metrics import check_emergence_condition

can_emerge, phi_inf = check_emergence_condition(
    local_coords=local_coords,
    epsilon=0.15,
    metric='phi9',
    adaptive_epsilon=True,
    context_entropy=0.7,  # Z analizy GTMØ
    register='casual'
)
```

---

## 4. Usprawnienie 3: Dekompozycja D-S-E w diagnostyce

### 4.1. Problem

Początkowa diagnostyka niepowodzenia emergencji pokazywała:
- `max_distance`: Maksymalna odległość od consensus
- `exceeds_by`: O ile przekroczono ε
- `num_outliers`: Liczba obserwatorów poza ε

**Problem:**
Nie wiadomo **która oś (D, S, E) powoduje desynchronizację**.

### 4.2. Rozwiązanie: Dekompozycja wkładu osi

Dodano funkcję `compute_axis_contributions()`:

```python
def compute_axis_contributions(
    coords1: np.ndarray,
    coords2: np.ndarray,
    metric: str = 'phi9'
) -> Dict[str, float]:
    """
    Oblicza wkład każdej osi (D, S, E) do całkowitej odległości.

    Returns:
        {
            'D': wkład_D,
            'S': wkład_S,
            'E': wkład_E,
            'total': suma,
            'D_pct': procent_D,
            'S_pct': procent_S,
            'E_pct': procent_E
        }
    """
    diff = np.abs(coords1 - coords2)

    if metric == 'phi9':
        weights = np.array([PHI**1, PHI**2, PHI**3])
        powers = np.array([1, 2, 3])

        contribution_D = weights[0] * (diff[0] ** powers[0])
        contribution_S = weights[1] * (diff[1] ** powers[1])
        contribution_E = weights[2] * (diff[2] ** powers[2])

    total = contribution_D + contribution_S + contribution_E

    return {
        'D': contribution_D,
        'S': contribution_S,
        'E': contribution_E,
        'total': total,
        'D_pct': (contribution_D / total * 100) if total > 0 else 0.0,
        'S_pct': (contribution_S / total * 100) if total > 0 else 0.0,
        'E_pct': (contribution_E / total * 100) if total > 0 else 0.0
    }
```

### 4.3. Rozszerzona diagnostyka

Funkcja `diagnose_emergence_failure()` teraz zwraca:

```python
{
    # ... (poprzednie pola)
    'axis_decomposition': {
        'D': {'absolute': 0.123, 'percentage': 26.1},
        'S': {'absolute': 0.005, 'percentage': 0.2},
        'E': {'absolute': 0.345, 'percentage': 73.7}
    },
    'dominant_axis': 'E',  # Która oś dominuje
    'interpretation': 'Rozbieżność w ENTROPII...'
}
```

**Interpretacje:**

```python
def _interpret_dominant_axis(axis: str) -> str:
    interpretations = {
        'D': 'Rozbieżność w OKREŚLONOŚCI (Determination) - '
             'obserwatorzy różnią się co do pewności/definitywności',
        'S': 'Rozbieżność w STABILNOŚCI (Stability) - '
             'obserwatorzy różnią się co do trwałości semantycznej',
        'E': 'Rozbieżność w ENTROPII (Entropy) - '
             'obserwatorzy różnią się co do poziomu chaosu/wieloznaczności'
    }
    return interpretations[axis]
```

### 4.4. Wyniki testów

**Przypadek 1: Halucynacje (O_hallucination)**

```
Decomposition:
  D: 26.1%
  S:  0.2%
  E: 73.7%  ████████████████

DOMINUJĄCA OŚ: E
Rozbieżność w ENTROPII (Entropy) -
obserwatorzy różnią się co do poziomu chaosu/wieloznaczności
```

**Przypadek 2: Nonsens (O_nonsense)**

```
Decomposition:
  D: 68.3%  █████████████
  S:  2.1%
  E: 29.6%  █████

DOMINUJĄCA OŚ: D
Rozbieżność w OKREŚLONOŚCI (Determination) -
obserwatorzy różnią się co do pewności/definitywności
```

### 4.5. Praktyczne zastosowania

**Detekcja typu patologii:**

| Patologia | Dominująca oś | Interpretacja |
|-----------|---------------|---------------|
| **Halucynacje LLM** | E ↑ | Chaos semantyczny, brak spójności |
| **Nonsens** | D ↓ | Brak określoności, niezdecydowanie |
| **Propaganda** | D ↑, E ↓ | Fałszywa pewność, niska entropia |
| **Ironia/Sarkasm** | D ↔, E ↑ | Wieloznaczność, gra znaczeń |
| **Neologizmy** | D ↓, E ↑ | Nowe znaczenie, wysoka niepewność |

**Rekomendacja kontekstu:**

```python
if dominant_axis == 'E':
    # Desynchronizacja w entropii → dodaj kontekst stabilizujący
    recommended_attractor = 'Ψᴷ'  # Formalny
elif dominant_axis == 'D':
    # Desynchronizacja w określoności → dodaj kontekst precyzyjny
    recommended_attractor = 'Ψᴸ'  # Legalny
elif dominant_axis == 'S':
    # Desynchronizacja w stabilności → analiza diachroniczna
    recommended_analysis = 'temporal_drift'
```

---

## 5. Podsumowanie implementacji

### 5.1. Dodane funkcje w `gtmo_adelic_metrics.py`

1. **Pseudo-metryka Minkowskiego:**
   ```python
   def minkowski_distance(coords1, coords2, kappa=1.0) -> float
   ```

2. **Adaptacyjny epsilon:**
   ```python
   def compute_adaptive_epsilon(base_epsilon, context_entropy, register, gamma=0.3) -> float
   ```

3. **Dekompozycja osi:**
   ```python
   def compute_axis_contributions(coords1, coords2, metric='phi9') -> Dict
   def _interpret_dominant_axis(axis: str) -> str
   ```

### 5.2. Zmodyfikowane funkcje

Wszystkie funkcje z parametrem `metric` teraz wspierają `'minkowski'`:
- `compute_communication_potential()`
- `check_emergence_condition()`
- `compute_emergence_probability()`
- `compute_pairwise_energies()`
- `compute_dispersion()`
- `diagnose_emergence_failure()`

Wszystkie funkcje z parametrem `epsilon` teraz wspierają `adaptive_epsilon`:
- `check_emergence_condition()`
- `compute_emergence_probability()`

### 5.3. Nowe parametry

```python
# Metryka Minkowskiego
metric='minkowski'
metric_kappa=1.0  # Parametr κ skalujący oś S

# Adaptacyjny epsilon
adaptive_epsilon=True
context_entropy=0.7  # Z analizy GTMØ
register='casual'    # Rejestr językowy
```

---

## 6. Wyniki testów jednostkowych

```bash
$ python gtmo_adelic_metrics.py

============================================================
GTMØ Adelic Metrics - Test modułu
============================================================

[Test 1] Metryka Φ⁹
  d_Φ⁹([0.8 0.8 0.2], [0.9 0.9 0.1]) = 0.192
  Symetria: 0.192 == 0.192? True

[Test 2] Potencjał V_Comm
  V_Comm (blisko) = 0.0005
  V_Comm (daleko) = 5.3608

[Test 3] Warunek emergencji
  Coords blisko: emerged=True
  Coords daleko: emerged=False

[Test 6] Pseudo-metryka Minkowskiego
  Timelike: d_M([0.8 0.3 0.2], [0.8 0.9 0.2]) = -0.600
  Jest ujemna (timelike): True
  Spacelike: d_M([0.3 0.8 0.2], [0.9 0.8 0.2]) = 0.822
  Jest dodatnia (spacelike): True

[Test 7] Adaptacyjny próg ε
  Formal + niska entropia (0.1): ε = 0.124
  Casual + wysoka entropia (0.7): ε = 0.218
  Casual/formal ratio: 1.76x

[Test 8] Dekompozycja D-S-E w diagnostyce
  Dominant axis: E
  Decomposition:
    D: 26.1%
    S:  0.2%
    E: 73.7%

[Test 9] Wkład osi do odległości
  D contribution: 13.1%
  S contribution:  1.1%
  E contribution: 85.8%

============================================================
✓ Moduł gtmo_adelic_metrics.py załadowany pomyślnie
  ✓ Metryka Φ⁹ (Riemannowska, symetryczna)
  ✓ Pseudo-metryka Minkowskiego (sygnatura -,+,+)
  ✓ Adaptacyjny próg ε(kontekst, rejestr)
  ✓ Dekompozycja D-S-E w diagnostyce
============================================================
```

---

## 7. Wnioski i rekomendacje

### 7.1. Teoretyczne usprawnienia

✅ **Usprawnienie 1: Pseudo-metryka Minkowskiego**
- Zgodna z pełną teorią GTMØ (sygnatura -,+,+)
- Rozróżnia ewolucję temporalną (S) od reorganizacji struktury (D,E)
- Zachowuje kauzalność semantyczną

✅ **Usprawnienie 2: Adaptacyjny próg emergencji**
- Uwzględnia "naturalną niepewność" rejestru
- Teoretyczne uzasadnienie: ε ∝ entropia kontekstu
- Rygorystyczny dla kontekstów formalnych, tolerancyjny dla casualowych

✅ **Usprawnienie 3: Dekompozycja D-S-E**
- Diagnostyka typu patologii (halucynacje vs nonsens vs propaganda)
- Rekomendacja kontekstu stabilizującego
- Analiza diachroniczna (która oś zmienia się w czasie)

### 7.2. Kiedy używać której metryki?

| Scenariusz | Metryka | Epsilon | Uzasadnienie |
|------------|---------|---------|--------------|
| Detekcja halucynacji | Φ⁹ | Stały (0.15) | Snapshot, focus na E |
| Tracking dryfu semantycznego | Minkowski | Adaptacyjny | Ewolucja temporalna, kauzalność |
| Analiza dokumentów prawnych | Φ⁹ lub Minkowski | Adaptacyjny (legal) | Rygorystyczny próg (ε≈0.10) |
| Analiza poezji | Φ⁹ | Adaptacyjny (poetic) | Tolerancyjny próg (ε≈0.20) |
| Diagnoza typu patologii | Φ⁹ | Stały | Dekompozycja D-S-E |

### 7.3. Następne kroki (opcjonalne)

1. **Wizualizacja lightcone semantycznego** (Minkowski)
   - Interwały timelike vs spacelike
   - Kauzalne trajektorie transformacji

2. **Uczenie parametrów ε_adaptive**
   - Optymalizacja γ, f_register na corpus
   - Per-domain calibration

3. **Rozszerzenie diagnostyki**
   - Tensor desynchronizacji D_ij
   - Principal Component Analysis w przestrzeni D-S-E
   - Anomaly detection (outlier obserwatorzy)

4. **Integracja z GTMOMorphosyntaxEngine**
   - Auto-detect kontekstu (rejestr, entropia)
   - Auto-select metryki (synchroniczna vs diachroniczna)
   - Feedback loop (obserwatorzy uczą się z historii)

---

## 8. Rekomendacje implementacyjne (post-demo)

### 8.1. Wybór domyślnej metryki

**Rekomendacja:** Używaj **Φ⁹ dla analiz synchronicznych**, **Minkowski dla analiz diachronicznych**.

```python
# Analiza snapshot (halucynacje, ambiguity, konsensus)
metric = 'phi9'

# Tracking ewolucji semantycznej w czasie
metric = 'minkowski'
```

**Uzasadnienie:**
- Φ⁹: Penalizuje zmiany E (entropia) → lepsze dla detekcji chaosu semantycznego
- Minkowski: Penalizuje zmiany S (stabilność) → lepsze dla kauzalności temporalnej

### 8.2. Ulepszona formuła adaptacyjnego ε

**Dodano:** `compute_adaptive_epsilon_enhanced()` z √ skalowaniem i Ø₀.

```python
def compute_adaptive_epsilon_enhanced(
    base_epsilon: float,
    context_entropy: float,
    register: str,
    use_sqrt_scaling: bool = True
) -> float:
    """
    ε_adaptive = ε₀ · √(1 + H_context/Ø₀) · f_register

    gdzie Ø₀ = 1.2925 (~√(φ² + 1/φ) / √2)
    """
```

**Porównanie:**

| Formuła | Formal (E=0.1) | Casual (E=0.7) | Ratio |
|---------|----------------|----------------|-------|
| **Linear** | 0.124 | 0.218 | 1.76x |
| **Enhanced (√)** | 0.128 | 0.223 | 1.75x |

**Zalety Enhanced:**
- √ skalowanie jest bardziej konserwatywne dla wysokich entropii
- Ø₀ jako naturalny próg oparty na φ (golden ratio)
- Dostosowane f_register dla precyzyjniejszych rejestrów (technical: 0.70)

### 8.3. Naprawiony bug EOFError

**Problem:** Demo przerywane przy `input()` podczas pipingu.

**Rozwiązanie:**
```python
try:
    input("\n\nNaciśnij Enter...")
except EOFError:
    pass  # Kontynuuj bez interakcji
```

### 8.4. Optymalne parametry rejestrów (Enhanced)

```python
register_modifiers = {
    'legal': 0.75,        # Najbardziej rygorystyczny
    'technical': 0.70,    # Precyzja techniczna
    'formal': 0.82,
    'medical': 0.80,      # Precyzja medyczna
    'journalistic': 0.90,
    'philosophical': 1.0, # Neutralny
    'sarcastic': 1.10,
    'casual': 1.20,
    'poetic': 1.45        # Bardzo tolerancyjny
}
```

---

## 9. Bibliografia teoretyczna

1. **P-adyczne struktury semantyczne:**
   - Dragovich, B. et al. (2017). "p-Adic mathematical physics and B-adic analysis"
   - Khrennikov, A. (2016). "Toward p-adic model of mental space"

2. **Pseudo-metryki Minkowskiego:**
   - Misner, Thorne, Wheeler (1973). "Gravitation" - Rozdział 2 (Foundations of Special Relativity)
   - Penrose, R. (2004). "The Road to Reality" - Rozdział 17 (Spacetime)

3. **Emergencja semantyczna:**
   - Barwise, J., Perry, J. (1983). "Situations and Attitudes"
   - Gärdenfors, P. (2000). "Conceptual Spaces" - Rozdział 3 (Similarity)

4. **Adaptacyjne progi:**
   - Shannon, C. (1948). "A Mathematical Theory of Communication"
   - Jaynes, E.T. (1957). "Information Theory and Statistical Mechanics"

---

**Dokument:** THEORETICAL_IMPROVEMENTS.md
**Wersja:** 1.0
**Autor:** GTMØ Development Team
**Data:** 2024-11-24
