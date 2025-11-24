# GTMØ Adelic Layer - Warstwa Adeliczna
## Teoretyczno-implementacyjna specyfikacja p-adycznej emergencji semantycznej

**Wersja:** 1.0
**Data:** 2024-11-24
**Status:** Implementacja w toku

---

## 📋 Spis treści

1. [Wprowadzenie teoretyczne](#1-wprowadzenie-teoretyczne)
2. [Architektura systemu](#2-architektura-systemu)
3. [Komponenty warstwy adelicznej](#3-komponenty-warstwy-adelicznej)
4. [Matematyka emergencji](#4-matematyka-emergencji)
5. [API i interfejsy](#5-api-i-interfejsy)
6. [Przykłady użycia](#6-przykłady-użycia)
7. [Formaty danych](#7-formaty-danych)
8. [Implementacja](#8-implementacja)
9. [Testy i walidacja](#9-testy-i-walidacja)

---

## 1. Wprowadzenie teoretyczne

### 1.1 Motywacja

W standardowym GTMØ każde słowo/tekst ma **jedną globalną wartość** w przestrzeni D-S-E: `φ(w) ∈ [0,1]³`. To jest wystarczające dla stabilnych, jednoznacznych znaczeń, ale nie modeluje:

- **Niejednoznaczności** - "Świetny pomysł" może być pozytywem lub ironią
- **Neologizmów** - słowa, które dopiero emergują w języku
- **Rozbieżności interpretacyjnych** - różni odbiorcy rozumieją inaczej
- **Procesu stabilizacji znaczenia** - jak znaczenie "kondensuje się" z chaosu

### 1.2 Rozwiązanie: Struktura adeliczna

**Idea kluczowa:** Znaczenie nie jest platformskim bytem, ale **emerguje z konsensusu lokalnych interpretacji**.

Inspiracja matematyczna pochodzi z **teorii liczb p-adycznych** i **pierścienia Adeli**:

- Każda liczba pierwsza `p` definiuje własną topologię bliskości (liczby p-adyczne ℚₚ)
- Pierścień Adeli 𝔸 łączy wszystkie lokalne koła: `𝔸 = ℝ × ∏'_p ℚₚ`
- Globalne rozwiązanie równania emerguje z konsensusu wszystkich lokalnych rozwiązań

**Analogia semantyczna:**

- Każdy **obserwator O** jest jak "semantyczna liczba pierwsza"
- Każdy obserwator ma własne **lokalne koło semantyczne 𝕂_O**
- **Pierścień Adeli semantycznego**: `𝔸_sem(w) = ℝ_sem × ∏'_O 𝕂_O`
- **Emergencja:** Gdy `n ≥ 2` obserwatorów osiąga consensus → wartość globalna φ_∞ ∈ [0,1]³

### 1.3 Kluczowe koncepty

| Koncept | Symbol | Opis |
|---------|--------|------|
| **Obserwator** | O | "Semantyczna liczba pierwsza" - niepodzielna jednostka interpretacji |
| **Lokalne koło** | 𝕂_O | Przestrzeń interpretacji obserwatora O |
| **Wartość lokalna** | φ_O(w) | Interpretacja słowa w przez obserwatora O (może być poza [0,1]³) |
| **AlienatedNumber** | 𝔸(w) | Wartość przedemergentna (n < 2 lub brak konsensusu) |
| **Wartość globalna** | φ_∞(w) | Wartość zemergowana z konsensusu (n ≥ 2, w [0,1]³) |
| **Potencjał komunikacyjny** | V_Comm | Energia desynchronizacji: `V = (1/2)κ Σ ‖φ_i - φ_j‖²` |
| **Gradient kolapsu** | ∇V_Comm | Kierunek synchronizacji lokalnych interpretacji |

---

## 2. Architektura systemu

### 2.1 Diagram komponentów

```
┌─────────────────────────────────────────────────────────────┐
│                 GTMØ Morphosyntax Engine                     │
│                    (istniejący system)                       │
└────────────────────┬────────────────────────────────────────┘
                     │
                     │ dodaje warstwę
                     ▼
┌─────────────────────────────────────────────────────────────┐
│              ADELIC SEMANTIC LAYER                           │
├─────────────────────────────────────────────────────────────┤
│  ┌──────────────┐  ┌──────────────┐  ┌─────────────────┐   │
│  │  Observer    │  │ Alienated    │  │  AdelicRing     │   │
│  │  Management  │  │  Number      │  │  (word memory)  │   │
│  └──────────────┘  └──────────────┘  └─────────────────┘   │
│                                                              │
│  ┌────────────────────────────────────────────────────────┐ │
│  │         Adelic Metrics Engine                          │ │
│  │  • Φ⁹ distance                                         │ │
│  │  • V_Comm computation                                  │ │
│  │  • ∇V gradient                                         │ │
│  └────────────────────────────────────────────────────────┘ │
│                                                              │
│  ┌────────────────────────────────────────────────────────┐ │
│  │         Emergence Engine                               │ │
│  │  • Consensus detection (n ≥ 2)                         │ │
│  │  • Collapse to [0,1]³                                  │ │
│  │  • Context-driven disambiguation                       │ │
│  └────────────────────────────────────────────────────────┘ │
└─────────────────────────────────────────────────────────────┘
```

### 2.2 Przepływ danych

```
Input text
    │
    ▼
┌──────────────────────────┐
│ Standard GTMØ analysis   │  → Base coords [D, S, E]
└──────────┬───────────────┘
           │
           ▼
┌──────────────────────────────────────────────────┐
│ Apply Observer biases                            │
│  φ_O₁ = base + bias₁  (może wyjść poza [0,1]³)  │
│  φ_O₂ = base + bias₂                             │
│  ...                                             │
└──────────┬───────────────────────────────────────┘
           │
           ▼
┌──────────────────────────────────────────────────┐
│ Check emergence condition                        │
│  if n ≥ 2 and distance(φ_O₁, φ_O₂) < ε:        │
│     → EMERGENCE                                  │
│  else:                                           │
│     → AlienatedNumber                            │
└──────────┬───────────────────────────────────────┘
           │
           ├─── EMERGENCE ───┐
           │                 ▼
           │    ┌────────────────────────────────┐
           │    │ Compute consensus              │
           │    │ Project to [0,1]³              │
           │    │ Return φ_∞(w)                  │
           │    └────────────────────────────────┘
           │
           └─── NO EMERGENCE ───┐
                                ▼
               ┌─────────────────────────────────┐
               │ Compute ∇V_Comm                 │
               │ Return AlienatedNumbers         │
               │ Suggest context for collapse    │
               └─────────────────────────────────┘
```

---

## 3. Komponenty warstwy adelicznej

### 3.1 Observer - semantyczna liczba pierwsza

```python
@dataclass
class Observer:
    """
    Obserwator jako niepodzielna jednostka interpretacji.
    Analogia do liczby pierwszej p w teorii p-adycznej.
    """
    id: str                          # Unikalny identyfikator (np. "O_formal", "O_poetic")
    interpretation_bias: np.ndarray  # [3] - systematyczne przesunięcie w D-S-E
    coherence_threshold: float       # Próg spójności dla tego obserwatora (0.0-1.0)
    topology_metric: str             # 'euclidean' | 'phi9' | 'taxicab'
    register: str                    # 'formal' | 'casual' | 'poetic' | 'legal' | ...

    # Parametry dynamiczne
    temperature: float = 1.0         # "Temperatura" interpretacji (rozrzut)
    history: List[Dict] = field(default_factory=list)  # Historia obserwacji
```

**Przykładowe obserwatory:**

```python
# Formalny obserwator prawniczy
observer_legal = Observer(
    id="O_legal",
    interpretation_bias=np.array([0.15, 0.12, -0.08]),  # ↑D, ↑S, ↓E
    coherence_threshold=0.88,
    topology_metric='euclidean',
    register='legal'
)

# Obserwator poetycki/ironiczny
observer_poetic = Observer(
    id="O_poetic",
    interpretation_bias=np.array([-0.20, -0.15, 0.25]),  # ↓D, ↓S, ↑E
    coherence_threshold=0.55,
    topology_metric='phi9',
    register='poetic'
)

# Obserwator kolokwialny
observer_casual = Observer(
    id="O_casual",
    interpretation_bias=np.array([0.05, 0.00, 0.08]),   # lekko ↑D, ↑E
    coherence_threshold=0.70,
    topology_metric='euclidean',
    register='casual'
)
```

### 3.2 AlienatedNumber - wartość przedemergentna

```python
@dataclass
class AlienatedNumber:
    """
    Wartość semantyczna PRZED emergencją globalnej.
    Istnieje tylko w lokalnym kole 𝕂_O obserwatora.
    Może wykraczać poza standardową przestrzeń [0,1]³.
    """
    local_value: np.ndarray          # [3] - współrzędne D-S-E (MOGĄ być poza [0,1]³)
    observer_id: str                 # Do którego obserwatora należy
    n_observers: int                 # Liczba obserwatorów, którzy widzieli słowo
    synchronization_energy: float    # V_Comm = energia desynchronizacji
    can_collapse: bool               # Czy spełnia warunek emergencji
    collapse_direction: Optional[np.ndarray]  # Gradient ∇V (kierunek kolapsu)
    timestamp: float                 # Kiedy utworzono

    def is_emerged(self) -> bool:
        """Czy wartość zemergowała do globalnej?"""
        return self.n_observers >= 2 and self.can_collapse

    def is_standard(self) -> bool:
        """Czy wartość mieści się w [0,1]³?"""
        return np.all(self.local_value >= 0) and np.all(self.local_value <= 1)

    def alienation_magnitude(self) -> float:
        """Jak daleko od standardowej przestrzeni?"""
        if self.is_standard():
            return 0.0

        # Odległość od najbliższego punktu w [0,1]³
        clamped = np.clip(self.local_value, 0, 1)
        return np.linalg.norm(self.local_value - clamped)
```

**Przykłady AlienatedNumbers:**

```python
# Przykład 1: Neologizm (n=1)
alienated_neologism = AlienatedNumber(
    local_value=np.array([-0.15, 0.22, 1.34]),  # POZA [0,1]³!
    observer_id="O_medical",
    n_observers=1,
    synchronization_energy=0.0,  # brak porównania (n=1)
    can_collapse=False,
    collapse_direction=None,
    timestamp=time.time()
)
# Interpretacja:
# D = -0.15: brak ustalonej denotacji (jeszcze nie "istnieje")
# S =  0.22: bardzo niestabilne znaczenie
# E =  1.34: semantyczny chaos (ponad maksimum!)

# Przykład 2: Dwuznaczność bez konsensusu (n=2, duża energia)
alienated_irony = AlienatedNumber(
    local_value=np.array([0.25, 0.30, 0.85]),
    observer_id="O_sarcastic",
    n_observers=2,
    synchronization_energy=0.94,  # WYSOKA - brak synchronizacji
    can_collapse=False,
    collapse_direction=np.array([0.15, 0.12, -0.08]),  # w stronę Ψᴷ
    timestamp=time.time()
)
# Drugi obserwator widzi: [0.85, 0.85, 0.15] (pozytyw)
# Energia 0.94 > threshold → brak emergencji
```

### 3.3 AdelicRing - pierścień semantyczny słowa

```python
class AdelicRing:
    """
    Pierścień Adeli dla konkretnego słowa/tekstu:
    𝔸_sem(w) = ℝ_sem × ∏'_{O∈Observers} 𝕂_O

    Przechowuje:
    - Wartość globalną φ_∞(w) (jeśli zemergowała)
    - Wszystkie lokalne wartości φ_O(w)
    - Historię prób emergencji
    """

    def __init__(self, word: str, base_coords: np.ndarray):
        self.word = word
        self.base_coords = base_coords  # Z podstawowej analizy GTMØ

        self.global_value: Optional[np.ndarray] = None  # φ_∞
        self.local_values: Dict[str, AlienatedNumber] = {}  # φ_O

        self.emergence_history: List[Dict] = []
        self.creation_time = time.time()

    def add_observer_interpretation(
        self,
        observer: Observer,
        local_coords: np.ndarray
    ) -> AlienatedNumber:
        """Dodaj lokalną interpretację obserwatora."""
        alienated = AlienatedNumber(
            local_value=local_coords,
            observer_id=observer.id,
            n_observers=len(self.local_values) + 1,
            synchronization_energy=0.0,  # obliczone później
            can_collapse=False,
            collapse_direction=None,
            timestamp=time.time()
        )

        self.local_values[observer.id] = alienated
        return alienated

    def compute_synchronization_energy(self, metric: str = 'phi9') -> float:
        """
        Oblicz V_Comm = (1/2) κ_comm Σ_ij ‖φ_i - φ_j‖²
        """
        if len(self.local_values) < 2:
            return 0.0

        coords_list = [av.local_value for av in self.local_values.values()]
        n = len(coords_list)

        total_energy = 0.0
        kappa_comm = 1.0  # stała komunikacyjna

        for i in range(n):
            for j in range(i+1, n):
                if metric == 'phi9':
                    dist = phi9_distance(coords_list[i], coords_list[j])
                else:
                    dist = np.linalg.norm(coords_list[i] - coords_list[j])

                total_energy += dist ** 2

        return 0.5 * kappa_comm * total_energy / (n * (n-1) / 2)

    def attempt_emergence(
        self,
        epsilon: float = 0.15,
        context_attractor: Optional[np.ndarray] = None
    ) -> Tuple[bool, Optional[np.ndarray]]:
        """
        Próba adelicznej emergencji.

        Returns:
            (success, global_value)
        """
        if len(self.local_values) < 2:
            return False, None

        # Oblicz consensus
        coords_list = [av.local_value for av in self.local_values.values()]
        consensus = np.mean(coords_list, axis=0)

        # Sprawdź warunek adeliczny
        all_close = True
        for coords in coords_list:
            distance = phi9_distance(coords, consensus)
            if distance > epsilon:
                all_close = False
                break

        if all_close:
            # EMERGENCJA!
            self.global_value = np.clip(consensus, 0, 1)

            # Zaloguj event
            self.emergence_history.append({
                'timestamp': time.time(),
                'n_observers': len(self.local_values),
                'consensus': consensus.tolist(),
                'global_value': self.global_value.tolist(),
                'energy': self.compute_synchronization_energy()
            })

            return True, self.global_value

        return False, None

    def compute_collapse_gradients(
        self,
        context_attractor: np.ndarray,
        kappa_comm: float = 1.0,
        kappa_context: float = 0.5
    ) -> Dict[str, np.ndarray]:
        """
        Oblicz gradient kolapsu dla każdego obserwatora:
        ∇V_total = κ_comm (φ_O - φ_consensus) + κ_context (φ_O - Ψ_attractor)
        """
        if len(self.local_values) < 2:
            return {}

        coords_list = [av.local_value for av in self.local_values.values()]
        consensus = np.mean(coords_list, axis=0)

        gradients = {}

        for obs_id, alienated in self.local_values.items():
            # Gradient komunikacyjny
            grad_comm = kappa_comm * (alienated.local_value - consensus)

            # Gradient kontekstowy
            grad_context = kappa_context * (alienated.local_value - context_attractor)

            # Gradient łączny (z minusem - kierunek spadku)
            total_gradient = -(grad_comm + grad_context)

            gradients[obs_id] = total_gradient

        return gradients
```

---

## 4. Matematyka emergencji

### 4.1 Metryka Φ⁹ w przestrzeni GTMØ

**Definicja:**

```
d_Φ⁹(φ₁, φ₂) = Σᵢ φⁱ · |φ₁ᵢ - φ₂ᵢ|ⁱ
```

gdzie:
- φ = złoty podział = (1 + √5)/2 ≈ 1.618
- i ∈ {D, S, E} z wagami: φ¹, φ², φ³

**Implementacja:**

```python
def phi9_distance(coords1: np.ndarray, coords2: np.ndarray) -> float:
    """
    Metryka Φ⁹ - nieliniowa metryka w przestrzeni D-S-E.
    Wyższe składowe (E) mają wykładniczo większą wagę.
    """
    phi = (1 + np.sqrt(5)) / 2  # 1.618...

    weights = np.array([phi**1, phi**2, phi**3])  # [1.618, 2.618, 4.236]
    powers = np.array([1, 2, 3])

    diff = np.abs(coords1 - coords2)
    terms = weights * np.power(diff, powers)

    return np.sum(terms)
```

**Właściwości:**
- Nieliniowa - duże różnice w Entropy (E) dominują
- Asymetryczna względem składowych - E > S > D w wadze
- Dla małych różnic ≈ metryka euklidesowa
- Dla dużych różnic - entropia dominuje

### 4.2 Potencjał komunikacyjny V_Comm

**Definicja:**

```
V_Comm = (1/2) κ_comm · (1/n(n-1)) · Σᵢ<ⱼ d_Φ⁹(φᵢ, φⱼ)²
```

**Interpretacja:**
- Energia potrzebna do "zsynchronizowania" obserwatorów
- V_Comm → 0: obserwatorzy się zgadzają (emergencja możliwa)
- V_Comm → ∞: całkowita desynchronizacja (brak konsensusu)

**Implementacja:**

```python
def compute_communication_potential(
    local_coords: List[np.ndarray],
    kappa_comm: float = 1.0
) -> float:
    """
    Oblicz potencjał komunikacyjny V_Comm.
    """
    n = len(local_coords)
    if n < 2:
        return 0.0

    total_energy = 0.0

    for i in range(n):
        for j in range(i+1, n):
            dist = phi9_distance(local_coords[i], local_coords[j])
            total_energy += dist ** 2

    # Normalizacja przez liczbę par
    num_pairs = n * (n - 1) / 2

    return 0.5 * kappa_comm * (total_energy / num_pairs)
```

### 4.3 Gradient kolapsu ∇V

**Definicja dla obserwatora O:**

```
∇_φO V_total = κ_comm · (φ_O - φ_consensus) + κ_context · (φ_O - Ψ_attractor)
```

**Interpretacja:**
- Wskazuje kierunek, w którym φ_O musi się przesunąć dla emergencji
- Składowa `φ_O - φ_consensus`: pcha w stronę innych obserwatorów
- Składowa `φ_O - Ψ_attractor`: pcha w stronę kontekstowego attraktora

**Implementacja:**

```python
def compute_collapse_gradient(
    phi_observer: np.ndarray,
    phi_consensus: np.ndarray,
    psi_attractor: np.ndarray,
    kappa_comm: float = 1.0,
    kappa_context: float = 0.5
) -> np.ndarray:
    """
    Oblicz gradient kolapsu dla pojedynczego obserwatora.
    """
    # Gradient komunikacyjny
    grad_comm = kappa_comm * (phi_observer - phi_consensus)

    # Gradient kontekstowy
    grad_context = kappa_context * (phi_observer - psi_attractor)

    # Gradient łączny (minus - kierunek spadku energii)
    gradient = -(grad_comm + grad_context)

    return gradient
```

### 4.4 Warunek adelicznej emergencji

**Twierdzenie emergencji:**

```
∀ O ∈ {O₁, O₂, ..., Oₙ}:  d_Φ⁹(φ_O, φ_consensus) ≤ ε
⟹  ∃! φ_∞ ∈ [0,1]³
```

**W słowach:** Jeśli wszystkie lokalne interpretacje są w promieniu ε od konsensusu, to emerguje unikalna wartość globalna.

**Implementacja:**

```python
def check_emergence_condition(
    local_coords: List[np.ndarray],
    epsilon: float = 0.15
) -> Tuple[bool, Optional[np.ndarray]]:
    """
    Sprawdź warunek adelicznej emergencji.

    Returns:
        (can_emerge, consensus_value)
    """
    n = len(local_coords)

    if n < 2:
        return False, None

    # Oblicz consensus (centroid)
    consensus = np.mean(local_coords, axis=0)

    # Sprawdź warunek adeliczny
    for coords in local_coords:
        distance = phi9_distance(coords, consensus)
        if distance > epsilon:
            return False, None

    # Emergencja możliwa - projekt na [0,1]³
    global_value = np.clip(consensus, 0, 1)

    return True, global_value
```

**Parametry:**
- `ε = 0.15` (domyślnie): próg synchronizacji
  - ε < 0.1: bardzo restrykcyjny (tylko bliskie interpretacje)
  - ε = 0.15: standardowy (umiarkowana różnorodność)
  - ε > 0.3: liberalny (duża różnorodność interpretacji)

---

## 5. API i interfejsy

### 5.1 Główne API warstwy adelicznej

```python
class AdelicSemanticLayer:
    """
    Główna fasada dla warstwy adelicznej.
    Integruje się z GTMOMorphosyntaxEngine.
    """

    def __init__(self,
                 default_observers: Optional[List[Observer]] = None,
                 epsilon: float = 0.15,
                 kappa_comm: float = 1.0,
                 kappa_context: float = 0.5):
        """
        Inicjalizacja warstwy adelicznej.

        Args:
            default_observers: Domyślni obserwatorzy (jeśli None, tworzy standardowy zestaw)
            epsilon: Próg emergencji adelicznej
            kappa_comm: Stała komunikacyjna
            kappa_context: Stała kontekstowa
        """
        self.observers = default_observers or self._create_default_observers()
        self.epsilon = epsilon
        self.kappa_comm = kappa_comm
        self.kappa_context = kappa_context

        # Pamięć pierścieni adelicznych (cache)
        self.adelic_rings: Dict[str, AdelicRing] = {}

    def analyze_with_observers(
        self,
        text: str,
        base_coords: np.ndarray,
        observers: Optional[List[Observer]] = None,
        context_attractor: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Analiza tekstu z warstwą adeliczną.

        Args:
            text: Tekst do analizy
            base_coords: Bazowe współrzędne z GTMØ [D, S, E]
            observers: Lista obserwatorów (jeśli None, użyj domyślnych)
            context_attractor: Nazwa attraktora kontekstowego ('Ψᴷ', 'Ψˢ', etc.)

        Returns:
            Dict z wynikami analizy adelicznej
        """
        pass

    def create_observer(
        self,
        observer_id: str,
        register: str,
        bias_d: float = 0.0,
        bias_s: float = 0.0,
        bias_e: float = 0.0,
        coherence_threshold: float = 0.75
    ) -> Observer:
        """Utwórz nowego obserwatora z podanymi parametrami."""
        pass

    def get_or_create_ring(self, word: str, base_coords: np.ndarray) -> AdelicRing:
        """Pobierz lub utwórz pierścień adeliczny dla słowa."""
        pass

    def compute_dialogue_energy(
        self,
        utterances: List[str],
        speaker_a_observer: Observer,
        speaker_b_observer: Observer
    ) -> float:
        """
        Oblicz całkowitą energię komunikacyjną dialogu.
        Wysoka energia = trudna komunikacja.
        """
        pass
```

### 5.2 Integracja z GTMOMorphosyntaxEngine

```python
# Modyfikacja istniejącej klasy
class GTMOMorphosyntaxEngine:

    def __init__(self):
        # ... existing code ...

        # Dodaj warstwę adeliczną
        self.adelic_layer = AdelicSemanticLayer(
            epsilon=0.15,
            kappa_comm=1.0,
            kappa_context=0.5
        )

    def analyze_adelic(
        self,
        text: str,
        observers: Optional[List[Observer]] = None,
        context_attractor: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Analiza z warstwą adeliczną.

        Wykonuje:
        1. Standardową analizę GTMØ → base_coords
        2. Aplikację biasów obserwatorów → local_coords
        3. Próbę emergencji adelicznej
        4. Obliczenie V_Comm i gradientów

        Returns:
            Dict z pełnymi wynikami (GTMØ + adeliczne)
        """
        # 1. Standardowa analiza
        base_result = self.analyze(text)
        base_coords = np.array([
            base_result['coordinates']['determination'],
            base_result['coordinates']['stability'],
            base_result['coordinates']['entropy']
        ])

        # 2. Analiza adeliczna
        adelic_result = self.adelic_layer.analyze_with_observers(
            text=text,
            base_coords=base_coords,
            observers=observers,
            context_attractor=context_attractor
        )

        # 3. Połącz wyniki
        base_result['adelic'] = adelic_result

        return base_result
```

---

## 6. Przykłady użycia

### Przykład 1: Podstawowa analiza z emergencją

```python
from gtmo_morphosyntax import GTMOMorphosyntaxEngine
from gtmo_adelic_layer import Observer

# Inicjalizacja
engine = GTMOMorphosyntaxEngine()

# Obserwatorzy
obs_formal = Observer(
    id="O_formal",
    interpretation_bias=np.array([0.10, 0.08, -0.06]),
    coherence_threshold=0.85,
    topology_metric='euclidean',
    register='formal'
)

obs_casual = Observer(
    id="O_casual",
    interpretation_bias=np.array([0.05, 0.00, 0.05]),
    coherence_threshold=0.70,
    topology_metric='euclidean',
    register='casual'
)

# Analiza
result = engine.analyze_adelic(
    text="Ustawa wchodzi w życie z dniem ogłoszenia",
    observers=[obs_formal, obs_casual]
)

# Wynik:
print(result['adelic'])
# {
#     'emerged': True,
#     'global_value': [0.87, 0.88, 0.14],
#     'local_values': {
#         'O_formal': [0.88, 0.89, 0.13],
#         'O_casual': [0.86, 0.87, 0.15]
#     },
#     'synchronization_energy': 0.03,  # Niska - consensus!
#     'n_observers': 2
# }
```

### Przykład 2: Brak emergencji - dwuznaczność

```python
# Tekst dwuznaczny (ironia vs pozytyw)
result = engine.analyze_adelic(
    text="Świetny pomysł, naprawdę genialny",
    observers=[obs_formal, obs_sarcastic]
)

print(result['adelic'])
# {
#     'emerged': False,
#     'local_values': {
#         'O_formal': [0.82, 0.85, 0.18],      # pozytyw
#         'O_sarcastic': [0.28, 0.32, 0.88]    # ironia
#     },
#     'synchronization_energy': 0.94,  # Wysoka - desynchronizacja!
#     'collapse_gradients': {
#         'O_formal': [-0.12, -0.08, 0.15],
#         'O_sarcastic': [0.18, 0.14, -0.22]
#     },
#     'recommended_context': 'Ψᴷ',  # Kontekst formalny rozwiąże dwuznaczność
#     'n_observers': 2
# }
```

### Przykład 3: Neologizm (AlienatedNumber)

```python
# Tylko jeden obserwator widział słowo
result = engine.analyze_adelic(
    text="covidoza",
    observers=[obs_medical]
)

print(result['adelic'])
# {
#     'emerged': False,
#     'local_values': {
#         'O_medical': [-0.15, 0.22, 1.34]  # POZA [0,1]³!
#     },
#     'synchronization_energy': 0.0,  # Brak porównania (n=1)
#     'alienation_magnitude': 0.38,   # Odległość od [0,1]³
#     'status': 'AlienatedNumber',
#     'n_observers': 1,
#     'reason': 'Insufficient observers (need n >= 2 for emergence)'
# }
```

### Przykład 4: Energia komunikacyjna dialogu

```python
dialogue = [
    "Proszę o przedłożenie dokumentacji.",
    "Dokumentację przedkładam niezwłocznie.",
    "Dziękuję za terminową odpowiedź."
]

total_energy = engine.adelic_layer.compute_dialogue_energy(
    utterances=dialogue,
    speaker_a_observer=obs_formal,
    speaker_b_observer=obs_legal
)

print(f"Dialog energy: {total_energy:.3f}")
# Dialog energy: 0.087  (niska - łatwa komunikacja)
```

---

## 7. Formaty danych

### 7.1 Format wyniku analizy adelicznej

```json
{
  "text": "Świetny pomysł",
  "base_coordinates": {
    "determination": 0.75,
    "stability": 0.80,
    "entropy": 0.25
  },
  "adelic": {
    "emerged": false,
    "n_observers": 2,
    "local_values": {
      "O_formal": {
        "coords": [0.85, 0.88, 0.19],
        "is_standard": true,
        "alienation_magnitude": 0.0
      },
      "O_sarcastic": {
        "coords": [0.25, 0.30, 0.85],
        "is_standard": true,
        "alienation_magnitude": 0.0
      }
    },
    "synchronization_energy": 0.94,
    "collapse_gradients": {
      "O_formal": [-0.12, -0.08, 0.15],
      "O_sarcastic": [0.18, 0.14, -0.22]
    },
    "recommended_context": {
      "attractor": "Ψᴷ",
      "expected_energy_reduction": 0.68
    },
    "status": "desynchronized",
    "timestamp": 1700000000.123
  }
}
```

### 7.2 Format AlienatedNumber

```json
{
  "word": "covidoza",
  "alienated_number": {
    "local_value": [-0.15, 0.22, 1.34],
    "observer_id": "O_medical",
    "n_observers": 1,
    "synchronization_energy": 0.0,
    "can_collapse": false,
    "is_standard": false,
    "alienation_magnitude": 0.38,
    "interpretation": {
      "D": "negative - no established denotation",
      "S": "very low - unstable meaning",
      "E": "exceeds 1.0 - semantic chaos"
    },
    "timestamp": 1700000000.456
  }
}
```

---

## 8. Implementacja

### 8.1 Struktura plików

```
d:\GTMO_MORPHOSYNTAX\
├── gtmo_adelic_layer.py          # Główny moduł warstwy
│   ├── Observer
│   ├── AlienatedNumber
│   ├── AdelicRing
│   └── AdelicSemanticLayer
│
├── gtmo_adelic_metrics.py        # Metryki adeliczne
│   ├── phi9_distance()
│   ├── compute_communication_potential()
│   ├── compute_collapse_gradient()
│   └── check_emergence_condition()
│
├── gtmo_morphosyntax.py          # Modyfikacja istniejącego
│   └── + analyze_adelic()
│
└── gtmo_documentation/
    └── ADELIC_LAYER.md           # Ten dokument
```

### 8.2 Zależności

```python
# Wymagane importy
import numpy as np
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass, field
import time
from enum import Enum

# Istniejące moduły GTMØ
from gtmo_morphosyntax import GTMOMorphosyntaxEngine
from gtmo_topological_attractors import TopologicalAttractorAnalyzer
```

### 8.3 Parametry konfiguracyjne

```python
# Domyślne parametry warstwy adelicznej
ADELIC_CONFIG = {
    'epsilon': 0.15,              # Próg emergencji
    'kappa_comm': 1.0,            # Stała komunikacyjna
    'kappa_context': 0.5,         # Stała kontekstowa

    # Obserwatorzy domyślni
    'default_observers': [
        {'id': 'O_formal', 'bias': [0.10, 0.08, -0.06], 'threshold': 0.85},
        {'id': 'O_casual', 'bias': [0.05, 0.00, 0.05], 'threshold': 0.70},
        {'id': 'O_poetic', 'bias': [-0.20, -0.15, 0.25], 'threshold': 0.55}
    ],

    # Cache
    'ring_cache_size': 1000,      # Maksymalna liczba pierścieni w cache
    'ring_ttl': 3600,             # Time-to-live pierścienia (sekundy)
}
```

---

## 9. Testy i walidacja

### 9.1 Unit testy

```python
# test_adelic_layer.py

def test_observer_creation():
    """Test tworzenia obserwatorów."""
    obs = Observer(
        id="test_obs",
        interpretation_bias=np.array([0.1, 0.0, -0.1]),
        coherence_threshold=0.75,
        topology_metric='euclidean',
        register='test'
    )
    assert obs.id == "test_obs"
    assert obs.coherence_threshold == 0.75

def test_phi9_distance():
    """Test metryki Φ⁹."""
    phi = (1 + np.sqrt(5)) / 2

    coords1 = np.array([0.5, 0.5, 0.5])
    coords2 = np.array([0.6, 0.6, 0.6])

    dist = phi9_distance(coords1, coords2)

    # Sprawdź że > 0 i skończona
    assert dist > 0
    assert np.isfinite(dist)

    # Sprawdź symetrię
    assert np.isclose(dist, phi9_distance(coords2, coords1))

def test_emergence_condition():
    """Test warunku emergencji."""
    # Przypadek 1: Blisko siebie → emergencja
    local_coords = [
        np.array([0.85, 0.87, 0.15]),
        np.array([0.87, 0.88, 0.14])
    ]
    can_emerge, consensus = check_emergence_condition(local_coords, epsilon=0.15)
    assert can_emerge == True
    assert consensus is not None

    # Przypadek 2: Daleko od siebie → brak emergencji
    local_coords = [
        np.array([0.85, 0.87, 0.15]),
        np.array([0.25, 0.30, 0.85])
    ]
    can_emerge, consensus = check_emergence_condition(local_coords, epsilon=0.15)
    assert can_emerge == False

def test_alienated_number():
    """Test AlienatedNumber."""
    # Wartość poza [0,1]³
    alienated = AlienatedNumber(
        local_value=np.array([-0.15, 0.22, 1.34]),
        observer_id="test_obs",
        n_observers=1,
        synchronization_energy=0.0,
        can_collapse=False,
        collapse_direction=None,
        timestamp=time.time()
    )

    assert alienated.is_emerged() == False
    assert alienated.is_standard() == False
    assert alienated.alienation_magnitude() > 0.0
```

### 9.2 Integration testy

```python
def test_full_adelic_analysis():
    """Test pełnej analizy adelicznej."""
    engine = GTMOMorphosyntaxEngine()

    obs1 = Observer(
        id="O_test1",
        interpretation_bias=np.array([0.05, 0.05, -0.05]),
        coherence_threshold=0.80,
        topology_metric='euclidean',
        register='test'
    )

    obs2 = Observer(
        id="O_test2",
        interpretation_bias=np.array([0.07, 0.06, -0.04]),
        coherence_threshold=0.80,
        topology_metric='euclidean',
        register='test'
    )

    result = engine.analyze_adelic(
        text="Test sentence",
        observers=[obs1, obs2]
    )

    assert 'adelic' in result
    assert 'emerged' in result['adelic']
    assert 'local_values' in result['adelic']
    assert len(result['adelic']['local_values']) == 2
```

### 9.3 Testy walidacyjne

```python
def test_energy_monotonicity():
    """
    Test: Im większa różnica między obserwatorami,
    tym większa energia synchronizacyjna.
    """
    coords_a = np.array([0.5, 0.5, 0.5])

    energies = []
    for delta in [0.1, 0.2, 0.3, 0.4]:
        coords_b = np.array([0.5 + delta, 0.5 + delta, 0.5 - delta])
        energy = compute_communication_potential([coords_a, coords_b])
        energies.append(energy)

    # Sprawdź monotoniczność
    for i in range(len(energies) - 1):
        assert energies[i] < energies[i+1]

def test_gradient_direction():
    """
    Test: Gradient wskazuje w stronę konsensusu.
    """
    phi_obs = np.array([0.9, 0.9, 0.1])
    phi_consensus = np.array([0.7, 0.7, 0.3])
    psi_attractor = np.array([0.85, 0.85, 0.15])

    gradient = compute_collapse_gradient(
        phi_obs, phi_consensus, psi_attractor,
        kappa_comm=1.0, kappa_context=0.0  # Tylko komunikacyjny
    )

    # Gradient powinien wskazywać "w dół" (w stronę consensus)
    assert gradient[0] < 0  # D
    assert gradient[1] < 0  # S
    assert gradient[2] > 0  # E
```

---

## 10. Roadmap implementacji

### Faza 1: Podstawy (1-2 dni)
- [x] Dokumentacja (ten plik)
- [ ] Implementacja `Observer`
- [ ] Implementacja `AlienatedNumber`
- [ ] Implementacja metryk: `phi9_distance`, `compute_communication_potential`

### Faza 2: Emergencja (2-3 dni)
- [ ] Implementacja `AdelicRing`
- [ ] Implementacja `check_emergence_condition`
- [ ] Implementacja `compute_collapse_gradient`
- [ ] Unit testy dla emergencji

### Faza 3: Integracja (2-3 dni)
- [ ] Implementacja `AdelicSemanticLayer`
- [ ] Modyfikacja `GTMOMorphosyntaxEngine.analyze_adelic()`
- [ ] Integration testy
- [ ] Dokumentacja API

### Faza 4: Zaawansowane (3-4 dni)
- [ ] Cache pierścieni adelicznych
- [ ] Historia emergencji i temporal tracking
- [ ] Analiza dialogów (`compute_dialogue_energy`)
- [ ] Detekcja code-switching
- [ ] Performance optimization

### Faza 5: Walidacja (2-3 dni)
- [ ] Comprehensive test suite
- [ ] Benchmarki wydajnościowe
- [ ] Przykłady użycia
- [ ] Dokumentacja końcowa

---

## 11. Bibliografia i odnośniki

### Matematyka
- Gouvêa, F. Q. (1997). *p-adic Numbers: An Introduction*. Springer.
- Ramakrishnan, D., & Valenza, R. J. (1999). *Fourier Analysis on Number Fields*. Springer.

### GTMØ
- [GTMØ Core Documentation](./GTMO_CORE.md)
- [Topological Attractors](./TOPOLOGICAL_ATTRACTORS.md)
- [Axiom System](./AXIOM_SYSTEM.md)

### Semantic Theory
- Fauconnier, G., & Turner, M. (2002). *The Way We Think: Conceptual Blending*.
- Gärdenfors, P. (2000). *Conceptual Spaces*.

---

**Status:** 🚧 Dokument żywy - aktualizowany podczas implementacji
**Kontakt:** GTMØ Development Team
**Licencja:** [Określ licencję projektu]
