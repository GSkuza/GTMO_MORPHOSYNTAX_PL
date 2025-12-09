"""
GeometricDSEComputer - Geometryczne obliczanie współrzędnych D-S-E dla morfemów
================================================================================

Implementacja zgodna z teorią GTMØ (Geometry Topology Mathematics Øndefiniteness)
Autor: Grzegorz Skuza / Ønderstand.ai

Ten moduł realizuje geometryczne obliczanie współrzędnych w przestrzeni fazowej F³
oraz pełnej przestrzeni Φ⁹ = F³ × K⁶ dla morfemów języka naturalnego.

KLUCZOWA FILOZOFIA PROJEKTOWA:
-----------------------------
Algorytm świadomie ODRZUCA klasyczne podejścia statystyczne:
- Entropia Shannona H = -Σp·log(p) dla rozkładów symboli → NIE
- Entropia Von Neumanna S = -Tr(ρ log ρ) dla macierzy gęstości → NIE

Zamiast tego UŻYWA metod geometrycznych naturalnych dla przestrzeni embeddingów:
- Kohezja klastra (cluster cohesion)
- Podobieństwo kosinusowe między próbkami
- Wariancja kontekstowa (nie rozkład symboli)
- Iteracje Julia i dynamika fraktalna

PRZESTRZEŃ GTMØ:
---------------
Φ⁹ = F³ × K⁶

F³ = [D, S, E] ⊂ [0,1]³ - Przestrzeń Fazowa
  D - Determinacja: kohezja semantyczna (1=jednoznaczny, 0=wieloznaczny)
  S - Stabilność: odporność na perturbacje kontekstowe (1=stabilny, 0=zmienny)
  E - Entropia/Dynamika: tempo zmian i potencjał generatywny (1=dynamiczny, 0=statyczny)

K⁶ = [θ_D, φ_D, θ_S, φ_S, θ_E, φ_E] - Wymiary Kierunkowe z dynamiki Julia
  θ - kąt polarny kierunku ucieczki w iteracji Julia
  φ - kąt azymutalny kierunku ucieczki
  ρ - tempo radialne dywergencji (escape rate)

LITERATURA:
----------
- GTMØ Theory Analysis (Skuza, 2024)
- 191 przestrzeni GTMO.pdf
- Male_jest_piekne_kompresja_LLM_GTMO_v2.md
- GTMO_Reconstruction_Theorem_FINAL.md
"""

import numpy as np
from typing import Dict, List, Tuple, Optional, Union, Literal
from dataclasses import dataclass, field
from enum import Enum
from scipy.stats import spearmanr
from scipy.spatial.distance import pdist, squareform
import warnings


# =============================================================================
# TYPY I STAŁE
# =============================================================================

class DeterminationMethod(Enum):
    """Metody obliczania Determinacji (D)"""
    CLUSTER_COHESION = "cluster_cohesion"    # Domyślna - kohezja geometryczna
    COSINE_SIMILARITY = "cosine_similarity"   # Średnie podobieństwo kosinusowe
    DIAMETER = "diameter"                     # Bazująca na średnicy klastra
    AMBIGUITY_CORRECTED = "ambiguity_corrected"  # Z korekcją ambiguity


class AmbiguitySource(Enum):
    """Źródło informacji o ambiguity morfologicznej"""
    GEOMETRIC = "geometric"      # Estymacja z rozproszenia embeddingów (domyślna)
    EXTERNAL = "external"        # Zewnętrzne dane (np. z Morfeusza2)
    SUBWORD = "subword"          # Z wariancji subword tokenów
    HYBRID = "hybrid"            # Kombinacja geometric + external


class StabilityMethod(Enum):
    """Metody obliczania Stabilności (S)"""
    CONTEXTUAL_VARIANCE = "contextual_variance"  # Wariancja kontekstowa (oryginalna)
    SEMANTIC_SPREAD = "semantic_spread"          # NOWA DOMYŚLNA - rozrzut semantyczny
    CENTROID_STABILITY = "centroid_stability"    # Bootstrap centroidu
    TEMPORAL_AUTOCORR = "temporal_autocorrelation"  # Dla danych diachronicznych


class EntropyMethod(Enum):
    """Metody obliczania Entropii/Dynamiki (E)"""
    COMBINED = "combined"              # Domyślna - kombinacja TV i Julia
    TOTAL_VARIATION = "total_variation"  # Suma różnic kolejnych embeddingów
    JULIA_ESCAPE = "julia_escape"        # Escape rate z iteracji Julia


# Stałe GTMØ zgodne z dokumentacją
JULIA_C_BASE = complex(-0.8, 0.156)  # Parametr c dla zbioru Julia
ESCAPE_RADIUS = 4.0                   # Promień ucieczki |z| > R
MAX_JULIA_ITERATIONS = 1000           # Maksymalna liczba iteracji Julia
HAUSDORFF_DIMENSION = 1.585           # Wymiar fraktalny granicy basenów (używany w skalowaniu E)
INDEFINITENESS_CONSTANT = 1.2925      # ∅₀ - stała niedefinitywności (używana w korekcji D)

# Progi dla klasyfikacji stanów
EMERGENCE_THRESHOLD = 0.7             # Próg emergencji semantycznej
ALIENATION_THRESHOLD = 0.99           # Próg alienacji (blisko singularności)


@dataclass
class DSEResult:
    """Wynik obliczenia współrzędnych D-S-E z metadanymi"""
    D: float                          # Determinacja ∈ [0,1]
    S: float                          # Stabilność ∈ [0,1]
    E: float                          # Entropia/Dynamika ∈ [0,1]
    confidence_D: float = 1.0         # Pewność oszacowania D
    confidence_S: float = 1.0         # Pewność oszacowania S
    confidence_E: float = 1.0         # Pewność oszacowania E
    method_D: str = ""                # Użyta metoda dla D
    method_S: str = ""                # Użyta metoda dla S
    method_E: str = ""                # Użyta metoda dla E
    n_samples: int = 0                # Liczba użytych próbek
    ambiguity: Optional[float] = None # Zewnętrzna wartość ambiguity (z Morfeusza)

    def __post_init__(self):
        """Walidacja zakresów po inicjalizacji"""
        self.D = float(np.clip(self.D, 0.0, 1.0))
        self.S = float(np.clip(self.S, 0.0, 1.0))
        self.E = float(np.clip(self.E, 0.0, 1.0))
        self.confidence_D = float(np.clip(self.confidence_D, 0.0, 1.0))
        self.confidence_S = float(np.clip(self.confidence_S, 0.0, 1.0))
        self.confidence_E = float(np.clip(self.confidence_E, 0.0, 1.0))

    def to_array(self) -> np.ndarray:
        """Konwersja do wektora numpy (float64)"""
        return np.array([self.D, self.S, self.E], dtype=np.float64)

    def __repr__(self) -> str:
        return (f"DSEResult(D={self.D:.4f}±{1-self.confidence_D:.3f}, "
                f"S={self.S:.4f}±{1-self.confidence_S:.3f}, "
                f"E={self.E:.4f}±{1-self.confidence_E:.3f})")


@dataclass
class K6Result:
    """
    Wynik obliczenia wymiarów kierunkowych K⁶ z dynamiki Julia.

    K⁶ = [θ_D, φ_D, θ_S, φ_S, θ_E, φ_E] - 6 wymiarów kierunkowych

    Dla każdej osi (D, S, E) obliczane są:
    - θ (theta): kąt polarny kierunku ucieczki ∈ [0, π]
    - φ (phi): kąt azymutalny kierunku ucieczki ∈ [0, 2π]
    - ρ (rho): tempo ucieczki (escape rate) - NIE jest częścią K⁶, ale przechowywane dla analizy
    """
    # Kierunki dla osi D (z płaszczyzny S-E)
    theta_D: float = 0.0   # Kąt polarny ∈ [0, π]
    phi_D: float = 0.0     # Kąt azymutalny ∈ [0, 2π]

    # Kierunki dla osi S (z płaszczyzny D-E)
    theta_S: float = 0.0
    phi_S: float = 0.0

    # Kierunki dla osi E (z płaszczyzny D-S)
    theta_E: float = 0.0
    phi_E: float = 0.0

    # Escape rates (pomocnicze, NIE część K⁶)
    rho_D: float = 0.0     # Tempo ucieczki dla osi D
    rho_S: float = 0.0     # Tempo ucieczki dla osi S
    rho_E: float = 0.0     # Tempo ucieczki dla osi E

    # Flagi ucieczki
    escaped_D: bool = False
    escaped_S: bool = False
    escaped_E: bool = False

    def to_array(self) -> np.ndarray:
        """Konwersja do wektora K⁶ (6D - zgodnie z Φ⁹ = F³ × K⁶)"""
        return np.array([
            self.theta_D, self.phi_D,
            self.theta_S, self.phi_S,
            self.theta_E, self.phi_E
        ], dtype=np.float64)

    def to_array_with_rho(self) -> np.ndarray:
        """Konwersja do wektora 9D z escape rates (dla rozszerzonej analizy)"""
        return np.array([
            self.theta_D, self.phi_D, self.rho_D,
            self.theta_S, self.phi_S, self.rho_S,
            self.theta_E, self.phi_E, self.rho_E
        ], dtype=np.float64)

    def mean_escape_rate(self) -> float:
        """Średni escape rate (skalowany przez HAUSDORFF_DIMENSION)"""
        rates = [self.rho_D, self.rho_S, self.rho_E]
        escaped = [self.escaped_D, self.escaped_S, self.escaped_E]
        valid_rates = [r for r, e in zip(rates, escaped) if e]
        if not valid_rates:
            return 0.0
        return float(np.mean(valid_rates) * HAUSDORFF_DIMENSION)


@dataclass
class Phi9Result:
    """
    Pełny wynik w przestrzeni Φ⁹ = F³ × K⁶

    Φ⁹ = [D, S, E, θ_D, φ_D, θ_S, φ_S, θ_E, φ_E]
    - F³ = [D, S, E] - przestrzeń fazowa (3D)
    - K⁶ = [θ_D, φ_D, θ_S, φ_S, θ_E, φ_E] - wymiary kierunkowe (6D)
    """
    dse: DSEResult
    k6: K6Result
    morpheme: str = ""

    def to_array(self) -> np.ndarray:
        """Pełny wektor Φ⁹ = [D, S, E, θ_D, φ_D, θ_S, φ_S, θ_E, φ_E]"""
        return np.concatenate([self.dse.to_array(), self.k6.to_array()])
    
    def __repr__(self) -> str:
        return (f"Φ⁹({self.morpheme}): "
                f"F³=[{self.dse.D:.3f}, {self.dse.S:.3f}, {self.dse.E:.3f}], "
                f"K⁶=[θD={self.k6.theta_D:.3f}, φD={self.k6.phi_D:.3f}, ...]")


# =============================================================================
# GŁÓWNA KLASA: GeometricDSEComputer
# =============================================================================

class GeometricDSEComputer:
    """
    Geometryczny kalkulator współrzędnych D-S-E dla morfemów
    
    Ta klasa implementuje geometryczne metody obliczania pozycji morfemu
    w 9-wymiarowej przestrzeni semantycznej GTMØ (Φ⁹ = F³ × K⁶).
    
    Kluczowe cechy:
    ---------------
    1. Ortogonalność D-S-E: Współrzędne obliczane niezależnie metodami geometrycznymi
    2. Integracja z dynamiką fraktalną: Julia escape rate dla wymiarów kierunkowych
    3. Wielometodowość z wagami pewności: Każda współrzędna ma confidence score
    4. Wsparcie dla danych diachronicznych: Embeddingi z różnych okresów czasowych
    
    Przykład użycia:
    ---------------
    ```python
    # Inicjalizacja z embeddingami kontekstowymi morfemu
    computer = GeometricDSEComputer(
        embeddings=morpheme_embeddings,  # [N, 768] - N próbek kontekstowych
        lambda_stability=1.0
    )
    
    # Obliczenie pełnej przestrzeni Φ⁹
    result = computer.compute_Phi9()
    print(f"Morfem w Φ⁹: {result.to_array()}")
    
    # Lub osobno F³
    dse = computer.compute_DSE()
    print(f"D={dse.D:.3f}, S={dse.S:.3f}, E={dse.E:.3f}")
    ```
    
    Parametry:
    ----------
    embeddings : np.ndarray
        Macierz embeddingów kontekstowych morfemu [N_samples, embedding_dim]
        Typowo embedding_dim = 768 dla modeli BERT/HerBERT
        
    lambda_stability : float
        Parametr λ w formule S = exp(-λ·σ²), kontroluje czułość stabilności
        
    julia_c : complex
        Parametr c dla iteracji Julia z_{n+1} = z_n² + c
        Domyślnie c = -0.8 + 0.156i (interesujący region zbioru Mandelbrota)
        
    temporal_epochs : Optional[List[int]]
        Indeksy epok czasowych dla każdego embeddingu (dla danych diachronicznych)
        Jeśli None, zakłada się synchroniczne embeddingi
    """
    
    def __init__(
        self,
        embeddings: np.ndarray,
        lambda_stability: float = 1.0,
        julia_c: complex = JULIA_C_BASE,
        temporal_epochs: Optional[List[int]] = None,
        normalize_embeddings: bool = True
    ):
        # Walidacja wejścia
        if embeddings.ndim == 1:
            embeddings = embeddings.reshape(1, -1)
        
        if embeddings.shape[0] < 1:
            raise ValueError("Wymagany co najmniej 1 embedding")
        
        self.raw_embeddings = embeddings.copy()
        self.embedding_dim = embeddings.shape[1]
        self.n_samples = embeddings.shape[0]
        
        # Normalizacja do sfery jednostkowej (opcjonalna, ale zalecana)
        if normalize_embeddings:
            norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
            norms = np.where(norms > 0, norms, 1.0)  # Unikamy dzielenia przez 0
            self.embeddings = embeddings / norms
        else:
            self.embeddings = embeddings.copy()
        
        # Parametry
        self.lambda_stability = lambda_stability
        self.julia_c = julia_c
        self.temporal_epochs = temporal_epochs
        
        # Cache dla obliczonych wartości
        self._cache: Dict[str, any] = {}
        
        # Centroid klastra
        self._centroid = np.mean(self.embeddings, axis=0)
    
    # =========================================================================
    # KROK 1: OBLICZANIE DETERMINACJI (D)
    # =========================================================================
    
    def compute_D(
        self,
        method: DeterminationMethod = DeterminationMethod.CLUSTER_COHESION
    ) -> Tuple[float, float]:
        """
        Oblicz Determinację (D) - kohezję geometryczną klastra embeddingów
        
        Semantyka GTMØ:
        - D → 1: morfem jednoznaczny ("woda", "krzesło") - embeddingi skupione
        - D → 0: morfem wieloznaczny ("rzecz", "sprawa") - embeddingi rozproszone
        
        Parameters:
        -----------
        method : DeterminationMethod
            Metoda obliczania:
            - CLUSTER_COHESION: 1 - (średnia_odległość / max_odległość)
            - COSINE_SIMILARITY: (mean_cos_sim + 1) / 2
            - DIAMETER: 1 - (diameter / 2)
        
        Returns:
        --------
        Tuple[float, float]: (D, confidence)
        """
        cache_key = f"D_{method.value}"
        if cache_key in self._cache:
            return self._cache[cache_key]
        
        if self.n_samples < 2:
            # Pojedynczy embedding - maksymalna determinacja (brak rozproszenia)
            result = (1.0, 0.5)  # Niska pewność przy jednej próbce
            self._cache[cache_key] = result
            return result
        
        if method == DeterminationMethod.CLUSTER_COHESION:
            D, conf = self._compute_D_cluster_cohesion()
        elif method == DeterminationMethod.COSINE_SIMILARITY:
            D, conf = self._compute_D_cosine_similarity()
        elif method == DeterminationMethod.DIAMETER:
            D, conf = self._compute_D_diameter()
        elif method == DeterminationMethod.AMBIGUITY_CORRECTED:
            D, conf = self._compute_D_ambiguity_corrected()
        else:
            raise ValueError(f"Nieznana metoda: {method}")
        
        # Clip do [0, 1]
        D = np.clip(D, 0.0, 1.0)
        
        result = (float(D), float(conf))
        self._cache[cache_key] = result
        return result
    
    def _compute_D_cluster_cohesion(self) -> Tuple[float, float]:
        """
        Metoda 1a: Cluster Cohesion (domyślna)
        
        D = 1 - (średnia_odległość_wewnątrz / max_odległość)
        
        Dla embeddingów znormalizowanych max_odległość = 2 (na sferze jednostkowej)
        """
        # Oblicz wszystkie pary odległości
        distances = pdist(self.embeddings, metric='euclidean')
        
        if len(distances) == 0:
            return 1.0, 0.5
        
        mean_distance = np.mean(distances)
        
        # Dla sfery jednostkowej max_distance = 2
        max_distance = 2.0
        
        D = 1.0 - (mean_distance / max_distance)
        
        # Pewność oparta na wariancji odległości
        # Mała wariancja = wysoka pewność (spójny klaster)
        var_distance = np.var(distances)
        confidence = np.exp(-var_distance)
        
        # Modyfikacja pewności przez liczbę próbek
        sample_factor = min(1.0, np.log(self.n_samples + 1) / np.log(100))
        confidence = confidence * sample_factor
        
        return D, confidence
    
    def _compute_D_cosine_similarity(self) -> Tuple[float, float]:
        """
        Metoda 1b: Cosine Similarity
        
        D = (mean_cosine_similarity + 1) / 2
        
        Mapowanie z [-1, 1] na [0, 1]
        """
        # Macierz podobieństw kosinusowych
        # Dla znormalizowanych wektorów: cos_sim = dot product
        cos_sim_matrix = self.embeddings @ self.embeddings.T
        
        # Wyciągnij górny trójkąt (bez diagonali)
        upper_triangle_indices = np.triu_indices(self.n_samples, k=1)
        cos_similarities = cos_sim_matrix[upper_triangle_indices]
        
        if len(cos_similarities) == 0:
            return 1.0, 0.5
        
        mean_cos_sim = np.mean(cos_similarities)
        
        # Mapowanie [-1, 1] → [0, 1]
        D = (mean_cos_sim + 1.0) / 2.0
        
        # Pewność z wariancji
        var_cos_sim = np.var(cos_similarities)
        confidence = np.exp(-var_cos_sim * 2)  # Skalowanie dla większej czułości
        
        return D, confidence
    
    def _compute_D_diameter(self) -> Tuple[float, float]:
        """
        Metoda 1c: Diameter
        
        D = 1 - (diameter / 2)
        
        Diameter = maksymalna odległość między dwoma punktami w klastrze
        """
        distances = pdist(self.embeddings, metric='euclidean')
        
        if len(distances) == 0:
            return 1.0, 0.5
        
        diameter = np.max(distances)
        
        # Dla sfery jednostkowej max_diameter = 2
        D = 1.0 - (diameter / 2.0)
        
        # Pewność: porównanie średnicy do średniej odległości
        mean_distance = np.mean(distances)
        # Jeśli diameter ≈ mean_distance, klaster jest jednorodny
        if diameter > 0:
            homogeneity = mean_distance / diameter
            confidence = homogeneity
        else:
            confidence = 1.0
        
        return D, confidence

    def _compute_D_ambiguity_corrected(
        self,
        ambiguity: Optional[float] = None,
        source: AmbiguitySource = AmbiguitySource.GEOMETRIC
    ) -> Tuple[float, float]:
        """
        Metoda 1d: Ambiguity-Corrected Determination

        Koryguje D z cluster_cohesion o zewnętrzną wartość ambiguity
        (np. z analizy morfologicznej Morfeusza).

        Formuła GTMØ:
        D_corrected = D_geometric * (1 - α * ambiguity / ∅₀)

        gdzie:
        - D_geometric: bazowa determinacja z kohezji klastra
        - ambiguity: zewnętrzna miara wieloznaczności ∈ [0, max_ambiguity]
        - ∅₀ = INDEFINITENESS_CONSTANT = 1.2925 (stała kalibracyjna)
        - α = współczynnik korekcji (domyślnie 0.3)

        Parameters:
        -----------
        ambiguity : float, optional
            Zewnętrzna wartość ambiguity. Jeśli None, estymowana geometrycznie.
        source : AmbiguitySource
            Źródło wartości ambiguity

        Returns:
        --------
        Tuple[float, float]: (D_corrected, confidence)
        """
        # Bazowa determinacja z cluster_cohesion
        D_base, conf_base = self._compute_D_cluster_cohesion()

        # Określ źródło ambiguity
        if source == AmbiguitySource.EXTERNAL and ambiguity is not None:
            # Użyj zewnętrznej wartości (np. z Morfeusza)
            amb = ambiguity
            conf_amb = 0.9  # Wysoka pewność dla danych zewnętrznych
        elif source == AmbiguitySource.SUBWORD:
            # Estymacja z wariancji subword tokenów (wymaga tokenizera)
            amb = self._estimate_ambiguity_subword()
            conf_amb = 0.7
        elif source == AmbiguitySource.HYBRID and ambiguity is not None:
            # Kombinacja external + geometric
            amb_ext = ambiguity
            amb_geo = self._estimate_ambiguity_geometric()
            amb = 0.6 * amb_ext + 0.4 * amb_geo
            conf_amb = 0.85
        else:
            # Domyślnie: estymacja geometryczna
            amb = self._estimate_ambiguity_geometric()
            conf_amb = 0.6

        # Normalizacja ambiguity przez INDEFINITENESS_CONSTANT
        amb_normalized = min(1.0, amb / INDEFINITENESS_CONSTANT)

        # Współczynnik korekcji
        alpha = 0.3

        # Korekcja D
        D_corrected = D_base * (1.0 - alpha * amb_normalized)
        D_corrected = np.clip(D_corrected, 0.0, 1.0)

        # Łączona pewność
        confidence = conf_base * conf_amb

        return float(D_corrected), float(confidence)

    def _estimate_ambiguity_geometric(self) -> float:
        """
        Estymacja ambiguity z rozproszenia geometrycznego embeddingów.

        Wysoka wariancja → wysoka ambiguity
        """
        if self.n_samples < 2:
            return 0.0

        # Wariancja odległości od centroidu
        distances_to_centroid = np.linalg.norm(
            self.embeddings - self._centroid, axis=1
        )
        variance = np.var(distances_to_centroid)

        # Mapowanie wariancji na ambiguity
        # Kalibracja: var=0.1 → amb≈0.5, var=0.5 → amb≈1.0
        ambiguity = 1.0 - np.exp(-variance * 5.0)

        return float(np.clip(ambiguity, 0.0, 1.0))

    def _estimate_ambiguity_subword(self) -> float:
        """
        Estymacja ambiguity z wariancji subword tokenów.

        Placeholder - wymaga integracji z tokenizerem.
        """
        # TODO: Implementacja z użyciem tokenizera HerBERT
        warnings.warn("Metoda SUBWORD wymaga integracji z tokenizerem. Używam GEOMETRIC.")
        return self._estimate_ambiguity_geometric()

    # =========================================================================
    # KROK 2: OBLICZANIE STABILNOŚCI (S)
    # =========================================================================

    def compute_S(
        self,
        method: StabilityMethod = StabilityMethod.CONTEXTUAL_VARIANCE
    ) -> Tuple[float, float]:
        """
        Oblicz Stabilność (S) - odporność na perturbacje kontekstowe
        
        Semantyka GTMØ:
        - S → 1: morfem stabilny ("woda = H₂O") - nie zmienia znaczenia
        - S → 0: morfem zmienny ("cool") - ciągle ewoluuje
        
        KLUCZOWA RÓŻNICA od klasycznych podejść:
        σ² to wariancja MIĘDZY KONTEKSTAMI, nie rozkład symboli!
        
        Formuła: S = exp(-λ·σ²)
        
        Parameters:
        -----------
        method : StabilityMethod
            Metoda obliczania:
            - CONTEXTUAL_VARIANCE: wariancja wymiarów między próbkami
            - CENTROID_STABILITY: bootstrap centroidu
            - TEMPORAL_AUTOCORR: korelacja Spearmana dla danych diachronicznych
        
        Returns:
        --------
        Tuple[float, float]: (S, confidence)
        """
        cache_key = f"S_{method.value}"
        if cache_key in self._cache:
            return self._cache[cache_key]
        
        if self.n_samples < 2:
            # Pojedynczy embedding - maksymalna stabilność (brak zmienności)
            result = (1.0, 0.5)
            self._cache[cache_key] = result
            return result
        
        if method == StabilityMethod.CONTEXTUAL_VARIANCE:
            S, conf = self._compute_S_contextual_variance()
        elif method == StabilityMethod.SEMANTIC_SPREAD:
            S, conf = self._compute_S_semantic_spread()
        elif method == StabilityMethod.CENTROID_STABILITY:
            S, conf = self._compute_S_centroid_stability()
        elif method == StabilityMethod.TEMPORAL_AUTOCORR:
            S, conf = self._compute_S_temporal_autocorr()
        else:
            raise ValueError(f"Nieznana metoda: {method}")
        
        # Clip do [0, 1]
        S = np.clip(S, 0.0, 1.0)
        
        result = (float(S), float(conf))
        self._cache[cache_key] = result
        return result
    
    def _compute_S_contextual_variance(self) -> Tuple[float, float]:
        """
        Metoda 2a: Contextual Variance (domyślna)
        
        Wariancja każdego wymiaru embeddingu między różnymi próbkami kontekstowymi.
        
        var_per_dim = np.var(embeddings, axis=0)  # [768 wartości]
        mean_var = np.mean(var_per_dim)
        S = exp(-λ·mean_var)
        """
        # Wariancja każdego wymiaru między próbkami
        var_per_dim = np.var(self.embeddings, axis=0)
        
        # Średnia wariancja
        mean_variance = np.mean(var_per_dim)
        
        # Formuła GTMØ
        S = np.exp(-self.lambda_stability * mean_variance)
        
        # Pewność oparta na wariancji wariancji (meta-wariancja)
        var_of_var = np.var(var_per_dim)
        confidence = np.exp(-var_of_var * 10)  # Skalowane dla czułości
        
        # Modyfikacja przez liczbę próbek
        sample_factor = min(1.0, np.log(self.n_samples + 1) / np.log(100))
        confidence = confidence * sample_factor
        
        return S, confidence

    def _compute_S_semantic_spread(self) -> Tuple[float, float]:
        """
        Metoda 2a-bis: Semantic Spread (ULEPSZONA)

        Rozrzut semantyczny mierzony jako ŚREDNIA ODLEGŁOŚĆ od centroidu.

        Formuła GTMØ:
        mean_dist = mean(||e_i - centroid||)
        S = exp(-λ · mean_dist · scaling_factor)

        Intuicja:
        - Mała średnia odległość → embeddingi skupione → wysokie S
        - Duża średnia odległość → embeddingi rozproszone → niskie S

        Dla znormalizowanych embeddingów na sferze jednostkowej:
        - max_distance ≈ 2 (przeciwległe punkty)
        - typowe mean_dist dla skupionego klastra: 0.1-0.3
        - typowe mean_dist dla rozproszonego klastra: 0.5-1.0

        Skalowanie: factor = 3.0 daje dobry zakres S ∈ [0.3, 0.95]
        """
        # Oblicz odległości od centroidu
        distances = np.linalg.norm(self.embeddings - self._centroid, axis=1)

        # Średnia odległość od centroidu
        mean_dist = np.mean(distances)

        # Formuła eksponencjalna (analogiczna do contextual_variance)
        # Wysoka mean_dist → niska stabilność
        scaling_factor = 3.0  # Kalibracja dla typowych embeddingów 768D
        S = np.exp(-self.lambda_stability * mean_dist * scaling_factor)

        # Pewność: bazująca na liczbie próbek i spójności rozrzutu
        sample_factor = min(1.0, np.log(self.n_samples + 1) / np.log(50))

        # Dodatkowa pewność z wariancji odległości
        # Niska wariancja = spójny klaster = wyższa pewność
        std_dist = np.std(distances)
        cv = std_dist / (mean_dist + 1e-10)
        spread_consistency = np.exp(-cv * 2)

        confidence = sample_factor * spread_consistency

        return S, confidence

    def _compute_S_centroid_stability(
        self,
        n_bootstrap: int = 100,
        sample_fraction: float = 0.7
    ) -> Tuple[float, float]:
        """
        Metoda 2b: Centroid Stability (bootstrap)
        
        Losujemy podpróbki i sprawdzamy, jak bardzo zmienia się centroid.
        Stabilny morfem → centroid mało się zmienia przy różnych próbkowaniach.
        """
        if self.n_samples < 3:
            # Za mało próbek dla sensownego bootstrapu
            return self._compute_S_contextual_variance()
        
        n_sample = max(2, int(self.n_samples * sample_fraction))
        
        centroids = []
        for _ in range(n_bootstrap):
            indices = np.random.choice(self.n_samples, size=n_sample, replace=False)
            subsample_centroid = np.mean(self.embeddings[indices], axis=0)
            centroids.append(subsample_centroid)
        
        centroids = np.array(centroids)
        
        # Średnie odchylenie centroidów od globalnego centroidu
        deviations = np.linalg.norm(centroids - self._centroid, axis=1)
        mean_deviation = np.mean(deviations)
        
        # Formuła - konwersja na stabilność
        # Małe odchylenie → wysoka stabilność
        S = np.exp(-self.lambda_stability * mean_deviation * 10)  # Skalowanie
        
        # Pewność z wariancji odchyleń
        var_deviation = np.var(deviations)
        confidence = np.exp(-var_deviation * 50)
        
        return S, confidence
    
    def _compute_S_temporal_autocorr(self) -> Tuple[float, float]:
        """
        Metoda 2c: Temporal Autocorrelation (dla danych diachronicznych)
        
        Jeśli dostępne są embeddingi z różnych okresów czasowych,
        obliczana jest korelacja Spearmana między centroidami kolejnych okresów.
        
        S = (mean_correlation + 1) / 2
        """
        if self.temporal_epochs is None:
            # Brak danych czasowych - fallback do contextual variance
            warnings.warn("Brak danych diachronicznych, używam contextual_variance")
            return self._compute_S_contextual_variance()
        
        epochs = np.array(self.temporal_epochs)
        unique_epochs = np.unique(epochs)
        
        if len(unique_epochs) < 2:
            # Tylko jedna epoka
            return self._compute_S_contextual_variance()
        
        # Sortuj epoki chronologicznie
        unique_epochs = np.sort(unique_epochs)
        
        # Oblicz centroidy dla każdej epoki
        epoch_centroids = []
        for epoch in unique_epochs:
            mask = epochs == epoch
            epoch_embeddings = self.embeddings[mask]
            epoch_centroid = np.mean(epoch_embeddings, axis=0)
            epoch_centroids.append(epoch_centroid)
        
        # Korelacja Spearmana między kolejnymi centroidami
        correlations = []
        for i in range(len(epoch_centroids) - 1):
            corr, _ = spearmanr(epoch_centroids[i], epoch_centroids[i + 1])
            if not np.isnan(corr):
                correlations.append(corr)
        
        if len(correlations) == 0:
            return self._compute_S_contextual_variance()
        
        mean_corr = np.mean(correlations)
        
        # Mapowanie [-1, 1] → [0, 1]
        S = (mean_corr + 1.0) / 2.0
        
        # Pewność z wariancji korelacji
        var_corr = np.var(correlations) if len(correlations) > 1 else 0.5
        confidence = np.exp(-var_corr * 2)
        
        return S, confidence
    
    # =========================================================================
    # KROK 3: OBLICZANIE ENTROPII/DYNAMIKI (E)
    # =========================================================================
    
    def compute_E(
        self,
        method: EntropyMethod = EntropyMethod.COMBINED
    ) -> Tuple[float, float]:
        """
        Oblicz Entropię/Dynamikę (E) - tempo zmian i potencjał generatywny
        
        Semantyka GTMØ:
        - E → 1: morfem dynamiczny ("meme", "viral") - generuje nowe formy
        - E → 0: morfem statyczny ("krzesło") - przewidywalne użycie
        
        Parameters:
        -----------
        method : EntropyMethod
            Metoda obliczania:
            - COMBINED: 0.6·TV + 0.4·Julia (domyślna)
            - TOTAL_VARIATION: suma różnic między kolejnymi embeddingami
            - JULIA_ESCAPE: escape rate z iteracji Julia
        
        Returns:
        --------
        Tuple[float, float]: (E, confidence)
        """
        cache_key = f"E_{method.value}"
        if cache_key in self._cache:
            return self._cache[cache_key]
        
        if self.n_samples < 2:
            # Pojedynczy embedding - zerowa dynamika (brak zmienności)
            result = (0.0, 0.5)
            self._cache[cache_key] = result
            return result
        
        if method == EntropyMethod.TOTAL_VARIATION:
            E, conf = self._compute_E_total_variation()
        elif method == EntropyMethod.JULIA_ESCAPE:
            E, conf = self._compute_E_julia_escape()
        elif method == EntropyMethod.COMBINED:
            E, conf = self._compute_E_combined()
        else:
            raise ValueError(f"Nieznana metoda: {method}")
        
        # Clip do [0, 1]
        E = np.clip(E, 0.0, 1.0)
        
        result = (float(E), float(conf))
        self._cache[cache_key] = result
        return result
    
    def _compute_E_total_variation(self) -> Tuple[float, float]:
        """
        Metoda 3a: Total Variation
        
        TV = (1/n) Σ ||e_{i+1} - e_i||
        E = 1 - exp(-TV·3)
        
        Suma znormalizowanych różnic między kolejnymi embeddingami.
        Mierzy "tempo zmian" w przestrzeni semantycznej.
        """
        # Różnice między kolejnymi embeddingami
        differences = np.diff(self.embeddings, axis=0)
        
        # Normy różnic
        diff_norms = np.linalg.norm(differences, axis=1)
        
        # Total Variation
        TV = np.mean(diff_norms)
        
        # Konwersja na E ∈ [0, 1]
        # Wysoki TV → wysoka dynamika
        E = 1.0 - np.exp(-TV * 3.0)
        
        # Pewność z wariancji różnic
        var_diff = np.var(diff_norms)
        confidence = np.exp(-var_diff * 5)
        
        return E, confidence
    
    def _compute_E_julia_escape(self) -> Tuple[float, float]:
        """
        Metoda 3b: Julia Escape Rate
        
        Innowacyjne podejście łączące geometrię embeddingów z dynamiką fraktalną.
        
        Projekcja embeddingu na punkt zespolony:
        x = sigmoid(mean(embedding[:384]))  # pierwsza połowa
        y = sigmoid(mean(embedding[384:]))  # druga połowa
        z₀ = x + iy
        
        Iteracja Julia z c = -0.8 + 0.156i:
        for n in range(max_iter):
            if |z| > escape_radius:
                escape_rate = 1/(n+1)
                break
            z = z² + c
        
        Wysoki escape rate → punkt w chaotycznej części przestrzeni → E wysoka
        """
        escape_rates = []
        
        for emb in self.embeddings:
            z0, escape_rate = self._julia_escape_for_embedding(emb)
            escape_rates.append(escape_rate)
        
        mean_escape_rate = np.mean(escape_rates)
        
        # Mapowanie escape rate na E
        # Wysoki escape rate = szybka ucieczka = wysoka dynamika
        E = mean_escape_rate
        
        # Pewność z wariancji escape rates
        var_escape = np.var(escape_rates)
        confidence = np.exp(-var_escape * 10)
        
        return E, confidence
    
    def _julia_escape_for_embedding(
        self,
        embedding: np.ndarray
    ) -> Tuple[complex, float]:
        """
        Oblicz escape rate dla pojedynczego embeddingu
        
        Returns:
        --------
        Tuple[complex, float]: (z0, escape_rate)
        """
        half_dim = self.embedding_dim // 2
        
        # Projekcja na punkt zespolony
        # Używamy sigmoid dla mapowania na [0, 1]
        x = self._sigmoid(np.mean(embedding[:half_dim]))
        y = self._sigmoid(np.mean(embedding[half_dim:]))
        
        z0 = complex(x, y)
        z = z0
        c = self.julia_c
        
        # Iteracja Julia
        for n in range(MAX_JULIA_ITERATIONS):
            if abs(z) > ESCAPE_RADIUS:
                # Ucieczka! Oblicz escape rate
                escape_rate = 1.0 / (n + 1)
                return z0, escape_rate
            z = z * z + c
        
        # Brak ucieczki - punkt w zbiorze Julia (lub blisko)
        return z0, 0.0
    
    def _compute_E_combined(self) -> Tuple[float, float]:
        """
        Metoda 3c: Combined (domyślna) z korekcją przez D

        Formuła GTMØ z adaptacyjnymi wagami:
        - Dla wysokiego D: więcej wagi na julia_escape (która jest NISKA dla stabilnych)
        - Dla niskiego D: więcej wagi na total_variation (wyższa zmienność)

        Kluczowa obserwacja:
        - E_julia dla stabilnych morfemów (D > 0.7) jest NISKA (~0.17)
        - E_tv może być wysoka z powodu szumu w embeddingach
        - Dla morfemów o wysokiej determinacji, julia lepiej odzwierciedla stabilność

        Formuła adaptacyjna:
        - D > 0.7: E = 0.3·E_tv + 0.7·E_julia (julia dominuje → niskie E)
        - D < 0.4: E = 0.7·E_tv + 0.3·E_julia (TV dominuje → wyższe E)
        """
        E_tv, conf_tv = self._compute_E_total_variation()
        E_julia, conf_julia = self._compute_E_julia_escape()

        # Pobierz D dla adaptacji wag (użyj cache jeśli dostępne)
        D, _ = self.compute_D()

        # Adaptacyjne wagi w zależności od D
        # Wysokie D → więcej julia (która jest niska) → niskie E
        # Niskie D → więcej TV (która jest wysoka) → wysokie E
        if D > 0.7:
            # Wysoka determinacja → julia dominuje (E_julia jest niska)
            weight_tv = 0.3
            weight_julia = 0.7
        elif D > 0.4:
            # Średnia determinacja → zbalansowane
            weight_tv = 0.5
            weight_julia = 0.5
        else:
            # Niska determinacja → TV dominuje (E_tv jest wysoka)
            weight_tv = 0.7
            weight_julia = 0.3

        # Ważona średnia
        E = weight_tv * E_tv + weight_julia * E_julia

        # Połączona pewność
        confidence = weight_tv * conf_tv + weight_julia * conf_julia

        return E, confidence
    
    # =========================================================================
    # KROK 4: OBLICZANIE WYMIARÓW KIERUNKOWYCH K⁶
    # =========================================================================
    
    def compute_K6(self, D: float = None, S: float = None, E: float = None) -> K6Result:
        """
        Oblicz wymiary kierunkowe K⁶ przez iteracje Julia
        
        Dla każdej pary współrzędnych F³ wykonywane są iteracje Julia:
        
        | Płaszczyzna | Punkt startowy z₀ | Wynik               |
        |-------------|-------------------|---------------------|
        | S-E (oś D)  | S + iE            | θ_D, φ_D, escape_D  |
        | D-E (oś S)  | D + iE            | θ_S, φ_S, escape_S  |
        | D-S (oś E)  | D + iS            | θ_E, φ_E, escape_E  |
        
        gdzie:
        - θ = arccos(Im(z)/|z|) — kąt polarny kierunku ucieczki
        - φ = arg(z) — kąt azymutalny
        - escape_rate = 1/(n_iter + 1) — tempo ucieczki
        
        Parameters:
        -----------
        D, S, E : float, optional
            Współrzędne F³. Jeśli None, obliczane automatycznie.
        
        Returns:
        --------
        K6Result: wynik z kierunkami dla każdej osi
        """
        # Oblicz D, S, E jeśli nie podano
        if D is None:
            D, _ = self.compute_D()
        if S is None:
            S, _ = self.compute_S()
        if E is None:
            E, _ = self.compute_E()
        
        # Kierunki dla każdej płaszczyzny
        k6 = K6Result()
        
        # Płaszczyzna S-E → kierunki dla osi D
        theta_D, phi_D, rho_D, escaped_D = self._julia_directions(S, E)
        k6.theta_D = theta_D
        k6.phi_D = phi_D
        k6.rho_D = rho_D
        k6.escaped_D = escaped_D
        
        # Płaszczyzna D-E → kierunki dla osi S
        theta_S, phi_S, rho_S, escaped_S = self._julia_directions(D, E)
        k6.theta_S = theta_S
        k6.phi_S = phi_S
        k6.rho_S = rho_S
        k6.escaped_S = escaped_S
        
        # Płaszczyzna D-S → kierunki dla osi E
        theta_E, phi_E, rho_E, escaped_E = self._julia_directions(D, S)
        k6.theta_E = theta_E
        k6.phi_E = phi_E
        k6.rho_E = rho_E
        k6.escaped_E = escaped_E
        
        return k6
    
    def _julia_directions(
        self,
        x: float,
        y: float
    ) -> Tuple[float, float, float, bool]:
        """
        Iteracje Julia dla pary współrzędnych (x, y)
        
        Parameters:
        -----------
        x, y : float
            Współrzędne startowe ∈ [0, 1]
        
        Returns:
        --------
        Tuple[float, float, float, bool]: (theta, phi, rho, escaped)
            - theta: kąt polarny ∈ [0, π]
            - phi: kąt azymutalny ∈ [0, 2π]
            - rho: tempo ucieczki (escape rate)
            - escaped: czy nastąpiła ucieczka
        """
        z0 = complex(x, y)
        z = z0
        
        # Dynamiczna modulacja parametru c przez trzecią współrzędną
        # Zgodnie z dokumentacją: c = c_base · (1 + α·E)
        # Tu używamy średniej z obu współrzędnych jako modulatora
        modulator = (x + y) / 2.0
        alpha = 0.1
        c = self.julia_c * (1 + alpha * modulator)
        
        # Iteracja Julia
        for n in range(MAX_JULIA_ITERATIONS):
            if abs(z) > ESCAPE_RADIUS:
                # ESCAPE = TRUE
                # Ekstrakcja kierunków z dokumentacji GTMØ
                
                # θ = arccos(Im(z)/|z|) ∈ [0, π]
                abs_z = abs(z)
                theta = np.arccos(z.imag / abs_z) if abs_z > 0 else 0.0
                
                # φ = arg(z) ∈ [0, 2π]
                phi = np.angle(z)
                if phi < 0:
                    phi += 2 * np.pi
                
                # ρ = (1/n) · log(|z_n|/|z₀|)
                abs_z0 = abs(z0) if abs(z0) > 0 else 1e-10
                rho = (1.0 / (n + 1)) * np.log(abs_z / abs_z0)
                
                return theta, phi, rho, True
            
            z = z * z + c
        
        # ESCAPE = FALSE (wnętrze zbioru Julia)
        return 0.0, 0.0, 0.0, False
    
    # =========================================================================
    # KROK 5: PEŁNA PRZESTRZEŃ Φ⁹
    # =========================================================================
    
    def compute_DSE(
        self,
        method_D: DeterminationMethod = DeterminationMethod.CLUSTER_COHESION,
        method_S: StabilityMethod = StabilityMethod.SEMANTIC_SPREAD,
        method_E: EntropyMethod = EntropyMethod.COMBINED
    ) -> DSEResult:
        """
        Oblicz pełne współrzędne F³ = [D, S, E]
        
        Parameters:
        -----------
        method_D, method_S, method_E : Enum
            Metody obliczania poszczególnych współrzędnych
        
        Returns:
        --------
        DSEResult: wynik z D, S, E i metadanymi
        """
        D, conf_D = self.compute_D(method=method_D)
        S, conf_S = self.compute_S(method=method_S)
        E, conf_E = self.compute_E(method=method_E)
        
        return DSEResult(
            D=D, S=S, E=E,
            confidence_D=conf_D,
            confidence_S=conf_S,
            confidence_E=conf_E,
            method_D=method_D.value,
            method_S=method_S.value,
            method_E=method_E.value,
            n_samples=self.n_samples
        )
    
    def compute_Phi9(
        self,
        method_D: DeterminationMethod = DeterminationMethod.CLUSTER_COHESION,
        method_S: StabilityMethod = StabilityMethod.SEMANTIC_SPREAD,
        method_E: EntropyMethod = EntropyMethod.COMBINED,
        morpheme: str = ""
    ) -> Phi9Result:
        """
        Oblicz pełną przestrzeń Φ⁹ = F³ × K⁶
        
        Φ⁹ = [D, S, E, θ_D, φ_D, θ_S, φ_S, θ_E, φ_E]
        
        To 9-wymiarowa reprezentacja morfemu w pełnej przestrzeni GTMØ.
        
        Parameters:
        -----------
        method_D, method_S, method_E : Enum
            Metody obliczania współrzędnych F³
        morpheme : str
            Nazwa morfemu (opcjonalna, dla identyfikacji)
        
        Returns:
        --------
        Phi9Result: pełny wynik w Φ⁹
        """
        # Oblicz F³
        dse = self.compute_DSE(
            method_D=method_D,
            method_S=method_S,
            method_E=method_E
        )
        
        # Oblicz K⁶ na podstawie F³
        k6 = self.compute_K6(D=dse.D, S=dse.S, E=dse.E)
        
        return Phi9Result(
            dse=dse,
            k6=k6,
            morpheme=morpheme
        )
    
    # =========================================================================
    # METODY POMOCNICZE
    # =========================================================================
    
    @staticmethod
    def _sigmoid(x: float) -> float:
        """Funkcja sigmoidalna dla mapowania na [0, 1]"""
        return 1.0 / (1.0 + np.exp(-x))
    
    def clear_cache(self):
        """Wyczyść cache obliczonych wartości"""
        self._cache.clear()
    
    @property
    def centroid(self) -> np.ndarray:
        """Centroid klastra embeddingów"""
        return self._centroid.copy()
    
    def distance_to_attractor(
        self,
        attractor_coords: np.ndarray,
        attractor_strength: float = 1.0
    ) -> float:
        """
        Oblicz efektywną odległość do atraktora GTMØ
        
        dist_effective = dist_euclidean / G_i
        
        gdzie G_i to siła atraktora
        """
        dse = self.compute_DSE()
        point = dse.to_array()
        
        diff = point - attractor_coords
        dist_euclidean = np.linalg.norm(diff)
        
        return dist_euclidean / attractor_strength
    
    # =========================================================================
    # METODA DLA POJEDYNCZEGO EMBEDDINGU (zgodna z dokumentacją GTMØ)
    # =========================================================================
    
    def compute_DSE_single_embedding(
        self,
        embedding: Optional[np.ndarray] = None,
        lambda_stability: Optional[float] = None
    ) -> DSEResult:
        """
        Oblicz D-S-E dla POJEDYNCZEGO embeddingu
        
        POPRAWIONE FORMUŁY dla pojedynczych embeddingów:
        
        1. DETERMINACJA - Hoyer Sparsity Measure
           Mierzy "skupienie" aktywacji embeddingu.
           - Sparsity → 1: kilka dominujących wymiarów (wysokie D)
           - Sparsity → 0: równomierne aktywacje (niskie D)
           
           Formuła:
           sparsity = (√n - ||v||₁/||v||₂) / (√n - 1)
           D = sparsity^α  (α kontroluje czułość)
        
        2. STABILNOŚĆ - Inverse Variance (z korekcją)
           S = 1 / (1 + β × σ²)
           gdzie β jest kalibrowany aby S ∈ [0,1] dla typowych embeddingów
        
        3. ENTROPIA - znormalizowana entropia rozkładu aktywacji
           E = H / H_max
        """
        if embedding is None:
            embedding = np.mean(self.raw_embeddings, axis=0)
        
        if lambda_stability is None:
            lambda_stability = self.lambda_stability
        
        dim = len(embedding)
        sqrt_dim = np.sqrt(dim)
        
        # =====================================================================
        # DETERMINACJA - Hoyer Sparsity Measure
        # =====================================================================
        abs_emb = np.abs(embedding)
        
        # L1 i L2 normy
        L1_norm = np.sum(abs_emb)
        L2_norm = np.linalg.norm(abs_emb)
        
        if L2_norm > 0:
            # Hoyer sparsity: (√n - L1/L2) / (√n - 1)
            # Zakres: [0, 1] gdzie 1 = maksymalna sparsity (jeden element dominuje)
            hoyer_sparsity = (sqrt_dim - L1_norm / L2_norm) / (sqrt_dim - 1)
            hoyer_sparsity = np.clip(hoyer_sparsity, 0, 1)
        else:
            hoyer_sparsity = 0.0
        
        # D jako funkcja sparsity z kontrolowaną czułością
        # α < 1 sprawia że D jest mniej ekstremalne
        alpha = 0.7
        D = hoyer_sparsity ** alpha
        
        # =====================================================================
        # STABILNOŚĆ - Inverse Variance z kalibracją
        # =====================================================================
        variance = np.var(embedding)
        
        # Kalibracja: typowe embeddingi HerBERT mają var ≈ 0.5-0.8
        # Chcemy aby S ≈ 0.8 dla var = 0.6 (typowa wartość)
        # S = 1/(1 + β*var), dla S=0.8, var=0.6: β = (1/0.8 - 1)/0.6 ≈ 0.42
        beta = 0.5
        S = 1.0 / (1.0 + beta * variance)
        S = np.clip(S, 0, 1)
        
        # =====================================================================
        # ENTROPIA - znormalizowana
        # =====================================================================
        p = abs_emb / np.sum(abs_emb) if np.sum(abs_emb) > 0 else np.ones(dim) / dim
        p = p + 1e-10
        p = p / np.sum(p)
        
        H = -np.sum(p * np.log2(p))
        H_max = np.log2(dim)
        
        E = H / H_max
        E = np.clip(E, 0, 1)
        
        # =====================================================================
        # Korekcja D-E: Kontinuum GTMØ
        # =====================================================================
        # Zgodnie z teorią GTMØ: D i E powinny być antypodalne
        # Wysokie D → niskie E i odwrotnie
        # Korygujemy D uwzględniając E
        
        # Współczynnik antykorelacji
        # Jeśli E jest wysokie, D powinno być niższe
        gamma = 0.3
        D_corrected = D * (1 - gamma * E)
        D_corrected = np.clip(D_corrected, 0, 1)
        
        # =====================================================================
        # Confidence scores
        # =====================================================================
        conf_D = min(1.0, hoyer_sparsity + 0.5)  # Wyższa sparsity = wyższa pewność
        conf_S = min(1.0, 1.0 / (1.0 + variance * 0.5))
        conf_E = min(1.0, E + 0.5)
        
        return DSEResult(
            D=float(D_corrected),
            S=float(S),
            E=float(E),
            confidence_D=float(conf_D),
            confidence_S=float(conf_S),
            confidence_E=float(conf_E),
            method_D="hoyer_sparsity",
            method_S="inverse_variance",
            method_E="normalized_entropy",
            n_samples=1
        )
    
    def compute_Phi9_single(
        self,
        embedding: Optional[np.ndarray] = None,
        morpheme: str = ""
    ) -> Phi9Result:
        """
        Oblicz pełną przestrzeń Φ⁹ dla POJEDYNCZEGO embeddingu
        """
        dse = self.compute_DSE_single_embedding(embedding)
        k6 = self.compute_K6(D=dse.D, S=dse.S, E=dse.E)
        
        return Phi9Result(
            dse=dse,
            k6=k6,
            morpheme=morpheme
        )


# =============================================================================
# FUNKCJE POMOCNICZE DO KLASYFIKACJI ATRAKTORÓW
# =============================================================================

# 8 atraktorów GTMØ zgodnie z dokumentacją
GTMO_ATTRACTORS = {
    "Ø_Singularity": {
        "coords": np.array([1.00, 1.00, 0.00]),
        "strength": 1.0,
        "description": "Paradoks, singularność semantyczna"
    },
    "ℓ∅_Alienated": {
        "coords": np.array([0.999, 0.999, 0.001]),
        "strength": 0.8,
        "description": "Wyalienowane znaczenie, infinitezymalnie blisko Ø"
    },
    "Ψᴷ_Particle": {
        "coords": np.array([0.85, 0.85, 0.15]),
        "strength": 1.2,
        "description": "Cząstka semantyczna - stabilne znaczenie"
    },
    "Ψʰ_Shadow": {
        "coords": np.array([0.15, 0.15, 0.85]),
        "strength": 1.2,
        "description": "Cień - symetrycznie do Ψᴷ"
    },
    "Ψᴺ_Emergent": {
        "coords": np.array([0.50, 0.30, 0.90]),
        "strength": 0.9,
        "description": "Emergentne znaczenie - wysokie E"
    },
    "Ψ↑_Transcendent": {
        "coords": np.array([0.70, 0.70, 0.30]),
        "strength": 0.7,
        "description": "Transcendentne - między Ψᴷ a Ø"
    },
    "Ψ~_Flux": {
        "coords": np.array([0.50, 0.50, 0.80]),
        "strength": 0.6,
        "description": "Strumień - centralny, entropiczny"
    },
    "Ψ◊_Void": {
        "coords": np.array([0.00, 0.00, 0.50]),
        "strength": 0.5,
        "description": "Pustka semantyczna"
    }
}


def classify_to_attractor(dse_result: DSEResult) -> Tuple[str, float]:
    """
    Klasyfikuj punkt F³ do najbliższego atraktora GTMØ
    
    Parameters:
    -----------
    dse_result : DSEResult
        Wynik obliczeń D-S-E
    
    Returns:
    --------
    Tuple[str, float]: (nazwa_atraktora, efektywna_odległość)
    """
    point = dse_result.to_array()
    
    min_distance = float('inf')
    nearest_attractor = None
    
    for name, attractor in GTMO_ATTRACTORS.items():
        diff = point - attractor["coords"]
        dist_euclidean = np.linalg.norm(diff)
        
        # Efektywna odległość z wagą siły
        dist_effective = dist_euclidean / attractor["strength"]
        
        if dist_effective < min_distance:
            min_distance = dist_effective
            nearest_attractor = name
    
    return nearest_attractor, min_distance


# =============================================================================
# CZASOPRZESTRZEŃ SEMANTYCZNA GTMØ (Minkowski-like)
# =============================================================================

class IntervalType(Enum):
    """Typ interwału semantycznego w czasoprzestrzeni GTMØ"""
    SPACELIKE = "spacelike"    # I² > 0: różnica kontekstowa (niezależne znaczenia)
    TIMELIKE = "timelike"      # I² < 0: ewolucja w czasie (jedno pochodzi od drugiego)
    LIGHTLIKE = "lightlike"    # I² = 0: przejście fazowe (granica między znaczeniami)


# Stała κ (kappa) dla metryki czasoprzestrzennej
# Kontroluje "prędkość światła semantycznego" - jak szybko S wpływa na interwał
# Kalibracja: κ = 1.0 daje równowagę między S a (D, E)
KAPPA_SEMANTIC = 1.0

# Tolerancja dla lightlike (przejście fazowe)
LIGHTLIKE_TOLERANCE = 0.01


@dataclass
class SemanticInterval:
    """
    Wynik obliczenia interwału semantycznego między dwoma punktami F³

    Pseudometryka Minkowskiego:
    ds² = -κ²dS² + dD² + dE²

    Sygnatura (-, +, +):
    - S (Stabilność) jest wymiarem "czasowym" (znak ujemny)
    - D (Determinacja) jest wymiarem "przestrzennym" (znak dodatni)
    - E (Entropia) jest wymiarem "przestrzennym" (znak dodatni)

    Interpretacja:
    - I² > 0 (spacelike): Dwa niezależne znaczenia w różnych kontekstach
    - I² < 0 (timelike): Ewolucja znaczenia w czasie (przyczynowość)
    - I² = 0 (lightlike): Przejście fazowe, granica semantyczna
    """
    ds_squared: float          # Wartość interwału ds²
    interval_type: IntervalType
    dD: float                  # Różnica w Determinacji
    dS: float                  # Różnica w Stabilności
    dE: float                  # Różnica w Entropii
    point1: np.ndarray         # Pierwszy punkt [D, S, E]
    point2: np.ndarray         # Drugi punkt [D, S, E]

    @property
    def proper_distance(self) -> float:
        """
        Odległość właściwa (proper distance) dla interwałów spacelike.
        Dla timelike zwraca proper time (czas własny).
        """
        if self.ds_squared >= 0:
            return np.sqrt(self.ds_squared)
        else:
            return np.sqrt(-self.ds_squared)  # proper time

    def __repr__(self) -> str:
        return (f"SemanticInterval(ds²={self.ds_squared:.6f}, "
                f"type={self.interval_type.value}, "
                f"dD={self.dD:.4f}, dS={self.dS:.4f}, dE={self.dE:.4f})")


def compute_semantic_interval(
    point1: Union[DSEResult, np.ndarray, Tuple[float, float, float]],
    point2: Union[DSEResult, np.ndarray, Tuple[float, float, float]],
    kappa: float = KAPPA_SEMANTIC
) -> SemanticInterval:
    """
    Oblicz interwał semantyczny między dwoma punktami w przestrzeni F³

    Formuła GTMØ (pseudometryka Minkowskiego):
    ds² = -κ²dS² + dD² + dE²

    gdzie:
    - dD = D₂ - D₁ (różnica Determinacji)
    - dS = S₂ - S₁ (różnica Stabilności)
    - dE = E₂ - E₁ (różnica Entropii)
    - κ = stała skalowania dla wymiaru czasowego

    Parameters:
    -----------
    point1, point2 : DSEResult, np.ndarray, lub Tuple
        Punkty w przestrzeni F³ = [D, S, E]
    kappa : float
        Stała κ dla metryki (domyślnie 1.0)

    Returns:
    --------
    SemanticInterval: wynik z ds², typem interwału i składowymi

    Examples:
    ---------
    >>> dse1 = DSEResult(D=0.8, S=0.9, E=0.2)
    >>> dse2 = DSEResult(D=0.3, S=0.4, E=0.7)
    >>> interval = compute_semantic_interval(dse1, dse2)
    >>> print(f"Typ: {interval.interval_type.value}, ds²={interval.ds_squared:.4f}")
    """
    # Konwersja do numpy arrays
    if isinstance(point1, DSEResult):
        p1 = point1.to_array()
    elif isinstance(point1, tuple):
        p1 = np.array(point1, dtype=np.float64)
    else:
        p1 = np.asarray(point1, dtype=np.float64)

    if isinstance(point2, DSEResult):
        p2 = point2.to_array()
    elif isinstance(point2, tuple):
        p2 = np.array(point2, dtype=np.float64)
    else:
        p2 = np.asarray(point2, dtype=np.float64)

    # Różnice współrzędnych: [D, S, E]
    dD = p2[0] - p1[0]
    dS = p2[1] - p1[1]
    dE = p2[2] - p1[2]

    # Metryka Minkowskiego: ds² = -κ²dS² + dD² + dE²
    # S jest wymiarem "czasowym" (znak ujemny)
    ds_squared = -(kappa ** 2) * (dS ** 2) + (dD ** 2) + (dE ** 2)

    # Klasyfikacja interwału
    if abs(ds_squared) < LIGHTLIKE_TOLERANCE:
        interval_type = IntervalType.LIGHTLIKE
    elif ds_squared > 0:
        interval_type = IntervalType.SPACELIKE
    else:
        interval_type = IntervalType.TIMELIKE

    return SemanticInterval(
        ds_squared=float(ds_squared),
        interval_type=interval_type,
        dD=float(dD),
        dS=float(dS),
        dE=float(dE),
        point1=p1,
        point2=p2
    )


def classify_semantic_relation(
    point1: Union[DSEResult, np.ndarray],
    point2: Union[DSEResult, np.ndarray],
    kappa: float = KAPPA_SEMANTIC
) -> Tuple[IntervalType, str]:
    """
    Klasyfikuj relację semantyczną między dwoma morfemami

    Returns:
    --------
    Tuple[IntervalType, str]: (typ_interwału, opis_relacji)
    """
    interval = compute_semantic_interval(point1, point2, kappa)

    descriptions = {
        IntervalType.SPACELIKE: (
            "Różnica kontekstowa - dwa niezależne znaczenia. "
            "Morfemy istnieją w różnych 'miejscach' semantycznych."
        ),
        IntervalType.TIMELIKE: (
            "Ewolucja temporalna - jedno znaczenie pochodzi od drugiego. "
            "Istnieje związek przyczynowy między morfemami."
        ),
        IntervalType.LIGHTLIKE: (
            "Przejście fazowe - granica między znaczeniami. "
            "Morfemy są na krawędzi transformacji semantycznej."
        )
    }

    return interval.interval_type, descriptions[interval.interval_type]


# =============================================================================
# KOMPOZYCJA MORFEMÓW (Chemia znaczenia)
# =============================================================================

@dataclass
class MorphemeComposition:
    """
    Wynik kompozycji morfemów w przestrzeni F³

    Model GTMØ:
    Φ(słowo) = Φ(rdzeń) + Σ Δ(afiks)

    gdzie:
    - Φ(rdzeń) = [D_root, S_root, E_root] - współrzędne rdzenia
    - Δ(afiks) = [δD, δS, δE] - wektor przesunięcia afiksu
    """
    root_dse: np.ndarray           # Φ(rdzeń) = [D, S, E]
    affix_deltas: List[np.ndarray] # Lista Δ(afiks) = [δD, δS, δE]
    composed_dse: np.ndarray       # Φ(słowo) = wynik
    root_name: str = ""
    affix_names: List[str] = field(default_factory=list)

    def to_DSEResult(self) -> DSEResult:
        """Konwertuj wynik do DSEResult"""
        # Clip do [0, 1] po kompozycji
        D = float(np.clip(self.composed_dse[0], 0.0, 1.0))
        S = float(np.clip(self.composed_dse[1], 0.0, 1.0))
        E = float(np.clip(self.composed_dse[2], 0.0, 1.0))

        return DSEResult(
            D=D, S=S, E=E,
            method_D="composition",
            method_S="composition",
            method_E="composition",
            n_samples=0
        )

    def __repr__(self) -> str:
        affixes_str = " + ".join(self.affix_names) if self.affix_names else "Σ afiksy"
        return (f"MorphemeComposition({self.root_name} + {affixes_str}): "
                f"[D={self.composed_dse[0]:.4f}, S={self.composed_dse[1]:.4f}, "
                f"E={self.composed_dse[2]:.4f}]")


# Typowe delty afiksów polskich (przykładowe wartości do kalibracji)
# Format: Δ = [δD, δS, δE]
POLISH_AFFIX_DELTAS = {
    # Prefiksy
    "nie-": np.array([-0.1, 0.0, 0.05]),      # Negacja: obniża D, lekko podnosi E
    "prze-": np.array([0.0, -0.1, 0.15]),     # Intensyfikacja: obniża S, podnosi E
    "do-": np.array([0.05, 0.1, -0.05]),      # Dopełnienie: podnosi D i S
    "wy-": np.array([0.0, -0.05, 0.1]),       # Ekstrakcja: lekko podnosi E
    "za-": np.array([0.1, 0.05, 0.0]),        # Rozpoczęcie: podnosi D i S
    "od-": np.array([-0.05, -0.05, 0.1]),     # Separacja: obniża D i S, podnosi E
    "nad-": np.array([0.05, 0.0, 0.05]),      # Nadmiar: lekko podnosi D i E
    "pod-": np.array([-0.05, 0.05, 0.0]),     # Podporządkowanie: obniża D, podnosi S

    # Sufiksy rzeczownikowe
    "-ość": np.array([0.15, 0.1, -0.1]),      # Abstrahowanie: podnosi D i S
    "-stwo": np.array([0.1, 0.15, -0.05]),    # Kolektyw: podnosi D i S
    "-enie": np.array([-0.05, 0.0, 0.1]),     # Nominalizacja: lekko podnosi E
    "-anie": np.array([-0.05, 0.0, 0.1]),     # Nominalizacja
    "-nik": np.array([0.1, 0.1, 0.0]),        # Agentywność: podnosi D i S
    "-ka": np.array([0.0, 0.05, 0.0]),        # Deminutyw żeński
    "-ek": np.array([0.0, 0.05, -0.05]),      # Deminutyw męski

    # Sufiksy przymiotnikowe
    "-owy": np.array([0.1, 0.05, 0.0]),       # Relacyjny: podnosi D
    "-ny": np.array([0.05, 0.1, 0.0]),        # Jakościowy: podnosi S
    "-ski": np.array([0.1, 0.1, -0.05]),      # Przynależność: podnosi D i S
    "-liwy": np.array([-0.1, -0.1, 0.15]),    # Skłonność: obniża D i S, podnosi E

    # Sufiksy czasownikowe
    "-ować": np.array([0.0, -0.05, 0.1]),     # Werbalizacja: podnosi E
    "-ywać": np.array([-0.05, -0.1, 0.15]),   # Iteratywność: obniża S, podnosi E
}


def compose_morphemes(
    root_dse: Union[DSEResult, np.ndarray, Tuple[float, float, float]],
    affix_deltas: List[Union[np.ndarray, Tuple[float, float, float], str]],
    root_name: str = "rdzeń",
    affix_names: Optional[List[str]] = None,
    clip_result: bool = True
) -> MorphemeComposition:
    """
    Składa morfemy według formuły GTMØ:
    Φ(słowo) = Φ(rdzeń) + Σ Δ(afiks)

    Parameters:
    -----------
    root_dse : DSEResult, np.ndarray, lub Tuple
        Współrzędne F³ rdzenia morfemu
    affix_deltas : List
        Lista delt afiksów. Każdy element może być:
        - np.ndarray [δD, δS, δE]
        - Tuple (δD, δS, δE)
        - str - nazwa afiksu z POLISH_AFFIX_DELTAS
    root_name : str
        Nazwa rdzenia (dla opisu)
    affix_names : List[str], optional
        Nazwy afiksów (dla opisu)
    clip_result : bool
        Czy przyciąć wynik do [0, 1]³

    Returns:
    --------
    MorphemeComposition: wynik kompozycji

    Examples:
    ---------
    >>> root = DSEResult(D=0.7, S=0.8, E=0.3)  # "pis-" (pisać)
    >>> composed = compose_morphemes(root, ["-anie", "nie-"], root_name="pis")
    >>> print(composed)  # "niepisanie"
    """
    # Konwersja rdzenia do numpy
    if isinstance(root_dse, DSEResult):
        root = root_dse.to_array()
    elif isinstance(root_dse, tuple):
        root = np.array(root_dse, dtype=np.float64)
    else:
        root = np.asarray(root_dse, dtype=np.float64)

    # Przetwórz delty afiksów
    processed_deltas = []
    resolved_names = []

    for i, delta in enumerate(affix_deltas):
        if isinstance(delta, str):
            # Nazwa afiksu - pobierz z słownika
            if delta in POLISH_AFFIX_DELTAS:
                processed_deltas.append(POLISH_AFFIX_DELTAS[delta])
                resolved_names.append(delta)
            else:
                warnings.warn(f"Nieznany afiks '{delta}', pomijam")
                continue
        elif isinstance(delta, tuple):
            processed_deltas.append(np.array(delta, dtype=np.float64))
            resolved_names.append(affix_names[i] if affix_names and i < len(affix_names) else f"Δ{i}")
        else:
            processed_deltas.append(np.asarray(delta, dtype=np.float64))
            resolved_names.append(affix_names[i] if affix_names and i < len(affix_names) else f"Δ{i}")

    # Kompozycja: Φ(słowo) = Φ(rdzeń) + Σ Δ(afiks)
    composed = root.copy()
    for delta in processed_deltas:
        composed = composed + delta

    # Opcjonalne przycinanie do [0, 1]³
    if clip_result:
        composed = np.clip(composed, 0.0, 1.0)

    return MorphemeComposition(
        root_dse=root,
        affix_deltas=processed_deltas,
        composed_dse=composed,
        root_name=root_name,
        affix_names=resolved_names
    )


def analyze_morpheme_derivation(
    base_dse: Union[DSEResult, np.ndarray],
    derived_dse: Union[DSEResult, np.ndarray],
    base_name: str = "baza",
    derived_name: str = "derywat"
) -> Dict:
    """
    Analizuje derywację morfologiczną przez porównanie D-S-E

    Oblicza:
    1. Interwał semantyczny między bazą a derywatem
    2. Estymowaną deltę afiksu
    3. Typ transformacji semantycznej

    Returns:
    --------
    Dict z analizą derywacji
    """
    # Konwersja do numpy
    if isinstance(base_dse, DSEResult):
        base = base_dse.to_array()
    else:
        base = np.asarray(base_dse, dtype=np.float64)

    if isinstance(derived_dse, DSEResult):
        derived = derived_dse.to_array()
    else:
        derived = np.asarray(derived_dse, dtype=np.float64)

    # Oblicz interwał
    interval = compute_semantic_interval(base, derived)

    # Estymowana delta afiksu
    estimated_delta = derived - base

    # Analiza wpływu na D-S-E
    impact = {
        "D": "↑ zwiększa" if estimated_delta[0] > 0.05 else ("↓ zmniejsza" if estimated_delta[0] < -0.05 else "→ bez zmian"),
        "S": "↑ zwiększa" if estimated_delta[1] > 0.05 else ("↓ zmniejsza" if estimated_delta[1] < -0.05 else "→ bez zmian"),
        "E": "↑ zwiększa" if estimated_delta[2] > 0.05 else ("↓ zmniejsza" if estimated_delta[2] < -0.05 else "→ bez zmian"),
    }

    return {
        "base_name": base_name,
        "derived_name": derived_name,
        "base_dse": base,
        "derived_dse": derived,
        "estimated_delta": estimated_delta,
        "interval": interval,
        "impact": impact,
        "transformation_type": interval.interval_type.value
    }


# =============================================================================
# PRZYKŁAD UŻYCIA
# =============================================================================

if __name__ == "__main__":
    print("=" * 70)
    print("GeometricDSEComputer - Demo GTMØ")
    print("=" * 70)
    
    # Symulacja embeddingów dla morfemu "woda" (wysokie D, wysokie S, niskie E)
    np.random.seed(42)
    
    # "woda" - skupione embeddingi z małą wariancją
    centroid_woda = np.random.randn(768)
    embeddings_woda = centroid_woda + np.random.randn(50, 768) * 0.1
    
    # "rzecz" - rozproszone embeddingi z dużą wariancją
    embeddings_rzecz = np.random.randn(50, 768) * 0.5
    
    print("\n--- Morfem: 'woda' (oczekiwane: wysokie D, wysokie S, niskie E) ---")
    computer_woda = GeometricDSEComputer(embeddings_woda)
    result_woda = computer_woda.compute_Phi9(morpheme="woda")
    print(f"F³ = [{result_woda.dse.D:.4f}, {result_woda.dse.S:.4f}, {result_woda.dse.E:.4f}]")
    print(f"Confidence: D={result_woda.dse.confidence_D:.3f}, "
          f"S={result_woda.dse.confidence_S:.3f}, "
          f"E={result_woda.dse.confidence_E:.3f}")
    
    attractor, dist = classify_to_attractor(result_woda.dse)
    print(f"Najbliższy atraktor: {attractor} (odległość: {dist:.4f})")
    
    print(f"\nPełny wektor Φ⁹:")
    phi9_vec = result_woda.to_array()
    print(f"  [D, S, E, θ_D, φ_D, θ_S, φ_S, θ_E, φ_E]")
    print(f"  {phi9_vec}")
    
    print("\n--- Morfem: 'rzecz' (oczekiwane: niskie D, niskie S, wysokie E) ---")
    computer_rzecz = GeometricDSEComputer(embeddings_rzecz)
    result_rzecz = computer_rzecz.compute_Phi9(morpheme="rzecz")
    print(f"F³ = [{result_rzecz.dse.D:.4f}, {result_rzecz.dse.S:.4f}, {result_rzecz.dse.E:.4f}]")
    
    attractor, dist = classify_to_attractor(result_rzecz.dse)
    print(f"Najbliższy atraktor: {attractor} (odległość: {dist:.4f})")
    
    print("\n--- Test różnych metod dla D ---")
    for method in DeterminationMethod:
        D, conf = computer_woda.compute_D(method=method)
        print(f"  {method.value}: D={D:.4f}, confidence={conf:.3f}")
    
    print("\n--- Test różnych metod dla S ---")
    for method in StabilityMethod:
        S, conf = computer_woda.compute_S(method=method)
        print(f"  {method.value}: S={S:.4f}, confidence={conf:.3f}")
    
    print("\n--- Test różnych metod dla E ---")
    for method in EntropyMethod:
        E, conf = computer_woda.compute_E(method=method)
        print(f"  {method.value}: E={E:.4f}, confidence={conf:.3f}")
    
    print("\n" + "=" * 70)
    print("Demo zakończone")
    print("=" * 70)