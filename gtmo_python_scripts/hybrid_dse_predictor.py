#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Hybrydowy Predyktor D-S-E: Morfosyntaktyka + Semantyka
=======================================================
Łączy cechy morfosyntaktyczne z embeddingami HerBERT dla lepszej predykcji.

Strategia:
- D: Morfosyntaktyka + Semantyka (D pochodzi głównie z analizy składniowej)
- S+E, E/(S+E): Głównie semantyka (z dodatkowymi cechami kontekstowymi)
- Model ensemble: Łączy wszystkie podmodele
"""

import json
import numpy as np
from pathlib import Path
from typing import Dict, Tuple, List, Optional
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV, RepeatedKFold
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.linear_model import ElasticNet, Ridge
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
import matplotlib.pyplot as plt
import joblib
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')


class HybridDSEPredictor:
    """
    Hybrydowy predyktor łączący:
    1. Cechy morfosyntaktyczne (z analizy Stanza/Morfeusz)
    2. Embeddingi semantyczne (HerBERT 768D)
    3. Cechy strukturalne (depth, ambiguity, etc.)
    """

    def __init__(self,
                 data_dir: str = None,
                 data_dirs: List[str] = None,
                 n_components: int = 15,
                 max_morph_features: Optional[int] = 25,
                 augment_factor: float = 0.3,
                 augmentation_noise: float = 0.01,
                 elasticnet_alphas: Optional[List[float]] = None,
                 elasticnet_l1_ratios: Optional[List[float]] = None,
                 cv_folds: int = 5):
        """
        Args:
            data_dir: Katalog z plikami JSON (pojedynczy)
            data_dirs: Lista katalogów z plikami JSON (wiele dokumentów)
            n_components: Liczba komponentów PCA dla embeddingów
            max_morph_features: Limit najważniejszych cech morfosyntaktycznych
            augment_factor: Procent dodatkowych próbek syntetycznych
            augmentation_noise: Odchylenie standardowe szumu dla augmentacji
            elasticnet_alphas: Niestandardowa siatka alpha (opcjonalnie)
            elasticnet_l1_ratios: Niestandardowa siatka l1_ratio (opcjonalnie)
            cv_folds: Liczba foldów do walidacji krzyżowej
        """
        if data_dirs:
            self.data_dirs = [Path(d) for d in data_dirs]
            self.data_dir = None
        elif data_dir:
            self.data_dir = Path(data_dir)
            self.data_dirs = None
        else:
            raise ValueError("Podaj data_dir lub data_dirs")
            
        self.n_components = n_components
        self.max_morph_features = max_morph_features
        self.augment_factor = augment_factor
        self.augmentation_noise = augmentation_noise
        self.elasticnet_alphas = elasticnet_alphas or [0.001, 0.01, 0.05, 0.1, 0.5, 1.0, 2.0]
        self.elasticnet_l1_ratios = elasticnet_l1_ratios or [0.1, 0.3, 0.5, 0.7, 0.9]
        self.cv_folds = cv_folds
        self.random_state = 42
        self.rng = np.random.default_rng(self.random_state)
        self.selected_morph_indices = None

        # PCA i Scalery
        self.pca = None
        self.scaler_embedding = StandardScaler()
        self.scaler_morphosyntax = StandardScaler()

        # Referencje normalizacyjne D-S-E
        self.max_norm_ref = None
        self.max_dispersion_ref = None

        # Modele
        self.models = {
            'D': None,              # Determination (morfosyntaktyka + semantyka)
            'SE': None,             # S+E (głównie semantyka)
            'E_ratio': None        # E/(S+E) (głównie semantyka)
        }

        # Dane
        self.X_train = {}
        self.X_test = {}
        self.y_train = {}
        self.y_test = {}
    def extract_morphosyntactic_features(self, data: dict) -> np.ndarray:
        """
        Ekstraktuj cechy morfosyntaktyczne z JSON.

        Returns:
            Wektor cech morfosyntaktycznych
        """
        features = []

        # 1. Cechy z 'content'
        content = data.get('content', {})
        features.append(content.get('length', 0))
        features.append(content.get('word_count', 0))

        # Średnia długość słowa
        word_count = content.get('word_count', 1)
        avg_word_len = content.get('length', 0) / max(word_count, 1)
        features.append(avg_word_len)

        # 2. Cechy z 'additional_metrics'
        add_metrics = data.get('additional_metrics', {})

        # Ambiguity
        features.append(add_metrics.get('ambiguity', 0))
        features.append(add_metrics.get('coords_count', 0))
        features.append(add_metrics.get('total_analyses', 0))

        # Przypadki (cases) - rozkład
        cases = add_metrics.get('cases', {})
        total_cases = sum(cases.values()) if cases else 1
        features.append(cases.get('nom', 0) / total_cases)  # nominative ratio
        features.append(cases.get('gen', 0) / total_cases)  # genitive ratio
        features.append(cases.get('acc', 0) / total_cases)  # accusative ratio
        features.append(len(cases))  # liczba różnych przypadków

        # POS tags - rozkład
        pos = add_metrics.get('pos', {})
        total_pos = sum(pos.values()) if pos else 1
        features.append(pos.get('subst', 0) / total_pos)  # noun ratio
        features.append(pos.get('verb', 0) / total_pos)   # verb ratio
        features.append(pos.get('adj', 0) / total_pos)    # adj ratio
        features.append(pos.get('prep', 0) / total_pos)   # preposition ratio
        features.append(len(pos))  # liczba różnych POS tags

        # 3. Depth metrics
        depth_metrics = data.get('depth_metrics', {})
        features.append(depth_metrics.get('max_depth', 0))
        features.append(depth_metrics.get('avg_depth', 0))
        features.append(depth_metrics.get('depth_variance', 0))
        features.append(data.get('depth', 0))

        # 4. Constitutional metrics (definiteness/indefiniteness)
        const_metrics = data.get('constitutional_metrics', {})
        definiteness = const_metrics.get('definiteness', {})
        indefiniteness = const_metrics.get('indefiniteness', {})

        features.append(definiteness.get('value', 0))
        features.append(indefiniteness.get('value', 0))

        # Decomposition of indefiniteness
        decomp = indefiniteness.get('decomposition', {})
        features.append(decomp.get('morphological', {}).get('value', 0))
        features.append(decomp.get('syntactic', {}).get('value', 0))
        features.append(decomp.get('semantic', {}).get('value', 0))

        semantic_acc = const_metrics.get('semantic_accessibility', {})
        features.append(semantic_acc.get('value', 0))

        # 5. Rhetorical analysis
        rhet = data.get('rhetorical_analysis', {})
        features.append(rhet.get('irony_score', 0))
        features.append(rhet.get('paradox_score', 0))
        features.append(rhet.get('structural_divergence', 0))

        pos_anom = rhet.get('pos_anomalies', {})
        features.append(pos_anom.get('adj_ratio', 0))
        features.append(pos_anom.get('verb_ratio', 0))
        features.append(pos_anom.get('anomaly_score', 0))

        # 6. Quantum metrics
        quantum = data.get('quantum_metrics', {})
        features.append(quantum.get('total_coherence', 0))
        features.append(quantum.get('quantum_words', 0))
        features.append(quantum.get('entanglements', 0))

        # 7. Geometric balance/tension
        features.append(data.get('geometric_balance', 0))
        features.append(data.get('geometric_tension', 0))

        return np.array(features, dtype=np.float32)

    def load_data(self) -> Tuple[Dict[str, np.ndarray], Dict[str, np.ndarray]]:
        """
        Wczytaj dane i przygotuj features.

        Returns:
            X: Dict z 'embedding', 'morphosyntax', 'combined'
            y: Dict z wartościami D, S, E, SE, E_ratio
        """
        print("[INFO] Wczytywanie danych...")

        embeddings = []
        morphosyntax_features = []
        d_values = []
        s_values = []
        e_values = []
        se_values = []
        e_ratio_values = []

        # Zbierz pliki z jednego lub wielu katalogów
        if self.data_dirs:
            sentence_files = []
            for data_dir in self.data_dirs:
                files = sorted(data_dir.glob("sentence_*.json"))
                sentence_files.extend(files)
                print(f"   - {data_dir.name}: {len(files)} plików")
        else:
            sentence_files = sorted(self.data_dir.glob("sentence_*.json"))

        print(f"\n   Łącznie znaleziono {len(sentence_files)} plików sentence_*.json")
        skipped = {'no_herbert': 0, 'no_coords': 0, 'wrong_size': 0, 'error': 0}

        for file_path in sentence_files:
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    data = json.load(f)

                # Sprawdź wymagane pola
                if 'herbert_embedding' not in data:
                    skipped['no_herbert'] += 1
                    continue
                if 'coordinates' not in data:
                    skipped['no_coords'] += 1
                    continue

                embedding = np.array(data['herbert_embedding'])
                coords = data['coordinates']

                if len(embedding) != 768:
                    skipped['wrong_size'] += 1
                    continue

                # Ekstraktuj cechy morfosyntaktyczne
                morph_features = self.extract_morphosyntactic_features(data)

                embeddings.append(embedding)
                morphosyntax_features.append(morph_features)

                d = coords['determination']
                s = coords['stability']
                e = coords['entropy']

                d_values.append(d)
                s_values.append(s)
                e_values.append(e)
                se_values.append(s + e)

                # E/(S+E) ratio
                se_sum = s + e
                e_ratio = e / se_sum if se_sum > 0 else 0.5
                e_ratio_values.append(e_ratio)

            except Exception as ex:
                skipped['error'] += 1
                continue

        print(f"[OK] Wczytano {len(embeddings)} probek")
        if sum(skipped.values()) > 0:
            print(f"[INFO] Pominięto: {skipped}")

        X_embedding = np.array(embeddings)
        X_morphosyntax = np.array(morphosyntax_features)

        y = {
            'D': np.array(d_values),
            'S': np.array(s_values),
            'E': np.array(e_values),
            'SE': np.array(se_values),
            'E_ratio': np.array(e_ratio_values)
        }

        # Statystyki
        print("\n[STATS] Statystyki:")
        print(f"   Embeddings shape: {X_embedding.shape}")
        print(f"   Morphosyntax shape: {X_morphosyntax.shape}")
        for metric, values in y.items():
            print(f"   {metric:10s}: mean={np.mean(values):.4f}, "
                  f"std={np.std(values):.4f}, range=[{np.min(values):.4f}, {np.max(values):.4f}]")

        # Weryfikacja S+E ≈ 1
        print("\n[VERIFY] Weryfikacja S+E:")
        se_issues = 0
        for i, (s, e) in enumerate(zip(s_values, e_values)):
            se_sum = s + e
            if abs(se_sum - 1.0) > 0.01:
                se_issues += 1
                if se_issues <= 5:  # Pokaż max 5 przykładów
                    print(f"   [WARN] Próbka {i}: S+E = {se_sum:.4f} (oczekiwano ≈1.0)")
        if se_issues == 0:
            print("   ✓ Wszystkie próbki: S+E ≈ 1.0")
        elif se_issues > 5:
            print(f"   [WARN] Łącznie {se_issues} próbek z S+E ≠ 1.0")

        # Oblicz referencje normalizacyjne
        self.max_norm_ref = np.linalg.norm(X_embedding, axis=1).max()
        centroid = X_embedding.mean(axis=0)
        distances = np.linalg.norm(X_embedding - centroid, axis=1)
        self.max_dispersion_ref = distances.mean()

        print(f"\n[REFS] Referencje normalizacyjne:")
        print(f"   max_norm_ref = {self.max_norm_ref:.4f}")
        print(f"   max_dispersion_ref = {self.max_dispersion_ref:.4f}")

        X = {
            'embedding': X_embedding,
            'morphosyntax': X_morphosyntax
        }

        return X, y

    def reduce_morph_features(self, X_morph_scaled: np.ndarray) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        """
        Wybierz najważniejsze cechy morfosyntaktyczne na podstawie wariancji.
        """
        if (self.max_morph_features is None or
                self.max_morph_features >= X_morph_scaled.shape[1]):
            print("   3b. Redukcja cech morfosyntaktycznych: pominięta")
            return X_morph_scaled, None

        variances = np.var(X_morph_scaled, axis=0)
        top_indices = np.argsort(variances)[::-1][:self.max_morph_features]
        top_indices = np.sort(top_indices)
        print(f"   3b. Wybrano {len(top_indices)}/{X_morph_scaled.shape[1]} cech morfosyntaktycznych")
        return X_morph_scaled[:, top_indices], top_indices

    def recalculate_DSE_with_fixed_refs(self, embeddings: np.ndarray) -> Tuple[float, float, float]:
        """
        Przelicz D-S-E używając stałych referencji z treningu.
        
        Args:
            embeddings: Macierz embeddingów (n_samples, embedding_dim)
            
        Returns:
            Tuple (D, S, E) znormalizowane przez referencje
        """
        if self.max_norm_ref is None or self.max_dispersion_ref is None:
            raise ValueError("Referencje normalizacyjne nie zostały obliczone. Najpierw wywołaj load_data().")
        
        # D: normalizacja przez max_norm_ref
        norms = np.linalg.norm(embeddings, axis=1)
        D = norms.mean() / self.max_norm_ref
        
        # E: normalizacja przez max_dispersion_ref
        centroid = embeddings.mean(axis=0)
        distances = np.linalg.norm(embeddings - centroid, axis=1)
        E_raw = distances.mean()
        E = E_raw / self.max_dispersion_ref
        
        # S = 1 - E (gwarantuje S+E=1)
        S = 1 - E
        
        return D, S, E

    def apply_augmentation(self,
                           X_for_D: np.ndarray,
                           X_for_SE: np.ndarray,
                           X_for_ratio: np.ndarray,
                           y: Dict[str, np.ndarray]) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Prosta augmentacja danych przez duplikację i dodanie szumu.
        """
        if self.augment_factor <= 0:
            print("   5. Augmentacja danych: pominięta (factor <= 0)")
            return X_for_D, X_for_SE, X_for_ratio

        n_samples = X_for_D.shape[0]
        n_aug = max(1, int(n_samples * self.augment_factor))
        print(f"   5. Augmentacja danych (~{n_aug} dodatkowych próbek)...")

        indices = self.rng.integers(0, n_samples, size=n_aug)

        def augment_matrix(matrix: np.ndarray) -> np.ndarray:
            noise = self.rng.normal(0, self.augmentation_noise, size=(n_aug, matrix.shape[1]))
            return np.vstack([matrix, matrix[indices] + noise])

        X_for_D = augment_matrix(X_for_D)
        X_for_SE = augment_matrix(X_for_SE)
        X_for_ratio = augment_matrix(X_for_ratio)

        for metric in ['D', 'SE', 'E_ratio', 'S', 'E']:
            noise = self.rng.normal(0, self.augmentation_noise, size=n_aug)
            augmented = np.concatenate([y[metric], y[metric][indices] + noise])
            y[metric] = np.clip(augmented, 0, 2)

        return X_for_D, X_for_SE, X_for_ratio

    def train_elastic_net(self, feature_key: str, target_key: str, label: str) -> Dict:
        """
        Trenuj i dostrajaj ElasticNet dla wskazanego celu.
        """
        print(f"\n[{label}] ElasticNet + GridSearch (CV={self.cv_folds})")
        param_grid = {
            'alpha': self.elasticnet_alphas,
            'l1_ratio': self.elasticnet_l1_ratios
        }

        base_model = ElasticNet(max_iter=5000, random_state=self.random_state)
        grid = GridSearchCV(
            base_model,
            param_grid,
            cv=self.cv_folds,
            scoring='r2',
            n_jobs=-1
        )
        grid.fit(self.X_train[feature_key], self.y_train[target_key])
        print(f"   Najlepsze parametry: {grid.best_params_} (R2 CV = {grid.best_score_:.4f})")

        return {
            'model': grid.best_estimator_,
            'name': 'ElasticNet',
            'cv_score': grid.best_score_,
            'best_params': grid.best_params_
        }

    def prepare_data(self, X: Dict[str, np.ndarray], y: Dict[str, np.ndarray],
                     test_size: float = 0.2):
        """
        Przygotuj dane bez wycieku:
        - Najpierw split
        - Fit transformery (StandardScaler, PCA, selekcja wariancyjna) tylko na train
        - Augmentacja tylko na train
        """
        print(f"\n[PREP] Przygotowanie danych...")

        # 0. Split surowych danych
        print(f"   0. Podzial danych: {int((1-test_size)*100)}% train / {int(test_size*100)}% test")
        (emb_train, emb_test,
         morph_train, morph_test,
         y_train_D, y_test_D,
         y_train_SE, y_test_SE,
         y_train_ratio, y_test_ratio,
         y_train_S, y_test_S,
         y_train_E, y_test_E) = train_test_split(
            X['embedding'],
            X['morphosyntax'],
            y['D'],
            y['SE'],
            y['E_ratio'],
            y['S'],
            y['E'],
            test_size=test_size,
            random_state=42
        )

        # 1. Skalowanie embeddingow (fit na train)
        print(f"   1. Skalowanie embeddingow (train)...")
        emb_train_scaled = self.scaler_embedding.fit_transform(emb_train)
        emb_test_scaled = self.scaler_embedding.transform(emb_test)

        # 2. PCA (fit na train)
        print(f"   2. PCA: 768D -> {self.n_components}D (fit na train)...")
        self.pca = PCA(n_components=self.n_components, random_state=42)
        emb_train_reduced = self.pca.fit_transform(emb_train_scaled)
        emb_test_reduced = self.pca.transform(emb_test_scaled)
        explained_var = np.sum(self.pca.explained_variance_ratio_)
        print(f"      Wyjasniona wariancja: {explained_var:.2%}")

        # 3. Skalowanie morfosyntaktyki (fit na train)
        print(f"   3. Skalowanie cech morfosyntaktycznych (train)...")
        morph_train_scaled = self.scaler_morphosyntax.fit_transform(morph_train)
        morph_test_scaled = self.scaler_morphosyntax.transform(morph_test)

        # 3b. Redukcja cech po wariancji (fit na train), zastosuj na test
        morph_train_reduced, selected_indices = self.reduce_morph_features(morph_train_scaled)
        self.selected_morph_indices = selected_indices
        if selected_indices is not None:
            morph_test_reduced = morph_test_scaled[:, selected_indices]
        else:
            morph_test_reduced = morph_test_scaled

        # 4. Laczenie features (bez twardych indeksow – uzyj wszystkich wybranych cech)
        # D: użyj wyłącznie pełnych cech morfosyntaktycznych (36)
        X_train_D = morph_train_scaled
        X_test_D  = morph_test_scaled

        X_train_SE = np.hstack([emb_train_reduced, morph_train_reduced])
        X_test_SE  = np.hstack([emb_test_reduced,  morph_test_reduced])
        X_train_ratio = X_train_SE.copy()
        X_test_ratio  = X_test_SE.copy()

        # 5. Augmentacja tylko train
        y_train = {
            'D': y_train_D,
            'SE': y_train_SE,
            'E_ratio': y_train_ratio,
            'S': y_train_S,
            'E': y_train_E
        }
        X_train_D, X_train_SE, X_train_ratio = self.apply_augmentation(
            X_train_D, X_train_SE, X_train_ratio, y_train
        )

        print(f"      X_train_D shape: {X_train_D.shape}")
        print(f"      X_train_SE shape: {X_train_SE.shape}")

        # 6. Zapisy
        self.X_train = {
            'D': X_train_D,
            'SE': X_train_SE,
            'E_ratio': X_train_ratio
        }
        self.X_test = {
            'D': X_test_D,
            'SE': X_test_SE,
            'E_ratio': X_test_ratio
        }
        self.y_train = y_train
        self.y_test = {
            'D': y_test_D,
            'SE': y_test_SE,
            'E_ratio': y_test_ratio,
            'S': y_test_S,
            'E': y_test_E
        }

        print(f"      Train: {self.X_train['D'].shape[0]} probek (po augmentacji)")
        print(f"      Test:  {self.X_test['D'].shape[0]} probek")

    def train_models(self):
        """
        Trenuj modele:
        1. Model dla D (morfosyntaktyka + semantyka)
        2. Model dla S+E (głównie semantyka)
        3. Model dla E/(S+E) ratio
        4. Ensemble model
        """
        print("\n[TRAIN] Trening modeli...")

        cv = RepeatedKFold(n_splits=4, n_repeats=2, random_state=42)

        # 1. Model dla D
        print("\n[1/3] Model dla DETERMINATION (morfosyntaktyka + semantyka):")

        d_models = {
            'GradientBoosting': {
                'model': GradientBoostingRegressor(random_state=42),
                'params': {
                    'n_estimators': [50, 100, 150],
                    'learning_rate': [0.01, 0.05, 0.1],
                    'max_depth': [2, 3],
                    'subsample': [0.7, 0.85],
                    'min_samples_leaf': [3, 5, 7]
                }
            },
            'RandomForest': {
                'model': RandomForestRegressor(random_state=42),
                'params': {
                    'n_estimators': [100, 200],
                    'max_depth': [3, 4, 5],
                    'min_samples_split': [3, 5],
                    'min_samples_leaf': [2, 4]
                }
            }
        }

        best_d_score = -np.inf
        best_d_model = None
        best_d_name = None

        for model_name, config in d_models.items():
            print(f"   Testowanie: {model_name}...", end=" ")

            grid = GridSearchCV(
                config['model'],
                config['params'],
                cv=cv,
                scoring='r2',
                n_jobs=-1,
                verbose=1
            )

            grid.fit(self.X_train['D'], self.y_train['D'])
            score = grid.best_score_
            print(f"R2 CV = {score:.4f}")

            if score > best_d_score:
                best_d_score = score
                best_d_model = grid.best_estimator_
                best_d_name = model_name

        self.models['D'] = {
            'model': best_d_model,
            'name': best_d_name,
            'cv_score': best_d_score
        }
        print(f"   >> NAJLEPSZY: {best_d_name} (R2 CV = {best_d_score:.4f})")

        # 2. Model dla S+E
        print("\n[2/3] Model dla S+E (glownie semantyka):")

        se_models = {
            'Ridge': {
                'model': Ridge(),
                'params': {'alpha': [0.1, 1.0, 10.0]}
            },
            'ElasticNet': {
                'model': ElasticNet(),
                'params': {
                    'alpha': [0.01, 0.1, 1.0],
                    'l1_ratio': [0.3, 0.5, 0.7]
                }
            },
            'GradientBoosting': {
                'model': GradientBoostingRegressor(random_state=42),
                'params': {
                    'n_estimators': [30, 60],
                    'learning_rate': [0.05, 0.1],
                    'max_depth': [2, 3]
                }
            }
        }

        best_se_score = -np.inf
        best_se_model = None
        best_se_name = None

        for model_name, config in se_models.items():
            print(f"   Testowanie: {model_name}...", end=" ")

            grid = GridSearchCV(
                config['model'],
                config['params'],
                cv=cv,
                scoring='r2',
                n_jobs=-1
            )

            grid.fit(self.X_train['SE'], self.y_train['SE'])
            score = grid.best_score_
            print(f"R2 CV = {score:.4f}")

            if score > best_se_score:
                best_se_score = score
                best_se_model = grid.best_estimator_
                best_se_name = model_name

        self.models['SE'] = {
            'model': best_se_model,
            'name': best_se_name,
            'cv_score': best_se_score
        }
        print(f"   >> NAJLEPSZY: {best_se_name} (R2 CV = {best_se_score:.4f})")

        # 3. Model dla E/(S+E) ratio
        print("\n[3/3] Model dla E/(S+E) RATIO:")

        best_ratio_score = -np.inf
        best_ratio_model = None
        best_ratio_name = None

        for model_name, config in se_models.items():  # Te same modele co dla SE
            print(f"   Testowanie: {model_name}...", end=" ")

            grid = GridSearchCV(
                config['model'],
                config['params'],
                cv=cv,
                scoring='r2',
                n_jobs=-1
            )

            grid.fit(self.X_train['E_ratio'], self.y_train['E_ratio'])
            score = grid.best_score_
            print(f"R2 CV = {score:.4f}")

            if score > best_ratio_score:
                best_ratio_score = score
                best_ratio_model = grid.best_estimator_
                best_ratio_name = model_name

        self.models['E_ratio'] = {
            'model': best_ratio_model,
            'name': best_ratio_name,
            'cv_score': best_ratio_score
        }
        print(f"   >> NAJLEPSZY: {best_ratio_name} (R2 CV = {best_ratio_score:.4f})")

    def reconstruct_SE(self, D_pred: np.ndarray, SE_pred: np.ndarray,
                       E_ratio_pred: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Rekonstruuj S i E z predykcji S+E i E/(S+E).
        """
        E_pred = SE_pred * E_ratio_pred
        S_pred = SE_pred - E_pred

        # Clipping
        E_pred = np.clip(E_pred, 0, 1)
        S_pred = np.clip(S_pred, 0, 2)  # S może być > 1 w modelu płynnym

        return S_pred, E_pred

    def evaluate_models(self) -> Dict[str, Dict[str, float]]:
        """
        Ewaluacja modeli na zbiorze testowym.
        """
        print("\n[EVAL] Ewaluacja modeli:")

        results = {}

        # 1. Predykcje
        D_pred = self.models['D']['model'].predict(self.X_test['D'])
        SE_pred = self.models['SE']['model'].predict(self.X_test['SE'])
        E_ratio_pred = self.models['E_ratio']['model'].predict(self.X_test['E_ratio'])

        # 2. Rekonstrukcja S i E
        S_pred, E_pred = self.reconstruct_SE(D_pred, SE_pred, E_ratio_pred)

        # 3. Metryki dla D
        y_true_D = self.y_test['D']
        mse_D = mean_squared_error(y_true_D, D_pred)
        rmse_D = np.sqrt(mse_D)
        mae_D = mean_absolute_error(y_true_D, D_pred)
        r2_D = r2_score(y_true_D, D_pred)

        results['D'] = {
            'model': self.models['D']['name'],
            'RMSE': rmse_D,
            'MAE': mae_D,
            'R2': r2_D,
            'CV_R2': self.models['D']['cv_score']
        }

        print(f"\n   DETERMINATION ({self.models['D']['name']}):")
        print(f"      RMSE:   {rmse_D:.4f}")
        print(f"      MAE:    {mae_D:.4f}")
        print(f"      R2:     {r2_D:.4f}")
        print(f"      CV R2:  {self.models['D']['cv_score']:.4f}")

        # 4. Metryki dla S
        y_true_S = self.y_test['S']
        rmse_S = np.sqrt(mean_squared_error(y_true_S, S_pred))
        mae_S = mean_absolute_error(y_true_S, S_pred)
        r2_S = r2_score(y_true_S, S_pred)

        results['S'] = {
            'model': 'Reconstructed',
            'RMSE': rmse_S,
            'MAE': mae_S,
            'R2': r2_S
        }

        print(f"\n   STABILITY (rekonstruowane):")
        print(f"      RMSE:   {rmse_S:.4f}")
        print(f"      MAE:    {mae_S:.4f}")
        print(f"      R2:     {r2_S:.4f}")

        # 5. Metryki dla E
        y_true_E = self.y_test['E']
        rmse_E = np.sqrt(mean_squared_error(y_true_E, E_pred))
        mae_E = mean_absolute_error(y_true_E, E_pred)
        r2_E = r2_score(y_true_E, E_pred)

        results['E'] = {
            'model': 'Reconstructed',
            'RMSE': rmse_E,
            'MAE': mae_E,
            'R2': r2_E
        }

        print(f"\n   ENTROPY (rekonstruowane):")
        print(f"      RMSE:   {rmse_E:.4f}")
        print(f"      MAE:    {mae_E:.4f}")
        print(f"      R2:     {r2_E:.4f}")

        # 6. Metryki dla S+E
        y_true_SE = self.y_test['SE']
        rmse_SE = np.sqrt(mean_squared_error(y_true_SE, SE_pred))
        mae_SE = mean_absolute_error(y_true_SE, SE_pred)
        r2_SE = r2_score(y_true_SE, SE_pred)

        results['SE'] = {
            'model': self.models['SE']['name'],
            'RMSE': rmse_SE,
            'MAE': mae_SE,
            'R2': r2_SE,
            'CV_R2': self.models['SE']['cv_score']
        }

        print(f"\n   S+E ({self.models['SE']['name']}):")
        print(f"      RMSE:   {rmse_SE:.4f}")
        print(f"      MAE:    {mae_SE:.4f}")
        print(f"      R2:     {r2_SE:.4f}")
        print(f"      CV R2:  {self.models['SE']['cv_score']:.4f}")

        # 7. Metryki dla E/(S+E) ratio
        y_true_ratio = self.y_test['E_ratio']
        rmse_ratio = np.sqrt(mean_squared_error(y_true_ratio, E_ratio_pred))
        mae_ratio = mean_absolute_error(y_true_ratio, E_ratio_pred)
        r2_ratio = r2_score(y_true_ratio, E_ratio_pred)

        results['E_ratio'] = {
            'model': self.models['E_ratio']['name'],
            'RMSE': rmse_ratio,
            'MAE': mae_ratio,
            'R2': r2_ratio,
            'CV_R2': self.models['E_ratio']['cv_score']
        }

        print(f"\n   E/(S+E) RATIO ({self.models['E_ratio']['name']}):")
        print(f"      RMSE:   {rmse_ratio:.4f}")
        print(f"      MAE:    {mae_ratio:.4f}")
        print(f"      R2:     {r2_ratio:.4f}")
        print(f"      CV R2:  {self.models['E_ratio']['cv_score']:.4f}")

        # 8. Weryfikacja rekonstrukcji
        SE_reconstructed = S_pred + E_pred
        SE_error = np.mean(np.abs(SE_reconstructed - SE_pred))
        print(f"\n   WERYFIKACJA REKONSTRUKCJI S+E:")
        print(f"      Sredni blad: {SE_error:.6f}")

        return results

    def plot_results(self, save_path: str = None):
        """
        Wizualizacja wyników.
        """
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))

        # Predykcje
        D_pred = self.models['D']['model'].predict(self.X_test['D'])
        SE_pred = self.models['SE']['model'].predict(self.X_test['SE'])
        E_ratio_pred = self.models['E_ratio']['model'].predict(self.X_test['E_ratio'])
        S_pred, E_pred = self.reconstruct_SE(D_pred, SE_pred, E_ratio_pred)

        predictions = {
            'D': D_pred,
            'S': S_pred,
            'E': E_pred,
            'SE': SE_pred,
            'E_ratio': E_ratio_pred
        }

        targets = ['D', 'S', 'E', 'SE', 'E_ratio']

        for idx, metric in enumerate(targets):
            row = idx // 3
            col = idx % 3
            ax = axes[row, col]

            y_pred = predictions[metric]
            y_true = self.y_test[metric]

            ax.scatter(y_true, y_pred, alpha=0.6, edgecolors='k', linewidth=0.5)

            min_val = min(y_true.min(), y_pred.min())
            max_val = max(y_true.max(), y_pred.max())
            ax.plot([min_val, max_val], [min_val, max_val], 'r--', lw=2, label='Ideal')

            r2 = r2_score(y_true, y_pred)
            rmse = np.sqrt(mean_squared_error(y_true, y_pred))

            if metric in self.models and self.models[metric]:
                model_name = self.models[metric]['name']
            else:
                model_name = 'Reconstructed'

            ax.set_xlabel(f'Rzeczywiste {metric}', fontsize=12)
            ax.set_ylabel(f'Przewidywane {metric}', fontsize=12)
            ax.set_title(f'{metric} ({model_name})\nR2={r2:.3f}, RMSE={rmse:.3f}', fontsize=12)
            ax.legend()
            ax.grid(alpha=0.3)

        # Usuń ostatni (pusty) subplot
        fig.delaxes(axes[1, 2])

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"[PLOT] Wykres zapisano: {save_path}")

        plt.show()

    def save_models(self, output_dir: str):
        """
        Zapisz modele i transformery.
        """
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        # Zapisz transformery
        joblib.dump(self.pca, output_path / 'pca.pkl')
        joblib.dump(self.scaler_embedding, output_path / 'scaler_embedding.pkl')
        joblib.dump(self.scaler_morphosyntax, output_path / 'scaler_morphosyntax.pkl')

        # Zapisz modele
        for metric in ['D', 'SE', 'E_ratio']:
            model_info = self.models[metric]
            model_path = output_path / f'{metric}_model.pkl'
            joblib.dump(model_info, model_path)
            print(f"[SAVE] {metric}: {model_info['name']} -> {model_path}")

        # Metadata
        metadata = {
            'timestamp': datetime.now().isoformat(),
            'n_components': self.n_components,
            'n_samples_train': self.X_train['D'].shape[0],
            'n_samples_test': self.X_test['D'].shape[0],
            'models': {k: v['name'] for k, v in self.models.items() if v is not None},
            'normalization_refs': {
                'max_norm_ref': float(self.max_norm_ref) if self.max_norm_ref is not None else None,
                'max_dispersion_ref': float(self.max_dispersion_ref) if self.max_dispersion_ref is not None else None
            }
        }

        with open(output_path / 'metadata.json', 'w', encoding='utf-8') as f:
            json.dump(metadata, f, indent=2, ensure_ascii=False)

        print(f"[SAVE] Metadata zapisana")


def main():
    print("=" * 80)
    print("  HYBRYDOWY PREDYKTOR D-S-E: Morfosyntaktyka + Semantyka")
    print("  AUTO-SKANOWANIE gtmo_results")
    print("=" * 80)

    # Automatyczne skanowanie wszystkich katalogów w gtmo_results
    results_root = Path(r"D:\GTMO_MORPHOSYNTAX\gtmo_results")
    
    # Znajdź wszystkie podkatalogi z analizami
    all_analysis_dirs = [d for d in results_root.iterdir() if d.is_dir() and d.name.startswith('analysis_')]
    
    print(f"\n[INFO] Znaleziono {len(all_analysis_dirs)} katalogów analizy")
    
    # Sprawdź, które mają pliki z herbert_embedding
    valid_dirs = []
    dir_stats = {}
    
    for analysis_dir in all_analysis_dirs:
        sentence_files = list(analysis_dir.glob("sentence_*.json"))
        if not sentence_files:
            continue
            
        # Sprawdź pierwszy plik, czy ma herbert_embedding
        has_herbert = False
        try:
            with open(sentence_files[0], 'r', encoding='utf-8') as f:
                data = json.load(f)
                if 'herbert_embedding' in data and len(data.get('herbert_embedding', [])) == 768:
                    has_herbert = True
        except:
            pass
        
        dir_stats[analysis_dir.name] = {
            'path': analysis_dir,
            'files': len(sentence_files),
            'has_herbert': has_herbert
        }
        
        if has_herbert:
            valid_dirs.append(analysis_dir)
    
    print(f"\n[KATALOGI] Status embeddingów HerBERT:")
    for name, stats in sorted(dir_stats.items()):
        status = "✓ HerBERT" if stats['has_herbert'] else "✗ brak HerBERT"
        print(f"  {status:15} {name}: {stats['files']} plików")
    
    if not valid_dirs:
        print("\n[ERROR] Nie znaleziono katalogów z embeddingami HerBERT!")
        print("[HINT] Uruchom najpierw gtmo_morphosyntax.py lub gtmo_general_text.py")
        return
    
    print(f"\n[INFO] Użyto {len(valid_dirs)} katalogów z embeddingami HerBERT")

    # Inicjalizuj predyktor z katalogami mającymi embeddingi
    predictor = HybridDSEPredictor(
        data_dirs=[str(d) for d in valid_dirs if d.exists()],
        n_components=20,
        augment_factor=0.3  # Standardowa augmentacja
    )

    # 1. Wczytaj dane
    X, y = predictor.load_data()

    # 2. Przygotuj
    predictor.prepare_data(X, y, test_size=0.2)

    # 3. Trenuj
    predictor.train_models()

    # 4. Ewaluuj
    results = predictor.evaluate_models()

    # 5. Wizualizacja
    print("\n[VIZ] Generowanie wykresow...")
    predictor.plot_results(save_path='hybrid_predictions_multi.png')

    # 6. Zapisz modele
    predictor.save_models('hybrid_models_multi')

    print("\n" + "=" * 80)
    print("[SUCCESS] ZAKONCZONO!")
    print("=" * 80)


if __name__ == "__main__":
    main()
