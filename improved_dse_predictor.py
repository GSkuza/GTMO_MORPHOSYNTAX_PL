#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Ulepszona wersja predyktora D-S-E z redukcją wymiarowości i regularyzacją
==========================================================================
"""

import json
import numpy as np
from pathlib import Path
from typing import Dict, Tuple
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.linear_model import Ridge, Lasso
import matplotlib.pyplot as plt
import joblib
from datetime import datetime


class ImprovedDSEPredictor:
    """
    Ulepszona wersja predyktora z:
    - Redukcją wymiarowości (PCA)
    - Wieloma algorytmami (GB, RF, Ridge, Lasso)
    - Grid Search dla hiperparametrów
    - Lepszą walidacją
    """

    def __init__(self, data_dir: str, n_components: int = 20):
        """
        Args:
            data_dir: Katalog z danymi
            n_components: Liczba komponentów PCA (domyślnie 20)
        """
        self.data_dir = Path(data_dir)
        self.n_components = n_components

        self.pca = None
        self.scaler = StandardScaler()
        self.models = {}
        self.best_models = {}

        self.X_train = None
        self.X_test = None
        self.y_train = {}
        self.y_test = {}

    def load_data(self) -> Tuple[np.ndarray, Dict[str, np.ndarray]]:
        """Wczytaj dane z plików JSON."""
        print("[INFO] Wczytywanie danych...")

        embeddings = []
        d_values = []
        s_values = []
        e_values = []

        sentence_files = sorted(self.data_dir.glob("sentence_*.json"))

        for file_path in sentence_files:
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    data = json.load(f)

                if 'herbert_embedding' not in data or 'coordinates' not in data:
                    continue

                embedding = np.array(data['herbert_embedding'])
                coords = data['coordinates']

                if len(embedding) != 768:
                    continue

                embeddings.append(embedding)
                d_values.append(coords['determination'])
                s_values.append(coords['stability'])
                e_values.append(coords['entropy'])

            except Exception as e:
                continue

        print(f"[OK] Wczytano {len(embeddings)} probek")

        X = np.array(embeddings)
        y = {
            'determination': np.array(d_values),
            'stability': np.array(s_values),
            'entropy': np.array(e_values)
        }

        # Statystyki
        print("\n[STATS] Statystyki:")
        for metric, values in y.items():
            print(f"   {metric.upper():15s}: mean={np.mean(values):.4f}, "
                  f"std={np.std(values):.4f}, range=[{np.min(values):.4f}, {np.max(values):.4f}]")

        return X, y

    def prepare_data(self, X: np.ndarray, y: Dict[str, np.ndarray], test_size: float = 0.2):
        """
        Przygotuj dane z:
        - Skalowaniem
        - Redukcją wymiarowości (PCA)
        - Podziałem train/test
        """
        print(f"\n[PREP] Przygotowanie danych...")

        # 1. Skalowanie
        print(f"   1. Skalowanie embeddingów (768D)...")
        X_scaled = self.scaler.fit_transform(X)

        # 2. Redukcja wymiarowości PCA
        print(f"   2. Redukcja wymiarowości: 768D -> {self.n_components}D (PCA)...")
        self.pca = PCA(n_components=self.n_components, random_state=42)
        X_reduced = self.pca.fit_transform(X_scaled)

        explained_var = np.sum(self.pca.explained_variance_ratio_)
        print(f"      Wyjasniona wariancja: {explained_var:.2%}")

        # 3. Split
        print(f"   3. Podzial danych: {int((1-test_size)*100)}% train / {int(test_size)*100}% test")
        self.X_train, self.X_test = train_test_split(
            X_reduced, test_size=test_size, random_state=42
        )

        for metric in ['determination', 'stability', 'entropy']:
            y_train, y_test = train_test_split(
                y[metric], test_size=test_size, random_state=42
            )
            self.y_train[metric] = y_train
            self.y_test[metric] = y_test

        print(f"      Train: {self.X_train.shape[0]} probek")
        print(f"      Test:  {self.X_test.shape[0]} probek")

    def train_multiple_models(self):
        """
        Trenuj wiele modeli i wybierz najlepszy dla każdej metryki.
        """
        print("\n[TRAIN] Trening wielu modeli...")

        # Definicje modeli do przetestowania
        model_configs = {
            'Ridge': {
                'model': Ridge(),
                'params': {'alpha': [0.1, 1.0, 10.0, 100.0]}
            },
            'Lasso': {
                'model': Lasso(),
                'params': {'alpha': [0.001, 0.01, 0.1, 1.0]}
            },
            'GradientBoosting': {
                'model': GradientBoostingRegressor(random_state=42),
                'params': {
                    'n_estimators': [50, 100],
                    'learning_rate': [0.05, 0.1],
                    'max_depth': [3, 5],
                    'subsample': [0.8]
                }
            },
            'RandomForest': {
                'model': RandomForestRegressor(random_state=42),
                'params': {
                    'n_estimators': [50, 100],
                    'max_depth': [5, 10, None],
                    'min_samples_split': [2, 5]
                }
            }
        }

        for metric in ['determination', 'stability', 'entropy']:
            print(f"\n[{metric.upper()}]")

            best_score = -np.inf
            best_model_name = None
            best_model = None

            for model_name, config in model_configs.items():
                print(f"   Testowanie: {model_name}...", end=" ")

                # Grid Search CV
                grid = GridSearchCV(
                    config['model'],
                    config['params'],
                    cv=5,
                    scoring='r2',
                    n_jobs=-1
                )

                grid.fit(self.X_train, self.y_train[metric])

                score = grid.best_score_
                print(f"R2 CV = {score:.4f}")

                if score > best_score:
                    best_score = score
                    best_model_name = model_name
                    best_model = grid.best_estimator_

            self.best_models[metric] = {
                'name': best_model_name,
                'model': best_model,
                'cv_score': best_score
            }

            print(f"   >> NAJLEPSZY: {best_model_name} (R2 CV = {best_score:.4f})")

    def evaluate_models(self) -> Dict[str, Dict[str, float]]:
        """Ewaluacja najlepszych modeli."""
        print("\n[EVAL] Ewaluacja na zbiorze testowym:")

        results = {}

        for metric in ['determination', 'stability', 'entropy']:
            model_info = self.best_models[metric]
            model = model_info['model']
            model_name = model_info['name']

            y_pred = model.predict(self.X_test)
            y_true = self.y_test[metric]

            mse = mean_squared_error(y_true, y_pred)
            rmse = np.sqrt(mse)
            mae = mean_absolute_error(y_true, y_pred)
            r2 = r2_score(y_true, y_pred)

            results[metric] = {
                'model': model_name,
                'MSE': mse,
                'RMSE': rmse,
                'MAE': mae,
                'R2': r2,
                'CV_R2': model_info['cv_score']
            }

            print(f"\n   {metric.upper()} ({model_name}):")
            print(f"      RMSE:   {rmse:.4f}")
            print(f"      MAE:    {mae:.4f}")
            print(f"      R2:     {r2:.4f}")
            print(f"      CV R2:  {model_info['cv_score']:.4f}")

        return results

    def plot_results(self, save_path: str = None):
        """Wizualizacja wyników."""
        fig, axes = plt.subplots(1, 3, figsize=(18, 5))

        for i, metric in enumerate(['determination', 'stability', 'entropy']):
            ax = axes[i]

            model = self.best_models[metric]['model']
            model_name = self.best_models[metric]['name']

            y_pred = model.predict(self.X_test)
            y_true = self.y_test[metric]

            ax.scatter(y_true, y_pred, alpha=0.6, edgecolors='k', linewidth=0.5)

            min_val = min(y_true.min(), y_pred.min())
            max_val = max(y_true.max(), y_pred.max())
            ax.plot([min_val, max_val], [min_val, max_val], 'r--', lw=2, label='Ideal')

            r2 = r2_score(y_true, y_pred)
            rmse = np.sqrt(mean_squared_error(y_true, y_pred))

            ax.set_xlabel(f'Rzeczywiste {metric}', fontsize=12)
            ax.set_ylabel(f'Przewidywane {metric}', fontsize=12)
            ax.set_title(f'{metric.upper()} ({model_name})\nR2={r2:.3f}, RMSE={rmse:.3f}', fontsize=12)
            ax.legend()
            ax.grid(alpha=0.3)

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"[PLOT] Wykres zapisano: {save_path}")

        plt.show()

    def analyze_pca_components(self, save_path: str = None):
        """Analiza komponentów PCA."""
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

        # 1. Explained variance
        ax1.bar(range(1, self.n_components + 1),
                self.pca.explained_variance_ratio_)
        ax1.set_xlabel('Komponent PCA', fontsize=12)
        ax1.set_ylabel('Wyjasniona wariancja', fontsize=12)
        ax1.set_title('Wyjasniona wariancja przez komponenty PCA', fontsize=14)
        ax1.grid(alpha=0.3)

        # 2. Cumulative variance
        cumsum = np.cumsum(self.pca.explained_variance_ratio_)
        ax2.plot(range(1, self.n_components + 1), cumsum, 'b-', linewidth=2)
        ax2.axhline(y=0.95, color='r', linestyle='--', label='95%')
        ax2.set_xlabel('Liczba komponentow', fontsize=12)
        ax2.set_ylabel('Skumulowana wariancja', fontsize=12)
        ax2.set_title('Skumulowana wyjasniona wariancja', fontsize=14)
        ax2.legend()
        ax2.grid(alpha=0.3)

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"[PLOT] Analiza PCA zapisana: {save_path}")

        plt.show()

    def save_models(self, output_dir: str):
        """Zapisz modele."""
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        # Zapisz PCA i scaler
        joblib.dump(self.pca, output_path / 'pca.pkl')
        joblib.dump(self.scaler, output_path / 'scaler.pkl')

        # Zapisz modele
        for metric in ['determination', 'stability', 'entropy']:
            model_info = self.best_models[metric]
            model_path = output_path / f'{metric}_model.pkl'
            joblib.dump(model_info, model_path)
            print(f"[SAVE] {metric}: {model_info['name']} -> {model_path}")

        # Metadata
        metadata = {
            'timestamp': datetime.now().isoformat(),
            'n_components': self.n_components,
            'n_samples_train': self.X_train.shape[0],
            'n_samples_test': self.X_test.shape[0],
            'models': {k: v['name'] for k, v in self.best_models.items()}
        }

        with open(output_path / 'metadata.json', 'w', encoding='utf-8') as f:
            json.dump(metadata, f, indent=2, ensure_ascii=False)

        print(f"[SAVE] Metadata zapisana")

    def predict(self, herbert_embedding: np.ndarray) -> Dict[str, float]:
        """Predykcja dla nowego embeddingu."""
        if herbert_embedding.shape != (768,):
            raise ValueError(f"Embedding musi miec 768 wymiarow")

        # Transform
        X = herbert_embedding.reshape(1, -1)
        X_scaled = self.scaler.transform(X)
        X_reduced = self.pca.transform(X_scaled)

        predictions = {}
        for metric in ['determination', 'stability', 'entropy']:
            model = self.best_models[metric]['model']
            pred = model.predict(X_reduced)[0]
            predictions[metric] = np.clip(pred, 0.0, 1.0)

        return predictions


def main():
    print("=" * 80)
    print("  ULEPSZONA PREDYKCJA D-S-E - PCA + Multi-Model")
    print("=" * 80)

    data_dir = r"D:\GTMO_MORPHOSYNTAX\gtmo_results\analysis_15112025_no1_projekt_poselski_edited"

    # Test różnych wartości n_components
    for n_comp in [10, 20, 30]:
        print(f"\n\n{'='*80}")
        print(f"TEST: PCA z {n_comp} komponentami")
        print(f"{'='*80}")

        predictor = ImprovedDSEPredictor(data_dir, n_components=n_comp)

        # 1. Wczytaj dane
        X, y = predictor.load_data()

        # 2. Przygotuj
        predictor.prepare_data(X, y, test_size=0.2)

        # 3. Trenuj
        predictor.train_multiple_models()

        # 4. Ewaluuj
        results = predictor.evaluate_models()

        # 5. Zapisz najlepszy
        if n_comp == 20:  # Zapisz dla n_comp=20
            predictor.plot_results(save_path=f'improved_predictions_pca{n_comp}.png')
            predictor.analyze_pca_components(save_path=f'pca_analysis_{n_comp}.png')
            predictor.save_models('improved_models')

            # Przyklad predykcji
            print("\n[PREDICT] Przyklad predykcji:")
            test_embedding = predictor.scaler.inverse_transform(
                predictor.pca.inverse_transform(predictor.X_test[0:1])
            )[0]
            predictions = predictor.predict(test_embedding)
            actual = {
                'determination': predictor.y_test['determination'][0],
                'stability': predictor.y_test['stability'][0],
                'entropy': predictor.y_test['entropy'][0]
            }

            print("\n   Przewidywane:")
            for metric in ['determination', 'stability', 'entropy']:
                print(f"      {metric:15s}: {predictions[metric]:.4f}")

            print("\n   Rzeczywiste:")
            for metric in ['determination', 'stability', 'entropy']:
                print(f"      {metric:15s}: {actual[metric]:.4f}")

    print("\n" + "=" * 80)
    print("[SUCCESS] ZAKONCZONO!")
    print("=" * 80)


if __name__ == "__main__":
    main()
