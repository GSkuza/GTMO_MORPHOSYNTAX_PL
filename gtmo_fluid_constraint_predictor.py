#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
GTMØ Fluid Constraint Predictor - Predykcja D-S-E z płynnymi korelacjami
=========================================================================
Bazuje na rewizji teorii GTMØ gdzie S+E ∈ [0.509, 1.219] zamiast S+E=1

Kluczowe założenia:
1. S+E zmienia się fraktalnie (Weierstrass, dim=2.356)
2. Przestrzeń fazowa bez sztywnych granic
3. Emergencja zamiast constraintu
4. Dziwne atraktory w przestrzeni semantycznej
"""

import json
import numpy as np
from pathlib import Path
from typing import Dict, Tuple, List
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.linear_model import Ridge, Lasso, ElasticNet
from sklearn.multioutput import MultiOutputRegressor
import matplotlib.pyplot as plt
import joblib
from datetime import datetime


class FluidConstraintPredictor:
    """
    Predyktor D-S-E uwzględniający płynny constraint S+E.

    Zamiast przewidywać D, S, E niezależnie, przewidujemy:
    - D (determination)
    - S+E (fluid constraint)
    - E/(S+E) (ratio entropy w constraincie)

    Następnie rekonstruujemy S i E.
    """

    def __init__(self, data_dir: str, n_components: int = 20):
        self.data_dir = Path(data_dir)
        self.n_components = n_components

        self.pca = None
        self.scaler = StandardScaler()

        # Modele dla D, S+E, i E/(S+E)
        self.models = {
            'D': None,           # Determination
            'SE': None,          # S+E (fluid constraint)
            'E_ratio': None      # E/(S+E) ratio
        }

        self.X_train = None
        self.X_test = None
        self.y_train = {}
        self.y_test = {}

        # Statystyki constraintu z danych
        self.se_stats = {
            'mean': 0.8212,
            'std': 0.1875,
            'min': 0.509,
            'max': 1.219
        }

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

            except Exception:
                continue

        print(f"[OK] Wczytano {len(embeddings)} probek")

        X = np.array(embeddings)

        # Oblicz S+E i E/(S+E) ratio
        d_array = np.array(d_values)
        s_array = np.array(s_values)
        e_array = np.array(e_values)

        se_array = s_array + e_array
        e_ratio_array = np.divide(e_array, se_array,
                                  out=np.zeros_like(e_array),
                                  where=se_array!=0)

        y = {
            'D': d_array,
            'S': s_array,
            'E': e_array,
            'SE': se_array,
            'E_ratio': e_ratio_array
        }

        # Statystyki
        print("\n[STATS] Statystyki oryginalnych wartosci:")
        for metric in ['D', 'S', 'E']:
            values = y[metric]
            print(f"   {metric:10s}: mean={np.mean(values):.4f}, "
                  f"std={np.std(values):.4f}, range=[{np.min(values):.4f}, {np.max(values):.4f}]")

        print(f"\n[CONSTRAINT] Statystyki S+E:")
        print(f"   mean={np.mean(se_array):.4f}, std={np.std(se_array):.4f}")
        print(f"   range=[{np.min(se_array):.4f}, {np.max(se_array):.4f}]")

        print(f"\n[RATIO] Statystyki E/(S+E):")
        print(f"   mean={np.mean(e_ratio_array):.4f}, std={np.std(e_ratio_array):.4f}")
        print(f"   range=[{np.min(e_ratio_array):.4f}, {np.max(e_ratio_array):.4f}]")

        # Aktualizuj statystyki constraintu
        self.se_stats = {
            'mean': float(np.mean(se_array)),
            'std': float(np.std(se_array)),
            'min': float(np.min(se_array)),
            'max': float(np.max(se_array))
        }

        return X, y

    def prepare_data(self, X: np.ndarray, y: Dict[str, np.ndarray], test_size: float = 0.2):
        """Przygotuj dane z PCA i skalowaniem."""
        print(f"\n[PREP] Przygotowanie danych...")

        # 1. Skalowanie
        print(f"   1. Skalowanie embeddingow...")
        X_scaled = self.scaler.fit_transform(X)

        # 2. PCA
        print(f"   2. PCA: 768D -> {self.n_components}D...")
        self.pca = PCA(n_components=self.n_components, random_state=42)
        X_reduced = self.pca.fit_transform(X_scaled)

        explained_var = np.sum(self.pca.explained_variance_ratio_)
        print(f"      Wyjasniona wariancja: {explained_var:.2%}")

        # 3. Split
        print(f"   3. Podzial: {int((1-test_size)*100)}% train / {int(test_size)*100}% test")
        self.X_train, self.X_test = train_test_split(
            X_reduced, test_size=test_size, random_state=42
        )

        # Split dla D, SE, E_ratio (zamiast D, S, E niezależnie)
        for metric in ['D', 'SE', 'E_ratio', 'S', 'E']:  # zapisz wszystkie dla walidacji
            y_train, y_test = train_test_split(
                y[metric], test_size=test_size, random_state=42
            )
            self.y_train[metric] = y_train
            self.y_test[metric] = y_test

        print(f"      Train: {self.X_train.shape[0]} probek")
        print(f"      Test:  {self.X_test.shape[0]} probek")

    def train_models(self):
        """
        Trenuj modele dla D, S+E, E/(S+E).
        Używamy multi-output dla lepszej korelacji.
        """
        print("\n[TRAIN] Trening modeli z płynnym constraintem...")

        # Definicje modeli
        model_configs = {
            'Ridge': {
                'model': Ridge(),
                'params': {'alpha': [0.1, 1.0, 10.0, 100.0]}
            },
            'ElasticNet': {
                'model': ElasticNet(),
                'params': {
                    'alpha': [0.1, 1.0, 10.0],
                    'l1_ratio': [0.1, 0.5, 0.9]
                }
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
                    'max_depth': [5, 10],
                    'min_samples_split': [2, 5]
                }
            }
        }

        # Trenuj dla D, SE, E_ratio
        for target in ['D', 'SE', 'E_ratio']:
            print(f"\n[{target}]")

            best_score = -np.inf
            best_model_name = None
            best_model = None

            for model_name, config in model_configs.items():
                print(f"   Testowanie: {model_name}...", end=" ")

                grid = GridSearchCV(
                    config['model'],
                    config['params'],
                    cv=5,
                    scoring='r2',
                    n_jobs=-1
                )

                grid.fit(self.X_train, self.y_train[target])
                score = grid.best_score_

                print(f"R2 CV = {score:.4f}")

                if score > best_score:
                    best_score = score
                    best_model_name = model_name
                    best_model = grid.best_estimator_

            self.models[target] = {
                'name': best_model_name,
                'model': best_model,
                'cv_score': best_score
            }

            print(f"   >> NAJLEPSZY: {best_model_name} (R2 CV = {best_score:.4f})")

    def reconstruct_SE(self, D_pred: np.ndarray, SE_pred: np.ndarray,
                       E_ratio_pred: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Rekonstruuj S i E z przewidywanych D, S+E, E/(S+E).

        Args:
            D_pred: Przewidywane determination
            SE_pred: Przewidywane S+E
            E_ratio_pred: Przewidywane E/(S+E)

        Returns:
            (S_pred, E_pred)
        """
        # E = (S+E) * E/(S+E)
        E_pred = SE_pred * E_ratio_pred

        # S = (S+E) - E
        S_pred = SE_pred - E_pred

        # Clip do rozsądnych zakresów
        E_pred = np.clip(E_pred, 0, 1)
        S_pred = np.clip(S_pred, 0, 2)  # S może być >1 w płynnym modelu

        return S_pred, E_pred

    def evaluate_models(self) -> Dict[str, Dict[str, float]]:
        """Ewaluacja modeli."""
        print("\n[EVAL] Ewaluacja na zbiorze testowym:")

        # 1. Przewiduj D, SE, E_ratio
        D_pred = self.models['D']['model'].predict(self.X_test)
        SE_pred = self.models['SE']['model'].predict(self.X_test)
        E_ratio_pred = self.models['E_ratio']['model'].predict(self.X_test)

        # 2. Rekonstruuj S i E
        S_pred, E_pred = self.reconstruct_SE(D_pred, SE_pred, E_ratio_pred)

        # 3. Ewaluuj każdą metrykę
        results = {}

        predictions = {
            'D': D_pred,
            'S': S_pred,
            'E': E_pred,
            'SE': SE_pred,
            'E_ratio': E_ratio_pred
        }

        for metric in ['D', 'S', 'E', 'SE', 'E_ratio']:
            y_pred = predictions[metric]
            y_true = self.y_test[metric]

            mse = mean_squared_error(y_true, y_pred)
            rmse = np.sqrt(mse)
            mae = mean_absolute_error(y_true, y_pred)
            r2 = r2_score(y_true, y_pred)

            if metric in ['D', 'SE', 'E_ratio']:
                model_name = self.models[metric]['name']
                cv_score = self.models[metric]['cv_score']
            else:
                model_name = 'reconstructed'
                cv_score = None

            results[metric] = {
                'model': model_name,
                'MSE': mse,
                'RMSE': rmse,
                'MAE': mae,
                'R2': r2,
                'CV_R2': cv_score
            }

            print(f"\n   {metric:10s} ({model_name}):")
            print(f"      RMSE:   {rmse:.4f}")
            print(f"      MAE:    {mae:.4f}")
            print(f"      R2:     {r2:.4f}")
            if cv_score:
                print(f"      CV R2:  {cv_score:.4f}")

        # 4. Sprawdź zgodność z constraintem
        SE_reconstructed = S_pred + E_pred
        SE_error = np.mean(np.abs(SE_reconstructed - SE_pred))
        print(f"\n[CONSTRAINT] Błąd rekonstrukcji S+E: {SE_error:.6f}")

        return results

    def plot_results(self, save_path: str = None):
        """Wizualizacja wyników."""
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))

        # Przewiduj
        D_pred = self.models['D']['model'].predict(self.X_test)
        SE_pred = self.models['SE']['model'].predict(self.X_test)
        E_ratio_pred = self.models['E_ratio']['model'].predict(self.X_test)
        S_pred, E_pred = self.reconstruct_SE(D_pred, SE_pred, E_ratio_pred)

        predictions = {
            'D': D_pred,
            'S': S_pred,
            'E': E_pred,
            'SE': SE_pred,
            'E_ratio': E_ratio_pred
        }

        metrics = ['D', 'S', 'E', 'SE', 'E_ratio']
        titles = ['Determination', 'Stability', 'Entropy', 'S+E (Constraint)', 'E/(S+E) Ratio']

        for i, (metric, title) in enumerate(zip(metrics, titles)):
            row = i // 3
            col = i % 3
            ax = axes[row, col]

            y_pred = predictions[metric]
            y_true = self.y_test[metric]

            ax.scatter(y_true, y_pred, alpha=0.6, edgecolors='k', linewidth=0.5)

            min_val = min(y_true.min(), y_pred.min())
            max_val = max(y_true.max(), y_pred.max())
            ax.plot([min_val, max_val], [min_val, max_val], 'r--', lw=2, label='Ideal')

            r2 = r2_score(y_true, y_pred)
            rmse = np.sqrt(mean_squared_error(y_true, y_pred))

            ax.set_xlabel(f'Rzeczywiste {metric}', fontsize=11)
            ax.set_ylabel(f'Przewidywane {metric}', fontsize=11)
            ax.set_title(f'{title}\nR2={r2:.3f}, RMSE={rmse:.3f}', fontsize=12)
            ax.legend()
            ax.grid(alpha=0.3)

        # Usuń ostatni subplot (6. miejsce)
        fig.delaxes(axes[1, 2])

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"[PLOT] Wykres zapisano: {save_path}")

        plt.show()

    def plot_constraint_analysis(self, save_path: str = None):
        """Analiza płynnego constraintu S+E."""
        # Przewiduj
        SE_pred = self.models['SE']['model'].predict(self.X_test)
        SE_true = self.y_test['SE']

        fig, axes = plt.subplots(1, 2, figsize=(14, 5))

        # 1. Histogram S+E
        ax1 = axes[0]
        ax1.hist(SE_true, bins=20, alpha=0.5, label='Rzeczywiste', edgecolor='black')
        ax1.hist(SE_pred, bins=20, alpha=0.5, label='Przewidywane', edgecolor='black')
        ax1.axvline(self.se_stats['mean'], color='r', linestyle='--',
                   label=f"Mean={self.se_stats['mean']:.3f}")
        ax1.set_xlabel('S+E (Fluid Constraint)', fontsize=12)
        ax1.set_ylabel('Liczba probek', fontsize=12)
        ax1.set_title('Rozklad S+E (Płynny Constraint)', fontsize=14)
        ax1.legend()
        ax1.grid(alpha=0.3)

        # 2. S+E vs D
        ax2 = axes[1]
        D_pred = self.models['D']['model'].predict(self.X_test)
        D_true = self.y_test['D']

        ax2.scatter(D_true, SE_true, alpha=0.6, label='Rzeczywiste', s=50)
        ax2.scatter(D_pred, SE_pred, alpha=0.6, label='Przewidywane', marker='x', s=50)
        ax2.set_xlabel('Determination (D)', fontsize=12)
        ax2.set_ylabel('S+E', fontsize=12)
        ax2.set_title('Zaleznosc D vs S+E', fontsize=14)
        ax2.legend()
        ax2.grid(alpha=0.3)

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"[PLOT] Analiza constraintu zapisana: {save_path}")

        plt.show()

    def save_models(self, output_dir: str):
        """Zapisz modele."""
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        # Zapisz PCA i scaler
        joblib.dump(self.pca, output_path / 'pca.pkl')
        joblib.dump(self.scaler, output_path / 'scaler.pkl')

        # Zapisz modele
        for target in ['D', 'SE', 'E_ratio']:
            model_info = self.models[target]
            model_path = output_path / f'{target}_model.pkl'
            joblib.dump(model_info, model_path)
            print(f"[SAVE] {target}: {model_info['name']} -> {model_path}")

        # Metadata
        metadata = {
            'timestamp': datetime.now().isoformat(),
            'n_components': self.n_components,
            'n_samples_train': self.X_train.shape[0],
            'n_samples_test': self.X_test.shape[0],
            'models': {k: v['name'] for k, v in self.models.items()},
            'se_stats': self.se_stats,
            'theory': 'fluid_constraint',
            'constraint_range': f"[{self.se_stats['min']:.3f}, {self.se_stats['max']:.3f}]"
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

        # Przewiduj D, SE, E_ratio
        D_pred = self.models['D']['model'].predict(X_reduced)[0]
        SE_pred = self.models['SE']['model'].predict(X_reduced)[0]
        E_ratio_pred = self.models['E_ratio']['model'].predict(X_reduced)[0]

        # Rekonstruuj S, E
        S_pred, E_pred = self.reconstruct_SE(
            np.array([D_pred]),
            np.array([SE_pred]),
            np.array([E_ratio_pred])
        )

        return {
            'determination': float(np.clip(D_pred, 0, 1)),
            'stability': float(S_pred[0]),
            'entropy': float(E_pred[0]),
            'SE_constraint': float(SE_pred),
            'E_ratio': float(E_ratio_pred)
        }


def main():
    print("=" * 80)
    print("  GTMO FLUID CONSTRAINT PREDICTOR - Plynne korelacje S+E")
    print("=" * 80)

    data_dir = r"D:\GTMO_MORPHOSYNTAX\gtmo_results\analysis_15112025_no1_projekt_poselski_edited"

    predictor = FluidConstraintPredictor(data_dir, n_components=20)

    # 1. Wczytaj
    X, y = predictor.load_data()

    # 2. Przygotuj
    predictor.prepare_data(X, y, test_size=0.2)

    # 3. Trenuj
    predictor.train_models()

    # 4. Ewaluuj
    results = predictor.evaluate_models()

    # 5. Wizualizuj
    print("\n[VIZ] Generowanie wizualizacji...")
    predictor.plot_results(save_path='fluid_constraint_predictions.png')
    predictor.plot_constraint_analysis(save_path='fluid_constraint_analysis.png')

    # 6. Zapisz
    predictor.save_models('fluid_constraint_models')

    # 7. Przykład
    print("\n[PREDICT] Przyklad predykcji:")
    test_idx = 0
    # Rekonstruuj oryginalny embedding
    X_test_original = predictor.scaler.inverse_transform(
        predictor.pca.inverse_transform(predictor.X_test[test_idx:test_idx+1])
    )[0]

    predictions = predictor.predict(X_test_original)
    actual = {
        'determination': predictor.y_test['D'][test_idx],
        'stability': predictor.y_test['S'][test_idx],
        'entropy': predictor.y_test['E'][test_idx],
        'SE_constraint': predictor.y_test['SE'][test_idx],
    }

    print("\n   Przewidywane:")
    for key, val in predictions.items():
        print(f"      {key:20s}: {val:.4f}")

    print("\n   Rzeczywiste:")
    for key, val in actual.items():
        print(f"      {key:20s}: {val:.4f}")

    print("\n" + "=" * 80)
    print("[SUCCESS] ZAKONCZONO!")
    print("=" * 80)


if __name__ == "__main__":
    main()
