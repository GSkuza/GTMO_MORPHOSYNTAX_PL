#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Gradient Boosting Regressor dla predykcji wartości D-S-E z embeddingów HerBERT
===============================================================================
Ten skrypt trenuje model Gradient Boosting do przewidywania wartości
Determination, Stability i Entropy na podstawie embeddingów HerBERT (768 wymiarów).
"""

import json
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, List, Tuple
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
from datetime import datetime


    def load_models(self, model_dir: str):
        """
        Wczytaj zapisane modele.

        Args:
            model_dir: Katalog z zapisanymi modelami
        """
        model_path = Path(model_dir)

        for metric in ['determinations', 'stability', 'entropy']:
            model_file = model_path / f'gbr_{metric}_model.pkl'

            try:
                if model_file.exists():
                    self.models[metric] = joblib.load(model_file)
                    print(f"[LOAD] Wczytano model: {model_file}")
                else:
                    print(f"[LOAD] Nie znaleziono modelu: {model_file}")
            except (FileNotFoundError, TypeError) as e:
                print(f"[LOAD] Błąd wczytywania modelu: {e}")


def main() -> None:
    """
    Główna funkcja demonstracyjna.
    """
    print("=" * 80)
    print("  GRADIENT BOOSTING REGRESSOR - Predykcja D-S-E z Embeddingów HerBERT")
    print("=" * 80)

    # Ścieżka do danych
    data_dir: str = r"D:\GTMO_MORPHOSYNTAX\gtmo_results\analysis_15112025_no1_projekt_poselski_edited"

    try:
        # Inicjalizacja predyktora
        predictor = DSEGradientBoostingPredictor(data_dir)

        # 1. Wczytaj dane (z użyciem Pandas)
        X, y = predictor.load_data_pandas()

        # 2. Przygotuj dane (split + skalowanie)
        predictor.prepare_data_pandas(X, y, test_size=0.2, random_state=42)

        # 3. Trenuj modele (z użyciem joblib)
        predictor.train_models_parallel(
            n_estimators=200,
            learning_rate=0.1,
            max_depth=5,
            subsample=0.8,
            verbose=True,
            n_jobs=-1
        )

        # 4. Ewaluacja
        results = predictor.evaluate_models()

        # 5. Wizualizacja
        print("\n[VIZ] Generowanie wykresów...")
        predictor.plot_results(save_path='gbr_predictions_vs_actual.png')
        predictor.plot_feature_importance(top_n=20, save_path='gbr_feature_importance.png')

        # 6. Zapisz modele
        predictor.save_models('trained_models')

        # 7. Przyklad predykcji
        print("\n[PREDICT] Przyklad predykcji dla pierwszej probki testowej:")
        test_embedding = predictor.X_test.iloc[0]
        predictions = predictor.predict(test_embedding)
        actual = {
            'determinations': predictor.y_test['determinations'].iloc[0],
            'stability': predictor.y_test['stability'].iloc[0],
            'entropy': predictor.y_test['entropy'].iloc[0]
        }

        print("\n   Przewidywane wartosci:")
        for metric in ['determinations', 'stability', 'entropy']:
            print(f"      {metric:15s}: {predictions[metric]:.4f}")

        print("\n   Rzeczywiste wartosci:")
        for metric in ['determinations', 'stability', 'entropy']:
            print(f"      {metric:15s}: {actual[metric]:.4f}")

    except Exception as e:
        print(f"\n[ERROR] {e.__class__.__name__}: {e}")
        print("   Sprawdź błędy w kodzie i sprób ponownie.")


if __name__ == "__main__":
    main()
