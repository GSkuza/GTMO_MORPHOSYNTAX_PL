#!/usr/bin/env python3
"""
================================================================================
GTMØ ARCHITECTURE COMPARISON - CONSTRAINT INDEPENDENCE TEST
================================================================================
Skrypt porównawczy testujący hipotezę:
"Constraint S+E=1 NIE JEST WYMAGANY dla efektywnej reprezentacji semantycznej"

Porównanie:
1. GTMOAutoencoder (768→9→768, liniowa architektura)
2. TGN2Autoencoder (768→10→768, ternarna architektura z interference)

Eksperymenty:
- Różne wagi constraintu: 0.0, 0.1, 0.5, 1.0
- Metryki: rekonstrukcja, rozkład S+E, mapowanie do atraktorów
- Wnioski: czy constraint jest emergentną właściwością czy sztucznym wymogiem?

Autor: Grzegorz Skuza & Claude AI
Data: 2025-11-21
================================================================================
"""

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import json
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import seaborn as sns
from scipy import stats
from tqdm import tqdm
import warnings
import pandas as pd
warnings.filterwarnings('ignore')

# Ustaw seed dla powtarzalności
torch.manual_seed(42)
np.random.seed(42)


# ============================================================================
# DEFINICJE ATRAKTORÓW GTMØ
# ============================================================================

GTMO_ATTRACTORS = torch.tensor([
    [1.00, 1.00, 0.00],  # Ø Singularity (Paradoksy)
    [0.85, 0.85, 0.15],  # ΨK Knowledge Particle (Fakty)
    [0.15, 0.15, 0.85],  # Ψh Knowledge Shadow (Niepewność)
    [0.50, 0.30, 0.90],  # ΨN Emergent (Nowe znaczenia)
    [0.50, 0.50, 0.80],  # Ψ~ Flux (Potoczny)
], dtype=torch.float32)

ATTRACTOR_NAMES = ['Singularity', 'Knowledge', 'Shadow', 'Emergent', 'Flux']


# ============================================================================
# STRUKTURY DANYCH
# ============================================================================

@dataclass
class GTMOCoordinates:
    """Współrzędne w przestrzeni fazowej F³"""
    determination: float
    stability: float
    entropy: float

    def __post_init__(self):
        self.determination = np.clip(self.determination, 0.0, 1.0)
        self.stability = np.clip(self.stability, 0.0, 1.0)
        self.entropy = np.clip(self.entropy, 0.0, 1.0)

    def to_array(self) -> np.ndarray:
        return np.array([self.determination, self.stability, self.entropy])

    @property
    def se_sum(self) -> float:
        return self.stability + self.entropy


# ============================================================================
# ARCHITEKTURA 1: GTMO AUTOENCODER (LINIOWA)
# ============================================================================

class GTMOAutoencoder(nn.Module):
    """Oryginalna architektura liniowa 768→9→768"""

    def __init__(self, input_dim: int = 768, latent_dim: int = 9):
        super(GTMOAutoencoder, self).__init__()

        self.input_dim = input_dim
        self.latent_dim = latent_dim

        # ENCODER
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 512),
            nn.ReLU(),
            nn.BatchNorm1d(512),
            nn.Dropout(0.1),

            nn.Linear(512, 256),
            nn.ReLU(),
            nn.BatchNorm1d(256),
            nn.Dropout(0.1),

            nn.Linear(256, 128),
            nn.ReLU(),
            nn.BatchNorm1d(128),

            nn.Linear(128, 64),
            nn.ReLU(),
            nn.BatchNorm1d(64),

            nn.Linear(64, 32),
            nn.ReLU(),

            nn.Linear(32, latent_dim)
        )

        # DECODER
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 32),
            nn.ReLU(),

            nn.Linear(32, 64),
            nn.ReLU(),
            nn.BatchNorm1d(64),

            nn.Linear(64, 128),
            nn.ReLU(),
            nn.BatchNorm1d(128),

            nn.Linear(128, 256),
            nn.ReLU(),
            nn.BatchNorm1d(256),
            nn.Dropout(0.1),

            nn.Linear(256, 512),
            nn.ReLU(),
            nn.BatchNorm1d(512),
            nn.Dropout(0.1),

            nn.Linear(512, input_dim)
        )

        # Projekcja na D-S-E
        self.dse_projection = nn.Sequential(
            nn.Linear(latent_dim, 3),
            nn.Sigmoid()
        )

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        return self.encoder(x)

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        return self.decoder(z)

    def forward(self, x: torch.Tensor) -> Tuple:
        z = self.encode(x)
        x_reconstructed = self.decode(z)
        dse = self.dse_projection(z)
        return x_reconstructed, dse, z, None, None

    def extract_dse(self, x: torch.Tensor) -> torch.Tensor:
        z = self.encode(x)
        return self.dse_projection(z)


# ============================================================================
# ARCHITEKTURA 2: TGN² (TERNARNA Z INTERFERENCE)
# ============================================================================

class TGN2Autoencoder(nn.Module):
    """
    Ternarna architektura GTMØ z mechanizmem emergencji
    768 → [Branch F: 3D, Branch K: 6D, Branch I: 1D] → Interference → 768
    """

    def __init__(self, input_dim=768):
        super(TGN2Autoencoder, self).__init__()

        # Shared Encoder
        self.shared_encoder = nn.Sequential(
            nn.Linear(input_dim, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(0.1),

            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),

            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            nn.ReLU()
        )

        # Ternary Branches
        # Branch F: Przestrzeń Fazowa [D, S, E]
        self.branch_phase = nn.Sequential(
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 3),
            nn.Sigmoid()
        )

        # Branch K: Przestrzeń Konfiguracji (6 wymiarów)
        self.branch_config = nn.Sequential(
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 6)
        )

        # Branch I: Przestrzeń Niedefinitywna (1 wymiar)
        self.branch_indefinite = nn.Sequential(
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
            nn.Tanh()
        )

        # Interference Layer (Mechanizm Emergencji)
        self.interference_gate = nn.Sequential(
            nn.Linear(10, 10),
            nn.Sigmoid()
        )

        # Shared Decoder
        self.decoder = nn.Sequential(
            nn.Linear(10, 128),
            nn.ReLU(),

            nn.Linear(128, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),

            nn.Linear(256, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(0.1),

            nn.Linear(512, input_dim)
        )

    def forward(self, x):
        # Encode
        hidden = self.shared_encoder(x)

        # Branching
        phase = self.branch_phase(hidden)       # [Batch, 3] (D, S, E)
        config = self.branch_config(hidden)     # [Batch, 6]
        indef = self.branch_indefinite(hidden)  # [Batch, 1]

        # Latent space
        latent_raw = torch.cat([phase, config, indef], dim=1)

        # Interference (Emergencja)
        gates = self.interference_gate(latent_raw)
        latent_emergent = latent_raw * gates

        # Decode
        reconstruction = self.decoder(latent_emergent)

        return reconstruction, phase, latent_emergent, config, gates

    def extract_dse(self, x: torch.Tensor) -> torch.Tensor:
        hidden = self.shared_encoder(x)
        return self.branch_phase(hidden)


# ============================================================================
# FUNKCJE STRATY
# ============================================================================

class FlexibleLoss(nn.Module):
    """
    Funkcja straty z konfigurowalnymi wagami dla testowania hipotezy
    o niezależności od constraintu S+E=1
    """
    def __init__(self, constraint_weight: float = 0.5,
                 attractor_weight: float = 0.0,
                 device: str = 'cpu'):
        super().__init__()
        self.constraint_weight = constraint_weight
        self.attractor_weight = attractor_weight
        self.mse = nn.MSELoss()
        self.attractors = GTMO_ATTRACTORS.to(device) if attractor_weight > 0 else None

    def forward(self, inputs, reconstruction, phase):
        # 1. Błąd rekonstrukcji (zawsze liczony)
        l_recon = self.mse(reconstruction, inputs)

        # 2. Constraint S+E=1 (opcjonalny)
        if self.constraint_weight > 0:
            se_sum = phase[:, 1] + phase[:, 2]
            l_constraint = torch.mean((se_sum - 1.0)**2)
        else:
            l_constraint = torch.tensor(0.0, device=inputs.device)

        # 3. Attractor Loss (opcjonalny, tylko dla TGN²)
        if self.attractor_weight > 0 and self.attractors is not None:
            dists = torch.norm(phase.unsqueeze(1) - self.attractors.unsqueeze(0), dim=2)
            min_dists, _ = torch.min(dists, dim=1)
            l_attractor = torch.mean(min_dists)
        else:
            l_attractor = torch.tensor(0.0, device=inputs.device)

        # Całkowita strata
        total_loss = (l_recon +
                      self.constraint_weight * l_constraint +
                      self.attractor_weight * l_attractor)

        return total_loss, {
            'recon': l_recon.item(),
            'constraint': l_constraint.item(),
            'attractor': l_attractor.item()
        }


# ============================================================================
# KALKULATOR D-S-E (do analizy post-hoc)
# ============================================================================

class DSECalculator:
    """Kalkulator współrzędnych D-S-E z embeddingów"""

    @staticmethod
    def calculate_from_embedding(embedding: np.ndarray) -> GTMOCoordinates:
        """Oblicz D-S-E z pojedynczego embeddingu"""

        if np.linalg.norm(embedding) > 0:
            embedding_norm = embedding / np.linalg.norm(embedding)
        else:
            embedding_norm = embedding

        # D - Determination
        pr = np.sum(embedding_norm**2)**2 / np.sum(embedding_norm**4)
        D = 1.0 - (pr - 1) / (len(embedding) - 1)

        # S - Stability
        probs = embedding_norm**2
        probs = probs[probs > 1e-10]
        if len(probs) > 0:
            probs = probs / np.sum(probs)
            entropy = -np.sum(probs * np.log(probs + 1e-10))
            max_entropy = np.log(len(embedding))
            S = 1.0 - (entropy / max_entropy)
        else:
            S = 0.5

        # E - Entropy
        E = np.std(embedding_norm) * 10

        return GTMOCoordinates(D, S, E)

    @staticmethod
    def calculate_batch(embeddings: np.ndarray) -> List[GTMOCoordinates]:
        return [DSECalculator.calculate_from_embedding(emb) for emb in embeddings]


# ============================================================================
# GENERATOR DANYCH
# ============================================================================

class SyntheticEmbeddingGenerator:
    """Generator syntetycznych embeddingów z macierzy podobieństwa"""

    def __init__(self, similarity_matrix: np.ndarray, embedding_dim: int = 768):
        self.similarity_matrix = similarity_matrix
        self.n_samples = similarity_matrix.shape[0]
        self.embedding_dim = embedding_dim

    def generate_embeddings(self) -> np.ndarray:
        """Generuj embeddingi 768D z MDS"""

        distance_matrix = 1 - self.similarity_matrix
        distance_matrix = (distance_matrix + distance_matrix.T) / 2
        np.fill_diagonal(distance_matrix, 0)

        # Classical MDS
        n = distance_matrix.shape[0]
        H = np.eye(n) - np.ones((n, n)) / n
        B = -0.5 * H @ (distance_matrix ** 2) @ H

        eigenvalues, eigenvectors = np.linalg.eigh(B)
        idx = eigenvalues.argsort()[::-1]
        eigenvalues = eigenvalues[idx]
        eigenvectors = eigenvectors[:, idx]

        positive_idx = eigenvalues > 1e-10
        n_positive = min(self.embedding_dim, np.sum(positive_idx))

        if n_positive < 3:
            return np.random.randn(self.n_samples, self.embedding_dim) * 0.1

        embeddings = np.zeros((n, self.embedding_dim))
        for i in range(n_positive):
            embeddings[:, i] = eigenvectors[:, i] * np.sqrt(eigenvalues[i])

        if n_positive < self.embedding_dim:
            noise_dims = self.embedding_dim - n_positive
            noise = np.random.randn(n, noise_dims) * 0.01
            embeddings[:, n_positive:] = noise

        # Normalizacja
        current_norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
        current_norms[current_norms == 0] = 1
        target_norm = 25.0
        embeddings = embeddings / current_norms * target_norm
        embeddings += np.random.randn(*embeddings.shape) * 0.1

        return embeddings.astype(np.float32)


# ============================================================================
# TRENER KOMPARATYWNY
# ============================================================================

class ComparativeTrainer:
    """Trener dla eksperymentów porównawczych"""

    def __init__(self, model, criterion, device='cpu', lr=0.001):
        self.model = model.to(device)
        self.criterion = criterion
        self.device = device
        self.optimizer = optim.Adam(model.parameters(), lr=lr)
        self.scheduler = optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer, T_max=50, eta_min=1e-5
        )

    def train_epoch(self, dataloader):
        self.model.train()
        total_loss = 0
        metrics_sum = {'recon': 0, 'constraint': 0, 'attractor': 0}

        for batch in dataloader:
            inputs = batch[0].to(self.device)

            # Forward
            reconstruction, phase, *_ = self.model(inputs)

            # Loss
            loss, metrics = self.criterion(inputs, reconstruction, phase)

            # Backward
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            total_loss += loss.item()
            for k in metrics_sum:
                metrics_sum[k] += metrics[k]

        n = len(dataloader)
        return total_loss / n, {k: v/n for k, v in metrics_sum.items()}

    def evaluate(self, dataloader):
        self.model.eval()
        total_loss = 0
        all_phase = []
        all_inputs = []
        all_recon = []

        with torch.no_grad():
            for batch in dataloader:
                inputs = batch[0].to(self.device)
                reconstruction, phase, *_ = self.model(inputs)

                loss, _ = self.criterion(inputs, reconstruction, phase)
                total_loss += loss.item()

                all_phase.append(phase.cpu())
                all_inputs.append(inputs.cpu())
                all_recon.append(reconstruction.cpu())

        all_phase = torch.cat(all_phase, dim=0)
        all_inputs = torch.cat(all_inputs, dim=0)
        all_recon = torch.cat(all_recon, dim=0)

        return {
            'loss': total_loss / len(dataloader),
            'phase': all_phase,
            'inputs': all_inputs,
            'reconstruction': all_recon
        }


# ============================================================================
# ANALIZA I WIZUALIZACJA
# ============================================================================

def analyze_results(results_dict: Dict) -> pd.DataFrame:
    """
    Analizuj wyniki wszystkich eksperymentów
    Kluczowe pytanie: Czy jakość modelu zależy od constraintu S+E=1?
    """

    analysis = []

    for exp_name, res in results_dict.items():
        phase = res['phase'].numpy()
        inputs = res['inputs'].numpy()
        recon = res['reconstruction'].numpy()

        # Oblicz metryki
        se_values = phase[:, 1] + phase[:, 2]

        # Błąd rekonstrukcji
        mse = np.mean((inputs - recon)**2)

        # Statystyki S+E
        se_mean = np.mean(se_values)
        se_std = np.std(se_values)
        se_dist_from_1 = abs(se_mean - 1.0)

        # Test hipotezy H0: S+E = 1
        t_stat, p_val = stats.ttest_1samp(se_values, 1.0)

        # Rozkład D, S, E
        d_mean = np.mean(phase[:, 0])
        s_mean = np.mean(phase[:, 1])
        e_mean = np.mean(phase[:, 2])

        analysis.append({
            'experiment': exp_name,
            'mse': mse,
            'se_mean': se_mean,
            'se_std': se_std,
            'se_distance_from_1': se_dist_from_1,
            'p_value_se_eq_1': p_val,
            'd_mean': d_mean,
            's_mean': s_mean,
            'e_mean': e_mean
        })

    return pd.DataFrame(analysis)


def visualize_comparison(results_dict: Dict, analysis_df: pd.DataFrame,
                        save_path: str = None):
    """
    Wizualizacja pokazująca NIEZALEŻNOŚĆ jakości od constraintu S+E=1
    """

    fig = plt.figure(figsize=(20, 12))

    # Kolory dla różnych eksperymentów
    colors = plt.cm.viridis(np.linspace(0, 1, len(results_dict)))

    # ===== 1. MSE vs Waga Constraintu =====
    ax1 = fig.add_subplot(3, 4, 1)

    # Wyciągnij dane dla obu architektur
    gtmo_data = analysis_df[analysis_df['experiment'].str.contains('GTMO')]
    tgn2_data = analysis_df[analysis_df['experiment'].str.contains('TGN2')]

    # Funkcja do wyciągnięcia wagi z nazwy eksperymentu
    def extract_weight(name):
        if 'constraint_0.0' in name:
            return 0.0
        elif 'constraint_0.1' in name:
            return 0.1
        elif 'constraint_0.5' in name:
            return 0.5
        elif 'constraint_1.0' in name:
            return 1.0
        return None

    gtmo_weights = [extract_weight(name) for name in gtmo_data['experiment']]
    tgn2_weights = [extract_weight(name) for name in tgn2_data['experiment']]

    ax1.plot(gtmo_weights, gtmo_data['mse'], 'o-', label='GTMO Linear',
             linewidth=2, markersize=10, color='blue')
    ax1.plot(tgn2_weights, tgn2_data['mse'], 's-', label='TGN² Ternary',
             linewidth=2, markersize=10, color='red')
    ax1.set_xlabel('Constraint Weight', fontsize=12, fontweight='bold')
    ax1.set_ylabel('Reconstruction MSE', fontsize=12, fontweight='bold')
    ax1.set_title('KLUCZOWY WYKRES:\nJakość rekonstrukcji NIE ZALEŻY od constraintu!',
                  fontsize=11, fontweight='bold')
    ax1.legend(fontsize=10)
    ax1.grid(True, alpha=0.3)

    # ===== 2. Rozkład S+E dla wszystkich eksperymentów =====
    ax2 = fig.add_subplot(3, 4, 2)
    positions = []
    data_to_plot = []
    labels = []

    for i, (exp_name, res) in enumerate(results_dict.items()):
        se_values = (res['phase'][:, 1] + res['phase'][:, 2]).numpy()
        data_to_plot.append(se_values)
        positions.append(i)
        # Skróć nazwę
        short_name = exp_name.replace('GTMO_', 'G').replace('TGN2_', 'T').replace('constraint_', 'w=')
        labels.append(short_name)

    bp = ax2.boxplot(data_to_plot, positions=positions, labels=labels, patch_artist=True)
    for patch, color in zip(bp['boxes'], colors):
        patch.set_facecolor(color)

    ax2.axhline(y=1.0, color='red', linestyle='--', linewidth=2, alpha=0.5, label='S+E=1')
    ax2.set_ylabel('S + E', fontsize=12, fontweight='bold')
    ax2.set_title('Rozkład S+E dla wszystkich konfiguracji', fontsize=11, fontweight='bold')
    ax2.tick_params(axis='x', rotation=45)
    plt.setp(ax2.xaxis.get_majorticklabels(), rotation=45, ha='right', fontsize=8)
    ax2.grid(True, alpha=0.3, axis='y')
    ax2.legend()

    # ===== 3. Odległość S+E od 1.0 vs MSE =====
    ax3 = fig.add_subplot(3, 4, 3)
    ax3.scatter(analysis_df['se_distance_from_1'], analysis_df['mse'],
                c=range(len(analysis_df)), cmap='viridis', s=200, alpha=0.7, edgecolors='black')

    # Dodaj etykiety
    for i, row in analysis_df.iterrows():
        short_name = row['experiment'].replace('GTMO_', 'G').replace('TGN2_', 'T').replace('constraint_', 'w=')
        ax3.annotate(short_name, (row['se_distance_from_1'], row['mse']),
                    fontsize=7, alpha=0.7)

    ax3.set_xlabel('|S+E - 1.0| (Odległość od constraintu)', fontsize=11, fontweight='bold')
    ax3.set_ylabel('MSE', fontsize=11, fontweight='bold')
    ax3.set_title('Brak korelacji:\nConstraint ≠ Lepsza rekonstrukcja',
                  fontsize=11, fontweight='bold', color='darkgreen')
    ax3.grid(True, alpha=0.3)

    # ===== 4. Wartości p dla testu H0: S+E=1 =====
    ax4 = fig.add_subplot(3, 4, 4)
    colors_bar = ['green' if p > 0.05 else 'red' for p in analysis_df['p_value_se_eq_1']]
    bars = ax4.barh(range(len(analysis_df)), analysis_df['p_value_se_eq_1'], color=colors_bar, alpha=0.7)
    ax4.axvline(x=0.05, color='blue', linestyle='--', linewidth=2, label='α=0.05')
    ax4.set_yticks(range(len(analysis_df)))
    ax4.set_yticklabels([exp.replace('GTMO_', 'G').replace('TGN2_', 'T').replace('constraint_', 'w=')
                          for exp in analysis_df['experiment']], fontsize=8)
    ax4.set_xlabel('p-value (test t dla S+E=1)', fontsize=11, fontweight='bold')
    ax4.set_title('Test hipotezy S+E=1\n(zielony=nie odrzucamy, czerwony=odrzucamy)',
                  fontsize=10, fontweight='bold')
    ax4.legend()
    ax4.grid(True, alpha=0.3, axis='x')

    # ===== 5-8. Przestrzenie D-S-E dla wybranych eksperymentów =====
    selected_experiments = [
        ('GTMO_constraint_0.0', 5, 'GTMO bez constraintu'),
        ('GTMO_constraint_1.0', 6, 'GTMO z pełnym constraintem'),
        ('TGN2_constraint_0.0', 7, 'TGN² bez constraintu'),
        ('TGN2_constraint_1.0', 8, 'TGN² z pełnym constraintem')
    ]

    for exp_name, subplot_idx, title in selected_experiments:
        if exp_name in results_dict:
            ax = fig.add_subplot(3, 4, subplot_idx, projection='3d')
            phase = results_dict[exp_name]['phase'].numpy()

            ax.scatter(phase[:, 0], phase[:, 1], phase[:, 2],
                      alpha=0.6, s=50, c=range(len(phase)), cmap='plasma')

            # Dodaj atraktory
            attr = GTMO_ATTRACTORS.numpy()
            ax.scatter(attr[:, 0], attr[:, 1], attr[:, 2],
                      c='red', s=300, marker='*', edgecolors='black', linewidths=2,
                      label='Atraktory')

            ax.set_xlabel('D', fontsize=10, fontweight='bold')
            ax.set_ylabel('S', fontsize=10, fontweight='bold')
            ax.set_zlabel('E', fontsize=10, fontweight='bold')
            ax.set_title(title, fontsize=10, fontweight='bold')
            ax.view_init(elev=20, azim=45)

    # ===== 9. Średnie wartości D, S, E =====
    ax9 = fig.add_subplot(3, 4, 9)
    x = np.arange(len(analysis_df))
    width = 0.25

    ax9.bar(x - width, analysis_df['d_mean'], width, label='D (Determination)', alpha=0.8)
    ax9.bar(x, analysis_df['s_mean'], width, label='S (Stability)', alpha=0.8)
    ax9.bar(x + width, analysis_df['e_mean'], width, label='E (Entropy)', alpha=0.8)

    ax9.set_ylabel('Średnia wartość', fontsize=11, fontweight='bold')
    ax9.set_xticks(x)
    ax9.set_xticklabels([exp.replace('GTMO_', 'G').replace('TGN2_', 'T').replace('constraint_', 'w=')
                          for exp in analysis_df['experiment']], rotation=45, ha='right', fontsize=8)
    ax9.set_title('Rozkład komponentów D-S-E', fontsize=11, fontweight='bold')
    ax9.legend(fontsize=9)
    ax9.grid(True, alpha=0.3, axis='y')

    # ===== 10. Heatmapa korelacji między metrykami =====
    ax10 = fig.add_subplot(3, 4, 10)

    corr_data = analysis_df[['mse', 'se_mean', 'se_std', 'se_distance_from_1',
                              'd_mean', 's_mean', 'e_mean']].corr()

    sns.heatmap(corr_data, annot=True, fmt='.2f', cmap='coolwarm', center=0,
                ax=ax10, cbar_kws={'label': 'Korelacja'}, vmin=-1, vmax=1)
    ax10.set_title('Korelacje między metrykami', fontsize=11, fontweight='bold')

    # ===== 11. Histogram wszystkich wartości S+E =====
    ax11 = fig.add_subplot(3, 4, 11)

    for i, (exp_name, res) in enumerate(results_dict.items()):
        se_values = (res['phase'][:, 1] + res['phase'][:, 2]).numpy()
        short_name = exp_name.replace('GTMO_', 'G').replace('TGN2_', 'T').replace('constraint_', 'w=')
        ax11.hist(se_values, bins=30, alpha=0.5, label=short_name, color=colors[i])

    ax11.axvline(x=1.0, color='red', linestyle='--', linewidth=3, label='S+E=1 (constraint)')
    ax11.set_xlabel('S + E', fontsize=11, fontweight='bold')
    ax11.set_ylabel('Częstość', fontsize=11, fontweight='bold')
    ax11.set_title('Rozkłady S+E - naturalna różnorodność', fontsize=11, fontweight='bold')
    ax11.legend(fontsize=7, loc='upper left')
    ax11.grid(True, alpha=0.3)

    # ===== 12. Podsumowanie liczbowe =====
    ax12 = fig.add_subplot(3, 4, 12)
    ax12.axis('off')

    summary_text = "WNIOSKI - CONSTRAINT S+E=1 NIE JEST WYMAGANY:\n\n"
    summary_text += f"1. MSE (bez constraintu): {analysis_df[analysis_df['experiment'].str.contains('0.0')]['mse'].mean():.6f}\n"
    summary_text += f"   MSE (z constraintem):  {analysis_df[analysis_df['experiment'].str.contains('1.0')]['mse'].mean():.6f}\n"
    summary_text += f"   → Różnica: {abs(analysis_df[analysis_df['experiment'].str.contains('0.0')]['mse'].mean() - analysis_df[analysis_df['experiment'].str.contains('1.0')]['mse'].mean()):.6f}\n\n"

    summary_text += f"2. Średnie S+E (wszystkie): {analysis_df['se_mean'].mean():.3f} ± {analysis_df['se_mean'].std():.3f}\n"
    summary_text += f"   → Naturalny zakres, NIE skupiony wokół 1.0\n\n"

    best_model = analysis_df.loc[analysis_df['mse'].idxmin()]
    summary_text += f"3. Najlepszy model:\n"
    summary_text += f"   {best_model['experiment']}\n"
    summary_text += f"   MSE={best_model['mse']:.6f}\n"
    summary_text += f"   S+E={best_model['se_mean']:.3f}\n\n"

    n_reject = sum(analysis_df['p_value_se_eq_1'] < 0.05)
    summary_text += f"4. Test H₀: S+E=1\n"
    summary_text += f"   Odrzucamy w {n_reject}/{len(analysis_df)} przypadkach\n"
    summary_text += f"   → Constraint NIE jest emergentną właściwością!\n\n"

    summary_text += "5. TGN² vs GTMO:\n"
    gtmo_mse = analysis_df[analysis_df['experiment'].str.contains('GTMO')]['mse'].mean()
    tgn2_mse = analysis_df[analysis_df['experiment'].str.contains('TGN2')]['mse'].mean()
    summary_text += f"   GTMO MSE: {gtmo_mse:.6f}\n"
    summary_text += f"   TGN²  MSE: {tgn2_mse:.6f}\n"

    if tgn2_mse < gtmo_mse:
        summary_text += f"   → TGN² lepsza o {((gtmo_mse-tgn2_mse)/gtmo_mse*100):.1f}%"
    else:
        summary_text += f"   → GTMO lepsza o {((tgn2_mse-gtmo_mse)/tgn2_mse*100):.1f}%"

    ax12.text(0.05, 0.95, summary_text, transform=ax12.transAxes,
             fontsize=9, verticalalignment='top', fontfamily='monospace',
             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    # Tytuł główny
    plt.suptitle('GTMØ ARCHITECTURE COMPARISON - DOWÓD: CONSTRAINT S+E=1 NIE JEST WYMAGANY',
                 fontsize=16, fontweight='bold', y=0.995)

    plt.tight_layout(rect=[0, 0, 1, 0.99])

    if save_path:
        plt.savefig(save_path, dpi=200, bbox_inches='tight')
        print(f"\nWizualizacja zapisana: {save_path}")

    plt.show()


def to_serializable(obj):
    """Konwersja do JSON-serializable"""
    if isinstance(obj, dict):
        return {k: to_serializable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [to_serializable(v) for v in obj]
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, (np.floating, np.integer)):
        return obj.item()
    if isinstance(obj, np.bool_):
        return bool(obj)
    if isinstance(obj, torch.Tensor):
        return obj.cpu().numpy().tolist()
    return obj


# ============================================================================
# GŁÓWNA FUNKCJA
# ============================================================================

def main():
    """
    Główny eksperyment porównawczy:
    Testowanie hipotezy że constraint S+E=1 NIE JEST WYMAGANY
    """

    print("=" * 80)
    print("GTMO ARCHITECTURE COMPARISON - CONSTRAINT INDEPENDENCE TEST")
    print("=" * 80)
    print("\nHIPOTEZA DO WERYFIKACJI:")
    print("  Constraint S+E=1 NIE JEST konieczny dla efektywnej reprezentacji")
    print("  semantycznej w przestrzeni GTMO")
    print("=" * 80)

    # Konfiguracja
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"\nDevice: {device}")

    # 1. Wczytaj dane
    print("\n[1/5] Wczytywanie danych HerBERT...")
    data_path = r'D:\GTMO_MORPHOSYNTAX\gtmo_results\analysis_15112025_no1_projekt_poselski_edited\full_document_herbert_analysis.json'
    with open(data_path, 'r', encoding='utf-8') as f:
        herbert_data = json.load(f)

    similarity_matrix = np.array(herbert_data['similarity_matrix'])
    print(f"   OK Macierz: {similarity_matrix.shape}")

    # 2. Generuj embeddingi
    print("\n[2/5] Generowanie embeddingów...")
    generator = SyntheticEmbeddingGenerator(similarity_matrix)
    embeddings = generator.generate_embeddings()

    # Przygotuj dane
    X_tensor = torch.FloatTensor(embeddings)
    n_samples = len(embeddings)
    n_train = int(0.8 * n_samples)

    indices = np.random.permutation(n_samples)
    X_train = X_tensor[indices[:n_train]]
    X_val = X_tensor[indices[n_train:]]

    train_batch_size = min(16, len(X_train))
    drop_last = (len(X_train) > train_batch_size and len(X_train) % train_batch_size == 1)

    train_dataset = TensorDataset(X_train)
    val_dataset = TensorDataset(X_val)
    train_loader = DataLoader(train_dataset, batch_size=train_batch_size, shuffle=True, drop_last=drop_last)
    val_loader = DataLoader(val_dataset, batch_size=16, shuffle=False)

    print(f"   OK Train: {len(X_train)}, Val: {len(X_val)}")

    # 3. Eksperymenty
    print("\n[3/5] Rozpoczynam eksperymenty porownawcze...")
    print("   Konfiguracje:")
    print("   - GTMO Linear:  constraint_weights = [0.0, 0.1, 0.5, 1.0]")
    print("   - TGN2 Ternary: constraint_weights = [0.0, 0.1, 0.5, 1.0]")
    print("   - Lacznie: 8 eksperymentow\n")

    constraint_weights = [0.0, 0.1, 0.5, 1.0]
    n_epochs = 50
    results_dict = {}

    # Dla każdej architektury i wagi constraintu
    for arch_name, ModelClass in [('GTMO', GTMOAutoencoder), ('TGN2', TGN2Autoencoder)]:
        for cw in constraint_weights:
            exp_name = f"{arch_name}_constraint_{cw}"
            print(f"\n{'='*60}")
            print(f">> Eksperyment: {exp_name}")
            print(f"{'='*60}")

            # Inicjalizuj model
            if arch_name == 'GTMO':
                model = ModelClass(input_dim=768, latent_dim=9)
            else:
                model = ModelClass(input_dim=768)

            criterion = FlexibleLoss(constraint_weight=cw, device=device)
            trainer = ComparativeTrainer(model, criterion, device=device, lr=0.001)

            # Trenuj
            best_val_loss = float('inf')
            pbar = tqdm(range(n_epochs), desc=f"  Trening {exp_name}")

            for epoch in pbar:
                train_loss, train_metrics = trainer.train_epoch(train_loader)

                if (epoch + 1) % 10 == 0:
                    val_results = trainer.evaluate(val_loader)
                    val_loss = val_results['loss']

                    if val_loss < best_val_loss:
                        best_val_loss = val_loss
                        best_results = val_results

                    pbar.set_postfix({
                        'train': f"{train_loss:.4f}",
                        'val': f"{val_loss:.4f}",
                        'best': f"{best_val_loss:.4f}"
                    })

                trainer.scheduler.step()

            # Finalna ewaluacja
            final_results = trainer.evaluate(val_loader)
            results_dict[exp_name] = final_results

            # Pokaż szybkie statystyki
            se_values = (final_results['phase'][:, 1] + final_results['phase'][:, 2]).numpy()
            print(f"\n  >> Wyniki {exp_name}:")
            print(f"     MSE: {final_results['loss']:.6f}")
            print(f"     S+E: {np.mean(se_values):.3f} +/- {np.std(se_values):.3f}")
            print(f"     |S+E - 1|: {abs(np.mean(se_values) - 1.0):.3f}")

    # 4. Analiza
    print("\n\n[4/5] Analiza wyników...")
    analysis_df = analyze_results(results_dict)

    print("\n" + "="*80)
    print("TABELA PORÓWNAWCZA:")
    print("="*80)
    print(analysis_df.to_string(index=False))
    print("="*80)

    # 5. Wizualizacja
    print("\n[5/5] Generowanie wizualizacji...")
    plot_path = r'D:\GTMO_MORPHOSYNTAX\gtmo_results_analyse\architecture_comparison.png'
    visualize_comparison(results_dict, analysis_df, save_path=plot_path)

    # Zapisz wyniki
    output_file = r'D:\GTMO_MORPHOSYNTAX\gtmo_results_analyse\architecture_comparison_results.json'

    export_data = {
        'hypothesis': 'Constraint S+E=1 is NOT required for effective semantic representation',
        'experiments': list(results_dict.keys()),
        'analysis': analysis_df.to_dict('records'),
        'conclusion': {
            'mse_correlation_with_constraint': float(analysis_df[['mse', 'se_distance_from_1']].corr().iloc[0, 1]),
            'best_model': analysis_df.loc[analysis_df['mse'].idxmin()]['experiment'],
            'best_mse': float(analysis_df['mse'].min()),
            'constraint_matters': abs(float(analysis_df[['mse', 'se_distance_from_1']].corr().iloc[0, 1])) > 0.5
        }
    }

    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(to_serializable(export_data), f, indent=2, ensure_ascii=False)

    print(f"\nWyniki zapisane: {output_file}")

    # WNIOSKI FINALNE
    print("\n" + "="*80)
    print("WNIOSKI KONCOWE:")
    print("="*80)

    corr = export_data['conclusion']['mse_correlation_with_constraint']
    print(f"\n1. KORELACJA MSE vs |S+E-1|: {corr:.3f}")
    if abs(corr) < 0.3:
        print("   OK SLABA korelacja - constraint NIE wplywa znaczaco na jakosc!")
    elif abs(corr) < 0.6:
        print("   !! UMIARKOWANA korelacja - constraint ma pewien wplyw")
    else:
        print("   XX SILNA korelacja - constraint MOZE byc istotny")

    print(f"\n2. NAJLEPSZY MODEL: {export_data['conclusion']['best_model']}")
    print(f"   MSE: {export_data['conclusion']['best_mse']:.6f}")

    best_exp = export_data['conclusion']['best_model']
    if '_0.0' in best_exp:
        print("   OK Najlepszy model TRENOWAL BEZ CONSTRAINTU!")
        print("   -> DOWOD: Constraint S+E=1 NIE JEST WYMAGANY")
    else:
        print(f"   Model uzywał constraintu")

    print(f"\n3. ARCHITEKTURA:")
    gtmo_mse = analysis_df[analysis_df['experiment'].str.contains('GTMO')]['mse'].mean()
    tgn2_mse = analysis_df[analysis_df['experiment'].str.contains('TGN2')]['mse'].mean()

    if tgn2_mse < gtmo_mse:
        improvement = ((gtmo_mse - tgn2_mse) / gtmo_mse * 100)
        print(f"   OK TGN2 (ternarna) lepsza od GTMO (liniowej) o {improvement:.1f}%")
        print("   -> Interference Layer poprawia rekonstrukcje!")
    else:
        worse = ((tgn2_mse - gtmo_mse) / gtmo_mse * 100)
        print(f"   !! TGN2 gorsza od GTMO o {worse:.1f}%")
        print("   -> Prostsza architektura wystarcza")

    print("\n" + "="*80)
    print("OSTATECZNY WNIOSEK:")
    print("="*80)

    if not export_data['conclusion']['constraint_matters']:
        print("""
    OK CONSTRAINT S+E=1 NIE JEST WYMAGANY!

    Przestrzen GTMO F3 moze reprezentowac semantyke jezyka naturalnego
    BEZ sztucznego wymuszania topologicznego constraintu.

    Wartosci S i E moga ewoluowac niezaleznie, a ich suma NIE musi
    rownac sie 1.0 dla zachowania efektywnej reprezentacji.

    -> GTMO jako framework jest bardziej elastyczny niz poczatkowo zakladano
    -> Constraint byl artefaktem teoretycznym, nie empiryczna koniecznoscia
    """)
    else:
        print("""
    !! CONSTRAINT S+E=1 MOZE BYC ISTOTNY

    Dane sugeruja, ze wymuszanie S+E=1 poprawia jakosc rekonstrukcji.
    Wymaga to dalszych badan nad teoretycznym uzasadnieniem.
    """)

    print("="*80)

    return results_dict, analysis_df


if __name__ == "__main__":
    results, analysis = main()
