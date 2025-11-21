#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
GTMØ HerBERT Analysis Tool
===========================

Kompleksowa analiza wyników HerBERT dla dokumentów prawnych.

Funkcjonalności:
1. Wykrywanie redundancji i powtórzeń (podobieństwo >0.99)
2. Klasteryzacja tematyczna
3. Analiza struktury prawnej (artykuły odstające, mosty)
4. Wizualizacje (heatmapa, graf sieci, dendrogram, t-SNE/UMAP)
5. Porównania międzydokumentowe
6. Analiza ewolucji dokumentów

Użycie:
    [opcje]

Przykład:
    python herbert_gtmo_analysis.py ../gtmo_results/analysis_xyz/full_document_herbert_analysis.json --all
    python herbert_gtmo_analysis.py <path_to_herbert_analysis.json> --redundancy
"""

import sys
import io
import json
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import argparse
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
from collections import defaultdict
import warnings
warnings.filterwarnings('ignore')

# Fix Windows console encoding for Unicode characters (emojis)
if sys.platform == 'win32':
    try:
        if not sys.stdout.closed:
            sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
        if not sys.stderr.closed:
            sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8')
    except (ValueError, AttributeError):
        # Skip if streams are already wrapped or closed
        pass

# Zaawansowane biblioteki do klasteryzacji i wizualizacji
try:
    from sklearn.cluster import AgglomerativeClustering, DBSCAN
    from sklearn.manifold import TSNE
    from scipy.cluster.hierarchy import dendrogram, linkage
    from scipy.spatial.distance import squareform
    import networkx as nx
    ADVANCED_LIBS = True
except ImportError:
    ADVANCED_LIBS = False
    print("⚠️  Niektóre zaawansowane funkcje niedostępne. Zainstaluj:")
    print("   pip install scikit-learn scipy networkx")

try:
    import umap
    UMAP_AVAILABLE = True
except ImportError:
    UMAP_AVAILABLE = False


@dataclass
class RedundancyPair:
    """Para redundantnych artykułów"""
    article_1: int
    article_2: int
    similarity: float

    def __str__(self):
        return f"Art.{self.article_1} ↔ Art.{self.article_2}: {self.similarity:.4f}"


@dataclass
class DocumentCluster:
    """Klaster tematyczny artykułów"""
    cluster_id: int
    articles: List[int]
    avg_similarity: float
    size: int

    def __str__(self):
        art_range = f"{min(self.articles)}-{max(self.articles)}" if len(self.articles) > 1 else str(self.articles[0])
        return f"Cluster #{self.cluster_id}: {self.size} artykułów [{art_range}], avg_sim={self.avg_similarity:.3f}"


class HerBERTAnalyzer:
    """Główna klasa analizy HerBERT"""

    def __init__(self, json_path: str):
        """
        Inicjalizacja analizatora

        Args:
            json_path: Ścieżka do pliku full_document_herbert_analysis.json
        """
        self.json_path = Path(json_path)
        self.output_dir = self.json_path.parent / "herbert_analysis_output"
        self.output_dir.mkdir(exist_ok=True)

        # Wczytaj dane
        with open(json_path, 'r', encoding='utf-8') as f:
            self.data = json.load(f)

        # Nie wszystkie pliki eksportu zawierają jawne pole article_count.
        # Jeśli go brakuje, ustalamy liczbę artykułów na podstawie rozmiaru macierzy podobieństw
        # (ew. liczby zdań, jeśli macierz nie jest dostępna).
        self.article_count = self.data.get('article_count')
        if self.article_count is None:
            if 'similarity_matrix' in self.data:
                self.article_count = len(self.data['similarity_matrix'])
            elif 'sentence_count' in self.data:
                self.article_count = self.data['sentence_count']
            else:
                raise KeyError("Brak pola 'article_count' oraz danych do jego wyliczenia w JSON")
        self.similarity_matrix = np.array(self.data['similarity_matrix'])
        self.avg_similarity = self.data['average_similarity']

        print(f"📂 Załadowano: {self.json_path.name}")
        print(f"📊 Artykułów: {self.article_count}")
        print(f"📈 Średnie podobieństwo: {self.avg_similarity:.4f}")
        print(f"💾 Wyniki zapisywane do: {self.output_dir}")


    # ========================================================================
    # 1. WYKRYWANIE REDUNDANCJI I POWTÓRZEŃ
    # ========================================================================

    def detect_redundancy(self, threshold: float = 0.99) -> List[RedundancyPair]:
        """
        Wykrywa pary artykułów o bardzo wysokim podobieństwie

        Args:
            threshold: Próg podobieństwa (domyślnie 0.99)

        Returns:
            Lista par redundantnych artykułów
        """
        print(f"\n🔍 Wykrywanie redundancji (próg={threshold})...")

        redundant_pairs = []

        for i in range(self.article_count):
            for j in range(i + 1, self.article_count):
                sim = self.similarity_matrix[i][j]
                if sim >= threshold:
                    redundant_pairs.append(RedundancyPair(i + 1, j + 1, sim))

        # Sortuj malejąco
        redundant_pairs.sort(key=lambda x: x.similarity, reverse=True)

        print(f"✓ Znaleziono {len(redundant_pairs)} par redundantnych")

        # Zapisz raport
        report_path = self.output_dir / f"redundancy_report_threshold_{threshold}.txt"
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write(f"RAPORT REDUNDANCJI - próg {threshold}\n")
            f.write("=" * 60 + "\n\n")
            f.write(f"Znaleziono {len(redundant_pairs)} par artykułów o podobieństwie ≥{threshold}\n\n")

            for pair in redundant_pairs:
                f.write(f"{pair}\n")

            # Analiza grup redundantnych
            f.write("\n" + "=" * 60 + "\n")
            f.write("GRUPY REDUNDANTNE (artykuły powiązane ze sobą):\n\n")

            groups = self._find_redundancy_groups(redundant_pairs)
            for i, group in enumerate(groups, 1):
                f.write(f"Grupa #{i}: Art. {', '.join(map(str, sorted(group)))}\n")

        print(f"💾 Raport zapisany: {report_path.name}")
        return redundant_pairs


    def _find_redundancy_groups(self, pairs: List[RedundancyPair]) -> List[set]:
        """Znajduje grupy wzajemnie podobnych artykułów"""
        # Graf nieskierowany
        graph = defaultdict(set)
        for pair in pairs:
            graph[pair.article_1].add(pair.article_2)
            graph[pair.article_2].add(pair.article_1)

        # Znajdź spójne składowe
        visited = set()
        groups = []

        def dfs(node, group):
            visited.add(node)
            group.add(node)
            for neighbor in graph[node]:
                if neighbor not in visited:
                    dfs(neighbor, group)

        for node in graph:
            if node not in visited:
                group = set()
                dfs(node, group)
                groups.append(group)

        return groups


    # ========================================================================
    # 2. KLASTERYZACJA TEMATYCZNA
    # ========================================================================

    def cluster_articles(self, n_clusters: int = 10, method: str = 'hierarchical') -> List[DocumentCluster]:
        """
        Klasteryzacja artykułów na grupy tematyczne

        Args:
            n_clusters: Liczba klastrów
            method: Metoda ('hierarchical', 'dbscan')

        Returns:
            Lista klastrów
        """
        if not ADVANCED_LIBS:
            print("❌ Klasteryzacja wymaga sklearn. Zainstaluj: pip install scikit-learn scipy")
            return []

        print(f"\n🎯 Klasteryzacja tematyczna ({method}, {n_clusters} klastrów)...")

        # Przekształć podobieństwo na dystans
        distance_matrix = 1 - self.similarity_matrix

        if method == 'hierarchical':
            clustering = AgglomerativeClustering(
                n_clusters=n_clusters,
                metric='precomputed',
                linkage='average'
            )
            labels = clustering.fit_predict(distance_matrix)

        elif method == 'dbscan':
            clustering = DBSCAN(eps=0.05, min_samples=3, metric='precomputed')
            labels = clustering.fit_predict(distance_matrix)
            n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
            print(f"   DBSCAN znalazł {n_clusters} klastrów")

        else:
            raise ValueError(f"Nieznana metoda: {method}")

        # Utwórz obiekty klastrów
        clusters = []
        for cluster_id in range(n_clusters):
            article_ids = [i + 1 for i, label in enumerate(labels) if label == cluster_id]

            if not article_ids:
                continue

            # Oblicz średnie podobieństwo wewnątrz klastra
            cluster_indices = [i for i, label in enumerate(labels) if label == cluster_id]
            if len(cluster_indices) > 1:
                cluster_sims = []
                for i in cluster_indices:
                    for j in cluster_indices:
                        if i < j:
                            cluster_sims.append(self.similarity_matrix[i][j])
                avg_sim = np.mean(cluster_sims) if cluster_sims else 1.0
            else:
                avg_sim = 1.0

            clusters.append(DocumentCluster(
                cluster_id=cluster_id,
                articles=article_ids,
                avg_similarity=avg_sim,
                size=len(article_ids)
            ))

        # Sortuj według rozmiaru
        clusters.sort(key=lambda x: x.size, reverse=True)

        print(f"✓ Utworzono {len(clusters)} klastrów")

        # Zapisz raport
        report_path = self.output_dir / f"clusters_{method}_{n_clusters}.txt"
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write(f"RAPORT KLASTERYZACJI - {method.upper()}\n")
            f.write("=" * 60 + "\n\n")
            f.write(f"Liczba klastrów: {len(clusters)}\n")
            f.write(f"Liczba artykułów: {self.article_count}\n\n")

            for cluster in clusters:
                f.write(f"{cluster}\n")
                f.write(f"  Artykuły: {', '.join(map(str, cluster.articles[:20]))}")
                if len(cluster.articles) > 20:
                    f.write(f" ... (+{len(cluster.articles) - 20} więcej)")
                f.write("\n\n")

        print(f"💾 Raport zapisany: {report_path.name}")
        return clusters


    # ========================================================================
    # 3. ANALIZA STRUKTURY PRAWNEJ
    # ========================================================================

    def analyze_structure(self) -> Dict:
        """
        Analiza struktury prawnej dokumentu:
        - Artykuły odstające (niskie podobieństwo)
        - Artykuły-mosty (łączniki między sekcjami)
        - Spójność rozdziałów
        """
        print(f"\n⚖️  Analiza struktury prawnej...")

        results = {}

        # 1. Artykuły odstające (outliers)
        avg_sims = np.mean(self.similarity_matrix, axis=1)
        threshold_low = np.percentile(avg_sims, 10)  # Dolne 10%
        outliers = [(i + 1, avg_sims[i]) for i in range(self.article_count) if avg_sims[i] < threshold_low]
        outliers.sort(key=lambda x: x[1])

        results['outliers'] = outliers
        print(f"   Artykuły odstające: {len(outliers)}")

        # 2. Artykuły-mosty (wysokie podobieństwo do wielu różnych grup)
        # Używamy odchylenia standardowego: niskie std = spójny z jedną grupą, wysokie = most
        std_sims = np.std(self.similarity_matrix, axis=1)
        threshold_high = np.percentile(std_sims, 90)  # Górne 10%
        bridges = [(i + 1, std_sims[i]) for i in range(self.article_count) if std_sims[i] > threshold_high]
        bridges.sort(key=lambda x: x[1], reverse=True)

        results['bridges'] = bridges
        print(f"   Artykuły-mosty: {len(bridges)}")

        # 3. Spójność dokumentu (cohesion)
        cohesion = self.avg_similarity
        results['cohesion'] = cohesion

        # Zapisz raport
        report_path = self.output_dir / "structure_analysis.txt"
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write("ANALIZA STRUKTURY PRAWNEJ\n")
            f.write("=" * 60 + "\n\n")

            f.write(f"Spójność dokumentu: {cohesion:.4f}\n\n")

            f.write("ARTYKUŁY ODSTAJĄCE (niskie średnie podobieństwo):\n")
            f.write("-" * 60 + "\n")
            for art, sim in outliers[:20]:
                f.write(f"Art.{art}: avg_sim={sim:.4f}\n")
            f.write(f"\n(Wyświetlono top 20 z {len(outliers)})\n\n")

            f.write("ARTYKUŁY-MOSTY (wysokie zróżnicowanie podobieństwa):\n")
            f.write("-" * 60 + "\n")
            for art, std in bridges[:20]:
                f.write(f"Art.{art}: std={std:.4f}\n")
            f.write(f"\n(Wyświetlono top 20 z {len(bridges)})\n")

        print(f"💾 Raport zapisany: {report_path.name}")
        return results


    # ========================================================================
    # 4. WIZUALIZACJE
    # ========================================================================

    def visualize_heatmap(self, figsize: Tuple[int, int] = (14, 12)):
        """Heatmapa podobieństw"""
        print(f"\n🎨 Generowanie heatmapy...")

        plt.figure(figsize=figsize)
        sns.heatmap(
            self.similarity_matrix,
            cmap='YlOrRd',
            vmin=0.9,
            vmax=1.0,
            xticklabels=50,
            yticklabels=50,
            cbar_kws={'label': 'Podobieństwo HerBERT'}
        )
        plt.title(f'Heatmapa podobieństw semantycznych ({self.article_count} artykułów)\nŚrednie podobieństwo: {self.avg_similarity:.4f}',
                  fontsize=14, pad=20)
        plt.xlabel('Numer artykułu', fontsize=12)
        plt.ylabel('Numer artykułu', fontsize=12)
        plt.tight_layout()

        output_path = self.output_dir / "heatmap.png"
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()

        print(f"✓ Heatmapa zapisana: {output_path.name}")


    def visualize_network(self, threshold: float = 0.98, max_edges: int = 500):
        """Graf sieci artykułów"""
        if not ADVANCED_LIBS:
            print("❌ Wizualizacja grafu wymaga networkx")
            return

        print(f"\n🕸️  Generowanie grafu sieci (próg={threshold})...")

        G = nx.Graph()

        # Dodaj węzły
        for i in range(self.article_count):
            G.add_node(i + 1)

        # Dodaj krawędzie (tylko powyżej progu)
        edges = []
        for i in range(self.article_count):
            for j in range(i + 1, self.article_count):
                sim = self.similarity_matrix[i][j]
                if sim >= threshold:
                    edges.append((i + 1, j + 1, sim))

        # Sortuj i ogranicz
        edges.sort(key=lambda x: x[2], reverse=True)
        edges = edges[:max_edges]

        for i, j, sim in edges:
            G.add_edge(i, j, weight=sim)

        print(f"   Graf: {G.number_of_nodes()} węzłów, {G.number_of_edges()} krawędzi")

        # Wizualizacja
        plt.figure(figsize=(16, 14))

        # Layout
        pos = nx.spring_layout(G, k=0.5, iterations=50, seed=42)

        # Rysuj
        nx.draw_networkx_nodes(G, pos, node_size=30, node_color='lightblue', alpha=0.7)
        nx.draw_networkx_edges(G, pos, alpha=0.3, width=0.5)

        # Etykiety tylko dla węzłów o wysokim stopniu
        degrees = dict(G.degree())
        high_degree_nodes = {node: node for node, deg in degrees.items() if deg > 5}
        nx.draw_networkx_labels(G, pos, labels=high_degree_nodes, font_size=8)

        plt.title(f'Sieć podobieństw artykułów (próg={threshold})\n{G.number_of_edges()} najsilniejszych połączeń',
                  fontsize=14, pad=20)
        plt.axis('off')
        plt.tight_layout()

        output_path = self.output_dir / f"network_graph_threshold_{threshold}.png"
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()

        print(f"✓ Graf zapisany: {output_path.name}")

        # Statystyki grafu
        stats_path = self.output_dir / f"network_stats_threshold_{threshold}.txt"
        with open(stats_path, 'w', encoding='utf-8') as f:
            f.write("STATYSTYKI GRAFU SIECI\n")
            f.write("=" * 60 + "\n\n")
            f.write(f"Próg podobieństwa: {threshold}\n")
            f.write(f"Węzły: {G.number_of_nodes()}\n")
            f.write(f"Krawędzie: {G.number_of_edges()}\n")
            f.write(f"Gęstość: {nx.density(G):.4f}\n")
            f.write(f"Średni stopień: {sum(dict(G.degree()).values()) / G.number_of_nodes():.2f}\n\n")

            # Top węzły według stopnia
            f.write("TOP 20 WĘZŁÓW (najwięcej połączeń):\n")
            f.write("-" * 60 + "\n")
            top_nodes = sorted(degrees.items(), key=lambda x: x[1], reverse=True)[:20]
            for node, deg in top_nodes:
                f.write(f"Art.{node}: {deg} połączeń\n")

        print(f"💾 Statystyki zapisane: {stats_path.name}")


    def visualize_dendrogram(self, figsize: Tuple[int, int] = (16, 10)):
        """Dendrogram hierarchiczny"""
        if not ADVANCED_LIBS:
            print("❌ Dendrogram wymaga scipy")
            return

        print(f"\n🌳 Generowanie dendrogramu...")

        # Przekształć macierz podobieństwa na dystans
        distance_matrix = 1 - self.similarity_matrix

        # Kondensuj do formy vectorized (wymagane przez linkage)
        condensed_dist = squareform(distance_matrix, checks=False)

        # Linkage
        linkage_matrix = linkage(condensed_dist, method='average')

        # Wizualizacja
        plt.figure(figsize=figsize)
        dendrogram(
            linkage_matrix,
            labels=[f"{i+1}" for i in range(self.article_count)],
            leaf_font_size=6,
            color_threshold=0.05
        )
        plt.title('Dendrogram hierarchiczny artykułów (average linkage)', fontsize=14, pad=20)
        plt.xlabel('Numer artykułu', fontsize=12)
        plt.ylabel('Dystans (1 - podobieństwo)', fontsize=12)
        plt.tight_layout()

        output_path = self.output_dir / "dendrogram.png"
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()

        print(f"✓ Dendrogram zapisany: {output_path.name}")


    def visualize_tsne(self, perplexity: int = 30, figsize: Tuple[int, int] = (12, 10)):
        """Projekcja t-SNE 2D"""
        if not ADVANCED_LIBS:
            print("❌ t-SNE wymaga sklearn")
            return

        print(f"\n🎯 Generowanie projekcji t-SNE (perplexity={perplexity})...")

        # t-SNE na macierzy dystansów
        distance_matrix = 1 - self.similarity_matrix

        tsne = TSNE(
            n_components=2,
            perplexity=perplexity,
            metric='precomputed',
            init='random',
            random_state=42,
            max_iter=1000
        )

        embedding = tsne.fit_transform(distance_matrix)

        # Wizualizacja
        plt.figure(figsize=figsize)
        scatter = plt.scatter(
            embedding[:, 0],
            embedding[:, 1],
            c=range(self.article_count),
            cmap='viridis',
            alpha=0.6,
            s=50
        )

        # Dodaj etykiety dla co 10-tego artykułu
        for i in range(0, self.article_count, 10):
            plt.annotate(
                f"{i+1}",
                (embedding[i, 0], embedding[i, 1]),
                fontsize=8,
                alpha=0.7
            )

        plt.colorbar(scatter, label='Numer artykułu')
        plt.title(f't-SNE: Projekcja 2D struktury semantycznej ({self.article_count} artykułów)',
                  fontsize=14, pad=20)
        plt.xlabel('t-SNE wymiar 1', fontsize=12)
        plt.ylabel('t-SNE wymiar 2', fontsize=12)
        plt.tight_layout()

        output_path = self.output_dir / f"tsne_perplexity_{perplexity}.png"
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()

        print(f"✓ t-SNE zapisany: {output_path.name}")


    def visualize_umap(self, n_neighbors: int = 15, figsize: Tuple[int, int] = (12, 10)):
        """Projekcja UMAP 2D"""
        if not UMAP_AVAILABLE:
            print("⚠️  UMAP niedostępny. Zainstaluj: pip install umap-learn")
            return

        print(f"\n🎯 Generowanie projekcji UMAP (n_neighbors={n_neighbors})...")

        # UMAP na macierzy dystansów
        distance_matrix = 1 - self.similarity_matrix

        reducer = umap.UMAP(
            n_components=2,
            n_neighbors=n_neighbors,
            metric='precomputed',
            random_state=42
        )

        embedding = reducer.fit_transform(distance_matrix)

        # Wizualizacja
        plt.figure(figsize=figsize)
        scatter = plt.scatter(
            embedding[:, 0],
            embedding[:, 1],
            c=range(self.article_count),
            cmap='viridis',
            alpha=0.6,
            s=50
        )

        # Dodaj etykiety dla co 10-tego artykułu
        for i in range(0, self.article_count, 10):
            plt.annotate(
                f"{i+1}",
                (embedding[i, 0], embedding[i, 1]),
                fontsize=8,
                alpha=0.7
            )

        plt.colorbar(scatter, label='Numer artykułu')
        plt.title(f'UMAP: Projekcja 2D struktury semantycznej ({self.article_count} artykułów)',
                  fontsize=14, pad=20)
        plt.xlabel('UMAP wymiar 1', fontsize=12)
        plt.ylabel('UMAP wymiar 2', fontsize=12)
        plt.tight_layout()

        output_path = self.output_dir / f"umap_neighbors_{n_neighbors}.png"
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()

        print(f"✓ UMAP zapisany: {output_path.name}")


    # ========================================================================
    # 5. PORÓWNANIA MIĘDZYDOKUMENTOWE
    # ========================================================================

    def compare_documents(self, other_json_path: str) -> Dict:
        """
        Porównanie struktury z innym dokumentem

        Args:
            other_json_path: Ścieżka do innego pliku herbert_analysis.json

        Returns:
            Słownik z wynikami porównania
        """
        print(f"\n🔬 Porównanie międzydokumentowe...")

        # Wczytaj drugi dokument
        with open(other_json_path, 'r', encoding='utf-8') as f:
            other_data = json.load(f)

        other_name = Path(other_json_path).parent.name

        results = {
            'doc1_name': self.json_path.parent.name,
            'doc2_name': other_name,
            'doc1_articles': self.article_count,
            'doc2_articles': other_data['article_count'],
            'doc1_avg_sim': self.avg_similarity,
            'doc2_avg_sim': other_data['average_similarity'],
            'similarity_diff': abs(self.avg_similarity - other_data['average_similarity'])
        }

        # Analiza rozkładów podobieństwa
        doc1_sims = self.similarity_matrix[np.triu_indices_from(self.similarity_matrix, k=1)]
        doc2_sims = np.array(other_data['similarity_matrix'])
        doc2_sims = doc2_sims[np.triu_indices_from(doc2_sims, k=1)]

        results['doc1_std'] = float(np.std(doc1_sims))
        results['doc2_std'] = float(np.std(doc2_sims))

        # Wizualizacja porównawcza
        fig, axes = plt.subplots(1, 2, figsize=(16, 6))

        axes[0].hist(doc1_sims, bins=50, alpha=0.7, color='blue', edgecolor='black')
        axes[0].axvline(self.avg_similarity, color='red', linestyle='--', linewidth=2, label=f'Średnia: {self.avg_similarity:.4f}')
        axes[0].set_title(f'{results["doc1_name"]}\n{self.article_count} artykułów', fontsize=12)
        axes[0].set_xlabel('Podobieństwo', fontsize=10)
        axes[0].set_ylabel('Liczba par', fontsize=10)
        axes[0].legend()
        axes[0].grid(alpha=0.3)

        axes[1].hist(doc2_sims, bins=50, alpha=0.7, color='green', edgecolor='black')
        axes[1].axvline(other_data['average_similarity'], color='red', linestyle='--', linewidth=2,
                       label=f'Średnia: {other_data["average_similarity"]:.4f}')
        axes[1].set_title(f'{results["doc2_name"]}\n{other_data["article_count"]} artykułów', fontsize=12)
        axes[1].set_xlabel('Podobieństwo', fontsize=10)
        axes[1].set_ylabel('Liczba par', fontsize=10)
        axes[1].legend()
        axes[1].grid(alpha=0.3)

        plt.suptitle('Porównanie rozkładów podobieństw', fontsize=14, y=1.02)
        plt.tight_layout()

        output_path = self.output_dir / f"comparison_{other_name}.png"
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()

        # Raport
        report_path = self.output_dir / f"comparison_{other_name}.txt"
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write("PORÓWNANIE MIĘDZYDOKUMENTOWE\n")
            f.write("=" * 60 + "\n\n")
            f.write(f"Dokument 1: {results['doc1_name']}\n")
            f.write(f"  Artykułów: {results['doc1_articles']}\n")
            f.write(f"  Średnie podobieństwo: {results['doc1_avg_sim']:.4f}\n")
            f.write(f"  Odchylenie standardowe: {results['doc1_std']:.4f}\n\n")

            f.write(f"Dokument 2: {results['doc2_name']}\n")
            f.write(f"  Artykułów: {results['doc2_articles']}\n")
            f.write(f"  Średnie podobieństwo: {results['doc2_avg_sim']:.4f}\n")
            f.write(f"  Odchylenie standardowe: {results['doc2_std']:.4f}\n\n")

            f.write("RÓŻNICE:\n")
            f.write(f"  Δ średnie podobieństwo: {results['similarity_diff']:.4f}\n")
            f.write(f"  Δ odchylenie std: {abs(results['doc1_std'] - results['doc2_std']):.4f}\n")

        print(f"✓ Porównanie zapisane: {report_path.name}")
        return results


    # ========================================================================
    # 6. ANALIZA EWOLUCJI
    # ========================================================================

    def analyze_evolution(self, historical_versions: List[str]):
        """
        Analiza ewolucji dokumentu przez wiele wersji

        Args:
            historical_versions: Lista ścieżek do plików herbert_analysis.json w kolejności chronologicznej
        """
        print(f"\n📈 Analiza ewolucji ({len(historical_versions)} wersji)...")

        # Wczytaj wszystkie wersje
        versions_data = []
        for path in historical_versions:
            with open(path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            versions_data.append({
                'name': Path(path).parent.name,
                'path': path,
                'article_count': data['article_count'],
                'avg_similarity': data['average_similarity'],
                'similarity_matrix': np.array(data['similarity_matrix'])
            })

        # Analiza zmian w czasie
        fig, axes = plt.subplots(2, 1, figsize=(14, 10))

        # Wykres 1: Średnie podobieństwo w czasie
        avg_sims = [v['avg_similarity'] for v in versions_data]
        axes[0].plot(range(len(versions_data)), avg_sims, marker='o', linewidth=2, markersize=8)
        axes[0].set_xlabel('Wersja', fontsize=12)
        axes[0].set_ylabel('Średnie podobieństwo', fontsize=12)
        axes[0].set_title('Ewolucja średniego podobieństwa', fontsize=14)
        axes[0].grid(alpha=0.3)
        axes[0].set_xticks(range(len(versions_data)))
        axes[0].set_xticklabels([v['name'][:30] for v in versions_data], rotation=45, ha='right')

        # Wykres 2: Liczba artykułów
        article_counts = [v['article_count'] for v in versions_data]
        axes[1].bar(range(len(versions_data)), article_counts, alpha=0.7, color='steelblue')
        axes[1].set_xlabel('Wersja', fontsize=12)
        axes[1].set_ylabel('Liczba artykułów', fontsize=12)
        axes[1].set_title('Ewolucja liczby artykułów', fontsize=14)
        axes[1].grid(alpha=0.3, axis='y')
        axes[1].set_xticks(range(len(versions_data)))
        axes[1].set_xticklabels([v['name'][:30] for v in versions_data], rotation=45, ha='right')

        plt.tight_layout()

        output_path = self.output_dir / "evolution_analysis.png"
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()

        # Raport
        report_path = self.output_dir / "evolution_analysis.txt"
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write("ANALIZA EWOLUCJI DOKUMENTU\n")
            f.write("=" * 60 + "\n\n")
            f.write(f"Liczba wersji: {len(versions_data)}\n\n")

            for i, version in enumerate(versions_data, 1):
                f.write(f"Wersja #{i}: {version['name']}\n")
                f.write(f"  Artykułów: {version['article_count']}\n")
                f.write(f"  Średnie podobieństwo: {version['avg_similarity']:.4f}\n")

                if i > 1:
                    prev = versions_data[i - 2]
                    delta_articles = version['article_count'] - prev['article_count']
                    delta_sim = version['avg_similarity'] - prev['avg_similarity']
                    f.write(f"  Δ artykuły: {delta_articles:+d}\n")
                    f.write(f"  Δ podobieństwo: {delta_sim:+.4f}\n")

                f.write("\n")

        print(f"✓ Analiza ewolucji zapisana: {report_path.name}")


    # ========================================================================
    # FUNKCJE POMOCNICZE
    # ========================================================================

    def generate_summary_report(self):
        """Generuje kompletny raport podsumowujący"""
        print(f"\n📋 Generowanie raportu podsumowującego...")

        report_path = self.output_dir / "SUMMARY_REPORT.txt"

        with open(report_path, 'w', encoding='utf-8') as f:
            f.write("=" * 70 + "\n")
            f.write("GTMØ HerBERT ANALYSIS - RAPORT PODSUMOWUJĄCY\n")
            f.write("=" * 70 + "\n\n")

            f.write(f"Dokument: {self.json_path.parent.name}\n")
            f.write(f"Plik źródłowy: {self.json_path.name}\n")
            f.write(f"Data analizy: {Path(report_path).stat().st_mtime}\n\n")

            f.write("PODSTAWOWE STATYSTYKI:\n")
            f.write("-" * 70 + "\n")
            f.write(f"Liczba artykułów: {self.article_count}\n")
            f.write(f"Średnie podobieństwo: {self.avg_similarity:.4f}\n")
            f.write(f"Min podobieństwo: {np.min(self.similarity_matrix[np.triu_indices_from(self.similarity_matrix, k=1)]):.4f}\n")
            f.write(f"Max podobieństwo: {np.max(self.similarity_matrix[np.triu_indices_from(self.similarity_matrix, k=1)]):.4f}\n")

            sims = self.similarity_matrix[np.triu_indices_from(self.similarity_matrix, k=1)]
            f.write(f"Mediana podobieństwa: {np.median(sims):.4f}\n")
            f.write(f"Odchylenie standardowe: {np.std(sims):.4f}\n\n")

            f.write("WYGENEROWANE ANALIZY:\n")
            f.write("-" * 70 + "\n")
            for file in sorted(self.output_dir.iterdir()):
                if file.is_file() and file.name != "SUMMARY_REPORT.txt":
                    f.write(f"  • {file.name}\n")

            f.write("\n" + "=" * 70 + "\n")

        print(f"✓ Raport podsumowujący: {report_path.name}")
        print(f"\n✅ Wszystkie wyniki zapisane w: {self.output_dir}")


# ============================================================================
# CLI
# ============================================================================

def main():
    parser = argparse.ArgumentParser(
        description='GTMØ HerBERT Analysis - Kompleksowa analiza wyników HerBERT',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Przykłady użycia:

  # Pełna analiza (wszystkie funkcje)
  python herbert_gtmo_analysis.py path/to/full_document_herbert_analysis.json --all

  # Tylko redundancja
  python herbert_gtmo_analysis.py path/to/file.json --redundancy

  # Klasteryzacja + wizualizacje
  python herbert_gtmo_analysis.py path/to/file.json --cluster --visualize

  # Porównanie dwóch dokumentów
  python herbert_gtmo_analysis.py doc1.json --compare doc2.json

  # Analiza ewolucji
  python herbert_gtmo_analysis.py latest.json --evolution v1.json v2.json v3.json
        """
    )

    parser.add_argument('json_file', help='Ścieżka do pliku full_document_herbert_analysis.json')

    # Opcje analiz
    parser.add_argument('--all', action='store_true', help='Uruchom wszystkie analizy')
    parser.add_argument('--redundancy', action='store_true', help='Wykrywanie redundancji')
    parser.add_argument('--redundancy-threshold', type=float, default=0.99, help='Próg redundancji (default: 0.99)')
    parser.add_argument('--cluster', action='store_true', help='Klasteryzacja tematyczna')
    parser.add_argument('--n-clusters', type=int, default=10, help='Liczba klastrów (default: 10)')
    parser.add_argument('--cluster-method', choices=['hierarchical', 'dbscan'], default='hierarchical', help='Metoda klasteryzacji')
    parser.add_argument('--structure', action='store_true', help='Analiza struktury prawnej')
    parser.add_argument('--visualize', action='store_true', help='Wszystkie wizualizacje')
    parser.add_argument('--heatmap', action='store_true', help='Tylko heatmapa')
    parser.add_argument('--network', action='store_true', help='Tylko graf sieci')
    parser.add_argument('--network-threshold', type=float, default=0.98, help='Próg dla grafu (default: 0.98)')
    parser.add_argument('--dendrogram', action='store_true', help='Tylko dendrogram')
    parser.add_argument('--tsne', action='store_true', help='Tylko t-SNE')
    parser.add_argument('--umap', action='store_true', help='Tylko UMAP')
    parser.add_argument('--compare', type=str, help='Porównaj z innym dokumentem (ścieżka do JSON)')
    parser.add_argument('--evolution', nargs='+', help='Analiza ewolucji (lista ścieżek do wersji historycznych)')

    args = parser.parse_args()

    # Inicjalizacja
    print("=" * 70)
    print("GTMO HerBERT ANALYSIS TOOL")
    print("=" * 70)

    analyzer = HerBERTAnalyzer(args.json_file)

    # Wykonaj analizy
    if args.all:
        # Wszystko
        analyzer.detect_redundancy(threshold=args.redundancy_threshold)
        analyzer.cluster_articles(n_clusters=args.n_clusters, method=args.cluster_method)
        analyzer.analyze_structure()
        analyzer.visualize_heatmap()
        analyzer.visualize_network(threshold=args.network_threshold)
        analyzer.visualize_dendrogram()
        analyzer.visualize_tsne()
        analyzer.visualize_umap()
    else:
        # Selektywnie
        if args.redundancy:
            analyzer.detect_redundancy(threshold=args.redundancy_threshold)

        if args.cluster:
            analyzer.cluster_articles(n_clusters=args.n_clusters, method=args.cluster_method)

        if args.structure:
            analyzer.analyze_structure()

        if args.visualize or args.heatmap:
            analyzer.visualize_heatmap()

        if args.visualize or args.network:
            analyzer.visualize_network(threshold=args.network_threshold)

        if args.visualize or args.dendrogram:
            analyzer.visualize_dendrogram()

        if args.visualize or args.tsne:
            analyzer.visualize_tsne()

        if args.visualize or args.umap:
            analyzer.visualize_umap()

    # Porównania i ewolucja
    if args.compare:
        analyzer.compare_documents(args.compare)

    if args.evolution:
        analyzer.analyze_evolution(args.evolution)

    # Raport końcowy
    analyzer.generate_summary_report()

    print("\n" + "=" * 70)
    print("✅ ANALIZA ZAKOŃCZONA")
    print("=" * 70)


if __name__ == "__main__":
    main()
