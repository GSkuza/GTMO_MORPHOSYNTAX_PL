import pandas as pd
import numpy as np
import json
import re
from scipy import stats
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns

# Set style
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")

# Read GTMØ data
print("Loading GTMØ data...")
gtmo_df = pd.read_csv(r"D:\GTMO_MORPHOSYNTAX\gtmo_results\analysis_15112025_no1_RP_KONSTYTUCJA_09112025\verdict_analysis_output\gtmo_full_analysis.csv")

# Extract article numbers
gtmo_df['article_num'] = gtmo_df['text'].str.extract(r'Art\.\s*(\d+)', expand=False).astype(float)

# Group by article and get mean SA
gtmo_articles = gtmo_df.groupby('article_num').agg({
    'SA': 'mean',
    'depth': 'mean',
    'CI': 'mean',
    'CD': 'mean',
    'ambiguity': 'mean',
    'D': 'mean',
    'S': 'mean',
    'E': 'mean'
}).reset_index()

print(f"GTMØ: {len(gtmo_articles)} articles loaded")

# Read HerBERT data - network stats
print("\nLoading HerBERT network data...")
with open(r"D:\GTMO_MORPHOSYNTAX\gtmo_results\analysis_15112025_no1_RP_KONSTYTUCJA_09112025\herbert_analysis_output\network_stats_threshold_0.98.txt", 'r', encoding='utf-8') as f:
    content = f.read()

# Extract connections per article
connections = {}
for line in content.split('\n'):
    match = re.search(r'Art\.(\d+):\s*(\d+)', line)
    if match:
        art_num = int(match.group(1))
        conn_count = int(match.group(2))
        connections[art_num] = conn_count

print(f"HerBERT connections: {len(connections)} articles")

# Read HerBERT similarity matrix from JSON
print("\nLoading HerBERT similarity data...")
with open(r"D:\GTMO_MORPHOSYNTAX\gtmo_results\analysis_15112025_no1_RP_KONSTYTUCJA_09112025\full_document_herbert_analysis.json", 'r', encoding='utf-8') as f:
    herbert_data = json.load(f)

# Calculate average similarity for each article
avg_similarities = {}
for article in herbert_data['articles']:
    art_id = article['article_id']
    similarities = article['similarities']
    # Calculate average similarity (excluding self-similarity of 1.0)
    sim_values = [s for s in similarities.values() if s < 0.9999]
    if sim_values:
        avg_similarities[art_id] = np.mean(sim_values)

print(f"HerBERT similarities: {len(avg_similarities)} articles")

# Read clusters
print("\nLoading HerBERT cluster data...")
with open(r"D:\GTMO_MORPHOSYNTAX\gtmo_results\analysis_15112025_no1_RP_KONSTYTUCJA_09112025\herbert_analysis_output\clusters_hierarchical_10.txt", 'r', encoding='utf-8') as f:
    cluster_content = f.read()

# Parse clusters
clusters = {}
current_cluster = None
for line in cluster_content.split('\n'):
    cluster_match = re.search(r'Cluster #(\d+):', line)
    if cluster_match:
        current_cluster = int(cluster_match.group(1))
    articles_match = re.search(r'Artykuły:\s*(.+)', line)
    if articles_match and current_cluster is not None:
        art_nums_str = articles_match.group(1).replace('...', '').split('(+')[0]
        art_nums = [int(x.strip()) for x in art_nums_str.split(',') if x.strip().isdigit()]
        for art in art_nums:
            clusters[art] = current_cluster

print(f"HerBERT clusters: {len(clusters)} articles assigned")

# Merge all data
print("\nMerging datasets...")
merged_data = []
for _, row in gtmo_articles.iterrows():
    art_num = int(row['article_num'])
    if art_num in avg_similarities:
        merged_data.append({
            'article': art_num,
            'SA': row['SA'],
            'depth': row['depth'],
            'CI': row['CI'],
            'CD': row['CD'],
            'ambiguity': row['ambiguity'],
            'D': row['D'],
            'S': row['S'],
            'E': row['E'],
            'herbert_avg_similarity': avg_similarities.get(art_num, np.nan),
            'herbert_connections': connections.get(art_num, 0),
            'herbert_cluster': clusters.get(art_num, -1)
        })

df = pd.DataFrame(merged_data)
print(f"Merged data: {len(df)} articles")

# Calculate correlations
print("\n" + "="*70)
print("PEARSON CORRELATION ANALYSIS: HerBERT vs GTMØ SA")
print("="*70)

# 1. Connections vs SA
corr_conn_sa, p_conn_sa = stats.pearsonr(df['herbert_connections'], df['SA'])
print(f"\n1. HerBERT Connections vs SA:")
print(f"   Pearson r = {corr_conn_sa:.4f}")
print(f"   p-value   = {p_conn_sa:.6f}")
print(f"   Interpretation: {'SIGNIFICANT' if p_conn_sa < 0.05 else 'NOT significant'}")
if corr_conn_sa < 0:
    print(f"   → Więcej połączeń HerBERT = Niższa dostępność SA")
else:
    print(f"   → Więcej połączeń HerBERT = Wyższa dostępność SA")

# 2. Average similarity vs SA
corr_sim_sa, p_sim_sa = stats.pearsonr(df['herbert_avg_similarity'], df['SA'])
print(f"\n2. HerBERT Average Similarity vs SA:")
print(f"   Pearson r = {corr_sim_sa:.4f}")
print(f"   p-value   = {p_sim_sa:.6f}")
print(f"   Interpretation: {'SIGNIFICANT' if p_sim_sa < 0.05 else 'NOT significant'}")
if corr_sim_sa < 0:
    print(f"   → Wyższe podobieństwo HerBERT = Niższa dostępność SA")
else:
    print(f"   → Wyższe podobieństwo HerBERT = Wyższa dostępność SA")

# 3. Cluster membership vs SA
cluster_sa_means = df.groupby('herbert_cluster')['SA'].mean()
print(f"\n3. HerBERT Cluster Membership vs SA:")
print(f"   Number of clusters: {len(cluster_sa_means)}")
print(f"\n   Average SA per cluster:")
for cluster_id in sorted(cluster_sa_means.index):
    if cluster_id >= 0:
        cluster_articles = df[df['herbert_cluster'] == cluster_id]
        print(f"   Cluster #{cluster_id}: SA mean = {cluster_sa_means[cluster_id]:.4f} ({len(cluster_articles)} articles)")

# ANOVA test for clusters
clusters_valid = df[df['herbert_cluster'] >= 0]
cluster_groups = [clusters_valid[clusters_valid['herbert_cluster'] == c]['SA'].values
                  for c in clusters_valid['herbert_cluster'].unique()]
if len(cluster_groups) > 1:
    f_stat, p_anova = stats.f_oneway(*cluster_groups)
    print(f"\n   ANOVA F-statistic = {f_stat:.4f}")
    print(f"   ANOVA p-value     = {p_anova:.6f}")
    print(f"   Interpretation: {'SIGNIFICANT cluster differences' if p_anova < 0.05 else 'NO significant cluster differences'}")
else:
    p_anova = 1.0
    print("\n   Not enough clusters for ANOVA")

# Additional correlations
print(f"\n4. Additional Correlations with SA:")
print(f"   SA vs Depth:      r = {stats.pearsonr(df['depth'], df['SA'])[0]:7.4f} (p = {stats.pearsonr(df['depth'], df['SA'])[1]:.6f})")
print(f"   SA vs CI:         r = {stats.pearsonr(df['CI'], df['SA'])[0]:7.4f} (p = {stats.pearsonr(df['CI'], df['SA'])[1]:.6f})")
print(f"   SA vs CD:         r = {stats.pearsonr(df['CD'], df['SA'])[0]:7.4f} (p = {stats.pearsonr(df['CD'], df['SA'])[1]:.6f})")
print(f"   SA vs Ambiguity:  r = {stats.pearsonr(df['ambiguity'], df['SA'])[0]:7.4f} (p = {stats.pearsonr(df['ambiguity'], df['SA'])[1]:.6f})")

# Cross-correlations
print(f"\n5. HerBERT vs GTMØ Structural Metrics:")
print(f"   Connections vs Depth:     r = {stats.pearsonr(df['herbert_connections'], df['depth'])[0]:7.4f}")
print(f"   Connections vs CI:        r = {stats.pearsonr(df['herbert_connections'], df['CI'])[0]:7.4f}")
print(f"   Similarity vs Depth:      r = {stats.pearsonr(df['herbert_avg_similarity'], df['depth'])[0]:7.4f}")
print(f"   Similarity vs CI:         r = {stats.pearsonr(df['herbert_avg_similarity'], df['CI'])[0]:7.4f}")

# Save statistics
output_dir = r"D:\GTMO_MORPHOSYNTAX\gtmo_results_analyse"
df.to_csv(f"{output_dir}\\herbert_gtmo_merged.csv", index=False)
print(f"\n✓ Merged data saved to herbert_gtmo_merged.csv")

# Summary statistics
print("\n" + "="*70)
print("SUMMARY STATISTICS")
print("="*70)
print(f"\nTop 10 Articles by SA (highest accessibility):")
print(df.nlargest(10, 'SA')[['article', 'SA', 'herbert_avg_similarity', 'herbert_connections', 'depth', 'CI']].to_string(index=False))

print(f"\nBottom 10 Articles by SA (lowest accessibility):")
print(df.nsmallest(10, 'SA')[['article', 'SA', 'herbert_avg_similarity', 'herbert_connections', 'depth', 'CI']].to_string(index=False))

print(f"\nTop 10 Most Connected Articles (HerBERT):")
print(df.nlargest(10, 'herbert_connections')[['article', 'herbert_connections', 'SA', 'depth', 'CI']].to_string(index=False))

print("\n" + "="*70)
print("ANALYSIS COMPLETE!")
print("="*70)
