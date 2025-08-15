
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.decomposition import PCA

# 1. Load dataset
df = pd.read_csv("Mall_Customers.csv")

print("\n=== Dataset Info ===")
print(f"Rows: {df.shape[0]}, Columns: {df.shape[1]}")
print("Columns:", list(df.columns))
print(df.head(), "\n")

# 2. Select features for clustering
features = ['Age', 'Annual Income (k$)', 'Spending Score (1-100)']
X = df[features]

# 3. Standardize
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# 4. Find best K
k_values = range(2, 11)
results = []
for k in k_values:
    km = KMeans(n_clusters=k, n_init=10, random_state=42)
    labels = km.fit_predict(X_scaled)
    inertia = km.inertia_
    sil = silhouette_score(X_scaled, labels)
    results.append({"K": k, "Inertia": inertia, "Silhouette": sil})

results_df = pd.DataFrame(results)
print("=== K vs Inertia vs Silhouette ===")
print(results_df.round(4), "\n")

# Choose K by max silhouette
best_row = results_df.loc[results_df['Silhouette'].idxmax()]
best_k = int(best_row['K'])
print(f"Best K by Silhouette: {best_k}, Score: {best_row['Silhouette']:.4f}\n")

# 5. Fit final model
final_km = KMeans(n_clusters=best_k, n_init=10, random_state=42)
final_labels = final_km.fit_predict(X_scaled)
df['Cluster'] = final_labels

# 6. Output cluster counts
counts = df['Cluster'].value_counts().sort_index()
print("=== Cluster Counts ===")
print(counts, "\n")

# 7. Centroids in original units
centroids_scaled = final_km.cluster_centers_
centroids_original = scaler.inverse_transform(centroids_scaled)
centroids_df = pd.DataFrame(centroids_original, columns=features)
centroids_df.insert(0, 'Cluster', range(best_k))
print("=== Cluster Centroids (Original Units) ===")
print(centroids_df.round(3), "\n")

# 8. Per-cluster means
numeric_cols = [col for col in df.select_dtypes(include=['number']).columns if col != 'Cluster']
profile_df = df.groupby('Cluster')[numeric_cols].mean().reset_index()
print("=== Per-Cluster Means ===")
print(profile_df.round(3), "\n")

# 9. Save labeled CSV
df.to_csv("Mall_Customers_with_clusters.csv", index=False)
print("Saved: Mall_Customers_with_clusters.csv\n")

# 10. Plots
# Elbow plot
plt.plot(results_df['K'], results_df['Inertia'], marker='o')
plt.title("Elbow Method (Inertia vs K)")
plt.xlabel("K")
plt.ylabel("Inertia (WCSS)")
plt.grid(True)
plt.savefig("elbow_kmeans_mall.png")
plt.close()

# Silhouette plot
plt.plot(results_df['K'], results_df['Silhouette'], marker='o')
plt.title("Silhouette Score vs K")
plt.xlabel("K")
plt.ylabel("Silhouette Score")
plt.grid(True)
plt.savefig("silhouette_vs_k_mall.png")
plt.close()

# PCA 2D scatter
pca = PCA(n_components=2, random_state=42)
X_pca = pca.fit_transform(X_scaled)
plt.scatter(X_pca[:,0], X_pca[:,1], c=final_labels, alpha=0.8)
plt.title(f"PCA 2D Clusters (K={best_k})")
plt.xlabel("PC1")
plt.ylabel("PC2")
plt.grid(True)
plt.savefig("clusters_pca_mall.png")
plt.close()

# Income vs Spending scatter
plt.scatter(df['Annual Income (k$)'], df['Spending Score (1-100)'], c=final_labels, alpha=0.8)
plt.title(f"Clusters on Income vs Spending (K={best_k})")
plt.xlabel("Annual Income (k$)")
plt.ylabel("Spending Score (1-100)")
plt.grid(True)
plt.savefig("clusters_income_spend_mall.png")
plt.close()
