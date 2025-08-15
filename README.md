# Mall Customers K-Means Clustering

This project implements **Task 8** from the Data Science learning module: performing **K-Means clustering** on the Mall Customers dataset to segment shoppers into distinct groups based on their **Age**, **Annual Income (k$)**, and **Spending Score (1–100)**.

The goal is to:
- Identify customer segments
- Visualize clusters using multiple plots
- Provide interpretable cluster profiles for marketing or business strategy

---

## 📂 Project Structure


.
├── Mall_Customers.csv # Input dataset
├── mall-data.py # Main Python script
├── Mall_Customers_with_clusters.csv # Output dataset with cluster labels
├── elbow_kmeans_mall.png # Elbow Method plot
├── silhouette_vs_k_mall.png # Silhouette Score vs K plot
├── clusters_pca_mall.png # PCA 2D cluster visualization
├── clusters_income_spend_mall.png # Income vs Spending cluster visualization
└── README.md # Project documentation


---

## 🚀 How to Run

### 1. Clone the repository
```bash
git clone https://github.com/<your-username>/mall-customers-kmeans-clustering.git
cd mall-customers-kmeans-clustering

2. Install dependencies

Make sure you have Python 3.x installed. Then install required libraries:

pip install pandas numpy scikit-learn matplotlib

3. Run the script
python mall-data.py

📊 Outputs
1. Console Output

Dataset info (rows, columns)

Table of K vs Inertia vs Silhouette

Best K selected (by silhouette score)

Cluster counts

Cluster centroids in original units

Per-cluster mean profiles

2. Files Generated

Mall_Customers_with_clusters.csv → original data + Cluster column

elbow_kmeans_mall.png → inertia vs K plot

silhouette_vs_k_mall.png → silhouette score vs K

clusters_pca_mall.png → PCA 2D projection colored by cluster

clusters_income_spend_mall.png → income vs spending score colored by cluster
