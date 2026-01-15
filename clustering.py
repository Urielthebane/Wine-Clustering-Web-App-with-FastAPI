# app/clustering.py
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_wine
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans, AgglomerativeClustering, DBSCAN
from sklearn.metrics import silhouette_score, davies_bouldin_score, calinski_harabasz_score, adjusted_rand_score

# Load and scale data
data = load_wine()
X = data.data
y = data.target

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Clustering algorithms
kmeans = KMeans(n_clusters=3, random_state=42)
agglo = AgglomerativeClustering(n_clusters=3)
dbscan = DBSCAN(eps=2, min_samples=2)

# Fit and predict
kmeans_labels = kmeans.fit_predict(X_scaled)
agglo_labels = agglo.fit_predict(X_scaled)
dbscan_labels = dbscan.fit_predict(X_scaled)

# Evaluation function
def evaluate_cluster(y_true, y_pred, X=X_scaled):
    return {
        "Silhouette": silhouette_score(X, y_pred),
        "Davies-Bouldin": davies_bouldin_score(X, y_pred),
        "Calinski-Harabasz": calinski_harabasz_score(X, y_pred),
        "Adjusted Rand Index": adjusted_rand_score(y_true, y_pred)
    }

# Plotting function
def plot_clusters(X, labels, title):
    plt.figure(figsize=(8,6))
    unique_labels = np.unique(labels)
    cmap = plt.cm.get_cmap("viridis", len(unique_labels))
    for idx, cluster in enumerate(unique_labels):
        mask = labels == cluster
        if cluster == -1:
            plt.scatter(X[mask,0], X[mask,1], c="red", label="Noise", s=50, alpha=0.6)
        else:
            plt.scatter(X[mask,0], X[mask,1], color=cmap(idx), label=f"Cluster {cluster}", s=50, alpha=0.6)
    plt.title(title)
    plt.xlabel("Feature 1")
    plt.ylabel("Feature 2")
    plt.legend()
    plt.show()
