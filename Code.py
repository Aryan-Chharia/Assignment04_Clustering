# Import necessary libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.cluster import KMeans, MeanShift, AgglomerativeClustering
from sklearn.metrics import silhouette_score, calinski_harabasz_score, davies_bouldin_score
import os

# Create output directory
output_dir = "clustering_outputs"
os.makedirs(output_dir, exist_ok=True)

# Load dataset from UCI (Iris dataset)
url = "https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data"
columns = ['sepal_length', 'sepal_width', 'petal_length', 'petal_width', 'species']
df = pd.read_csv(url, names=columns)
df.drop(columns=['species'], inplace=True)  # Remove labels

# Preprocessing Techniques
scaler_standard = StandardScaler()
scaler_minmax = MinMaxScaler()
pca = PCA(n_components=2)

df_standard = pd.DataFrame(scaler_standard.fit_transform(df), columns=df.columns)
df_minmax = pd.DataFrame(scaler_minmax.fit_transform(df), columns=df.columns)
df_pca = pd.DataFrame(pca.fit_transform(df), columns=['PC1', 'PC2'])
df_tn = pd.DataFrame(scaler_standard.fit_transform(df_minmax), columns=df.columns)  # Transform + Normalize
df_tn_pca = pd.DataFrame(pca.fit_transform(df_tn), columns=['PC1', 'PC2'])  # Transform + Normalize + PCA

# Define different preprocessing techniques
preprocessing_methods = {
    "No Data Processing": df,
    "Using Normalization": df_minmax,
    "Using Transform": df_standard,
    "Using PCA": df_pca,
    "Using T+N": df_tn,
    "T+N+PCA": df_tn_pca
}

# Function to apply clustering and compute metrics
def apply_clustering(data, n_clusters):
    results = []
    algorithms = {
        "K-Means": KMeans(n_clusters=n_clusters, random_state=42),
        "Hierarchical": AgglomerativeClustering(n_clusters=n_clusters),
        "Mean-Shift": MeanShift()
    }

    for algo_name, algorithm in algorithms.items():
        labels = algorithm.fit_predict(data)

        if len(set(labels)) > 1:  # Only compute metrics if multiple clusters exist
            silhouette = silhouette_score(data, labels)
            calinski_harabasz = calinski_harabasz_score(data, labels)
            davies_bouldin = davies_bouldin_score(data, labels)
        else:
            silhouette = calinski_harabasz = davies_bouldin = np.nan  # Not computable

        results.append([algo_name, silhouette, calinski_harabasz, davies_bouldin])

    return results

# Store final results
final_results = {"K-Means": [], "Hierarchical": [], "Mean-Shift": []}

# Apply clustering to all preprocessing methods and cluster sizes
for preprocess_name, dataset in preprocessing_methods.items():
    for clusters in [3, 4, 5]:
        results = apply_clustering(dataset, clusters)
        for row in results:
            final_results[row[0]].append([preprocess_name, clusters] + row[1:])

# Convert results to DataFrame
columns = ["Preprocessing", "Clusters", "Silhouette Score", "Calinski-Harabasz", "Davies-Bouldin"]
df_kmeans = pd.DataFrame(final_results["K-Means"], columns=columns)
df_hierarchical = pd.DataFrame(final_results["Hierarchical"], columns=columns)
df_meanshift = pd.DataFrame(final_results["Mean-Shift"], columns=columns)

# Save tables as PNG images
def save_table_as_image(df, title, filename):
    fig, ax = plt.subplots(figsize=(12, 4))
    ax.axis('tight')
    ax.axis('off')
    table = ax.table(cellText=df.values, colLabels=df.columns, cellLoc='center', loc='center')
    plt.title(title)
    plt.savefig(f"{output_dir}/{filename}")
    plt.close()

save_table_as_image(df_kmeans, "Using K-Means Clustering", "table_kmeans.png")
save_table_as_image(df_hierarchical, "Using Hierarchical Clustering", "table_hierarchical.png")
save_table_as_image(df_meanshift, "Using Mean-Shift Clustering", "table_meanshift.png")

# Save results as CSV
df_kmeans.to_csv(f"{output_dir}/results_kmeans.csv", index=False)
df_hierarchical.to_csv(f"{output_dir}/results_hierarchical.csv", index=False)
df_meanshift.to_csv(f"{output_dir}/results_meanshift.csv", index=False)

# Generate Heatmap of results
fig, ax = plt.subplots(figsize=(8, 6))
sns.heatmap(df_kmeans.drop(columns=["Preprocessing", "Clusters"]), annot=True, fmt=".3f", cmap="coolwarm", ax=ax)
plt.title("K-Means Performance Heatmap")
plt.savefig(f"{output_dir}/heatmap_kmeans.png")
plt.close()

fig, ax = plt.subplots(figsize=(8, 6))
sns.heatmap(df_hierarchical.drop(columns=["Preprocessing", "Clusters"]), annot=True, fmt=".3f", cmap="coolwarm", ax=ax)
plt.title("Hierarchical Clustering Heatmap")
plt.savefig(f"{output_dir}/heatmap_hierarchical.png")
plt.close()

fig, ax = plt.subplots(figsize=(8, 6))
sns.heatmap(df_meanshift.drop(columns=["Preprocessing", "Clusters"]), annot=True, fmt=".3f", cmap="coolwarm", ax=ax)
plt.title("Mean-Shift Clustering Heatmap")
plt.savefig(f"{output_dir}/heatmap_meanshift.png")
plt.close()

print(" Clustering analysis completed! All results saved in 'clu_outputs' folder.")

     
