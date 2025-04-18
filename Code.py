# Import necessary libraries for data manipulation, preprocessing, clustering, and visualization
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.cluster import KMeans, MeanShift, AgglomerativeClustering
from sklearn.metrics import silhouette_score, calinski_harabasz_score, davies_bouldin_score
import os

# Create output directory if it doesn't exist to store results
output_dir = "clustering_outputs"
os.makedirs(output_dir, exist_ok=True)

# Load the Iris dataset from UCI repository and assign column names
url = "https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data"
columns = ['sepal_length', 'sepal_width', 'petal_length', 'petal_width', 'species']
df = pd.read_csv(url, names=columns)
# Drop the species label to perform unsupervised clustering
df.drop(columns=['species'], inplace=True)

# Initialize scalers and PCA transformer
scaler_standard = StandardScaler()   # Standardization: zero mean, unit variance
scaler_minmax = MinMaxScaler()       # Normalization: scale features to [0,1]
pca = PCA(n_components=2)           # Reduce dimensionality to 2 components for visualization

# Apply different preprocessing techniques to the original dataset
# 1. Standardized data
df_standard = pd.DataFrame(scaler_standard.fit_transform(df), columns=df.columns)
# 2. Min-Max normalized data
df_minmax = pd.DataFrame(scaler_minmax.fit_transform(df), columns=df.columns)
# 3. PCA on original data
df_pca = pd.DataFrame(pca.fit_transform(df), columns=['PC1', 'PC2'])
# 4. Standardize then normalize (transform + normalize)
df_tn = pd.DataFrame(scaler_standard.fit_transform(df_minmax), columns=df.columns)
# 5. Transform + normalize + PCA
df_tn_pca = pd.DataFrame(pca.fit_transform(df_tn), columns=['PC1', 'PC2'])

# Collect all preprocessing variations in a dictionary for iteration
preprocessing_methods = {
    "No Data Processing": df,
    "Using Normalization": df_minmax,
    "Using Transform": df_standard,
    "Using PCA": df_pca,
    "Using T+N": df_tn,
    "T+N+PCA": df_tn_pca
}

# Function to apply various clustering algorithms and compute evaluation metrics
def apply_clustering(data, n_clusters):
    """
    Runs K-Means, Hierarchical, and Mean-Shift clustering on `data`,
    then computes silhouette, Calinski-Harabasz, and Davies-Bouldin scores.
    """
    results = []
    algorithms = {
        "K-Means": KMeans(n_clusters=n_clusters, random_state=42),
        "Hierarchical": AgglomerativeClustering(n_clusters=n_clusters),
        "Mean-Shift": MeanShift()
    }

    for algo_name, algorithm in algorithms.items():
        # Fit the model and predict cluster labels
        labels = algorithm.fit_predict(data)

        # Metrics require more than one cluster to compute
        if len(set(labels)) > 1:
            silhouette = silhouette_score(data, labels)
            calinski_harabasz = calinski_harabasz_score(data, labels)
            davies_bouldin = davies_bouldin_score(data, labels)
        else:
            silhouette = calinski_harabasz = davies_bouldin = np.nan

        # Store the method name and metrics
        results.append([algo_name, silhouette, calinski_harabasz, davies_bouldin])

    return results

# Prepare storage for final aggregated results per algorithm
final_results = {"K-Means": [], "Hierarchical": [], "Mean-Shift": []}

# Iterate over each preprocessing method and cluster count to collect metrics
for preprocess_name, dataset in preprocessing_methods.items():
    for clusters in [3, 4, 5]:
        results = apply_clustering(dataset, clusters)
        for row in results:
            # Append preprocessing technique, number of clusters, and computed metrics
            final_results[row[0]].append([preprocess_name, clusters] + row[1:])

# Convert aggregated results into pandas DataFrames for each algorithm
columns = ["Preprocessing", "Clusters", "Silhouette Score", "Calinski-Harabasz", "Davies-Bouldin"]
df_kmeans = pd.DataFrame(final_results["K-Means"], columns=columns)
df_hierarchical = pd.DataFrame(final_results["Hierarchical"], columns=columns)
df_meanshift = pd.DataFrame(final_results["Mean-Shift"], columns=columns)

# Function to save DataFrame as a PNG table image
def save_table_as_image(df, title, filename):
    fig, ax = plt.subplots(figsize=(12, 4))
    ax.axis('tight')
    ax.axis('off')
    table = ax.table(cellText=df.values, colLabels=df.columns, cellLoc='center', loc='center')
    plt.title(title)
    plt.savefig(f"{output_dir}/{filename}")
    plt.close()

# Save each algorithm's results table as an image
save_table_as_image(df_kmeans, "Using K-Means Clustering", "table_kmeans.png")
save_table_as_image(df_hierarchical, "Using Hierarchical Clustering", "table_hierarchical.png")
save_table_as_image(df_meanshift, "Using Mean-Shift Clustering", "table_meanshift.png")

# Also export the results to CSV files for further analysis
df_kmeans.to_csv(f"{output_dir}/results_kmeans.csv", index=False)
df_hierarchical.to_csv(f"{output_dir}/results_hierarchical.csv", index=False)
df_meanshift.to_csv(f"{output_dir}/results_meanshift.csv", index=False)

# Generate and save heatmaps for quick visual comparison of metrics
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

print("Clustering analysis completed!")
