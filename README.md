# Iris Clustering Analysis

This repository contains a Python-based analysis pipeline for applying multiple clustering algorithms to the classic Iris dataset. The goal is to compare the performance of different clustering methods under various preprocessing techniques.

---

## Table of Contents
1. [Project Overview](#project-overview)
2. [Dataset](#dataset)
3. [Installation & Dependencies](#installation--dependencies)
4. [Preprocessing Methods](#preprocessing-methods)
5. [Clustering Algorithms & Metrics](#clustering-algorithms--metrics)
6. [Outputs](#outputs)

---

## Project Overview
The Iris dataset is a well-known benchmark containing 150 samples of three Iris species, described by four features: sepal length, sepal width, petal length, and petal width. In this project, we:

- Load and preprocess data using various scaling and dimensionality reduction techniques.
- Apply three clustering algorithms (K-Means, Hierarchical Agglomerative, Mean-Shift) with varying cluster counts.
- Evaluate cluster quality using Silhouette Score, Calinski-Harabasz Index, and Davies-Bouldin Score.
- Visualize results in tabular form and heatmaps.

---

## Dataset
- **Source**: UCI Machine Learning Repository
- **URL**: https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data

The dataset has 150 rows and 5 columns (the last being the species label, which we drop for clustering).

---

## Installation & Dependencies
Ensure you have Python 3.7+ installed. Install required packages via pip:

```bash
pip install numpy pandas matplotlib seaborn scikit-learn
```

---

## Preprocessing Methods
We compare performance under six data representations:
1. **No Data Processing**: Raw feature values.
2. **Normalization**: Min-Max scaling to [0,1].
3. **Transform**: Standard scaling (zero mean, unit variance).
4. **PCA**: Projection into first 2 principal components.
5. **T+N**: Standard scaling applied after normalization.
6. **T+N+PCA**: T+N, then PCA.

---

## Clustering Algorithms & Metrics
- **Algorithms**:
  - K-Means (with k = 3, 4, 5)
  - Hierarchical Agglomerative Clustering (same k values)
  - Mean-Shift (automatically finds clusters)

- **Evaluation Metrics**:
  - **Silhouette Score**: Measures cohesion vs separation.
  - **Calinski-Harabasz Index**: Ratio of between-cluster to within-cluster dispersion.
  - **Davies-Bouldin Score**: Lower values indicate better separation.

---

## Outputs
- **Tabular Results**: PNG tables and CSVs for each algorithm summarizing metrics across preprocessing methods and cluster counts.
- **Heatmaps**: Visual comparison of scoring metrics for quick interpretation.
