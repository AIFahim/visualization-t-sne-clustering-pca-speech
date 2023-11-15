from sklearn.cluster import DBSCAN
from sklearn.metrics import silhouette_score
import numpy as np
from sklearn.cluster import DBSCAN
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import MinMaxScaler
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA

# Data Loading
data = pd.read_csv("/home/asif/viz_eda_max_non_speech_regions/output_features_w_guid_cutted.csv")

# Level and Features Separation
y = data['GUID_name']
X = data.loc[:, data.columns != 'GUID_name']

# Check for missing values in X
print("Missing values in X before dropping rows:\n", X.isnull().sum())

# Remove rows with missing values (NaN)
X = X.dropna(how='any')

# Features Scaling
cols = X.columns
ms = MinMaxScaler()
X = ms.fit_transform(X)
X = pd.DataFrame(X, columns=cols)

# Dimensionality reduction using PCA to 2 components
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X)
# Create a DataFrame with PCA components and cluster assignments
df_pca = pd.DataFrame(X_pca, columns=['PCA1', 'PCA2'])
X = df_pca

# Define the range of values for eps and min_samples to explore
eps_values = np.linspace(0.1, 1.0, num=10)
min_samples_values = range(2, 10)

best_score = -1  # Initialize the best silhouette score
best_eps = None
best_min_samples = None

for eps in eps_values:
    for min_samples in min_samples_values:
        dbscan = DBSCAN(eps=eps, min_samples=min_samples)
        labels = dbscan.fit_predict(X)
        if len(np.unique(labels)) > 1:  # Ensure at least one non-noise cluster is found
            score = silhouette_score(X, labels)
            if score > best_score:
                best_score = score
                best_eps = eps
                best_min_samples = min_samples

print("Best combination: eps =", best_eps, ", min_samples =", best_min_samples)
