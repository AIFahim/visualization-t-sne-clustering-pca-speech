from sklearn.cluster import KMeans
from sklearn import preprocessing
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import silhouette_score

# Data Loading and Preprocessing (same as before)
data = pd.read_csv("/home/asif/viz_eda_max_non_speech_regions/clusters_3_cutted_500ms_cluster_0_subcluster_3_cluster_0.csv")
y = data['GUID_name']
X = data.loc[:, data.columns != 'GUID_name']
X = X.dropna(how='any')
cols = X.columns
ms = MinMaxScaler()
X = ms.fit_transform(X)
X = pd.DataFrame(X, columns=cols)

# Calculate the silhouette scores for different numbers of clusters
silhouette_scores = []
for k in range(2, 20):
    kmeans = KMeans(n_clusters=k, random_state=0, n_init=10)
    kmeans.fit(X)
    labels = kmeans.labels_
    silhouette_scores.append(silhouette_score(X, labels))
    print(f"{k} completed")

# Plotting the silhouette scores
plt.plot(range(2, 20), silhouette_scores)
plt.xlabel('Number of Clusters')
plt.ylabel('Silhouette Score')
plt.title('Silhouette Score vs. Number of Clusters')

# Set x-label interval to 1 by 1
plt.xticks(np.arange(min(range(2, 20)), max(range(2, 20))+1, 1))

plt.savefig('Silhouette_score_cutted_500ms_cluster_0_subcluster_3_cluster_0.png')
plt.show()

# Finding the elbow point (optimal number of clusters)
diff = np.diff(silhouette_scores)
elbow_point = np.argmax(diff) + 2  # Add 2 because range starts from 2
print("Elbow Point (Optimal Number of Clusters):", elbow_point)
