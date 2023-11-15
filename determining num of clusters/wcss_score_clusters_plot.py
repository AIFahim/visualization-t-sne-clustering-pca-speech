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
data = pd.read_csv("/home/asif/viz_eda_max_non_speech_regions/output_features_w_guid_cutted.csv")
y = data['GUID_name']
X = data.loc[:, data.columns != 'GUID_name']
X = X.dropna(how='any')
cols = X.columns
# ms = MinMaxScaler()
# X = ms.fit_transform(X)


# Dimensionality reduction using PCA to 2 components
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X)

# Create a DataFrame with PCA components and cluster assignments
df_pca = pd.DataFrame(X_pca, columns=['PCA1', 'PCA2'])
# X = pd.DataFrame(X, columns=cols)
X = df_pca

# Applying KMeans
# kmeans = KMeans(n_clusters=2, random_state=0, n_init=10) 
# kmeans.fit(X)

# # Silhouette Score
# silhouette_avg = silhouette_score(X, kmeans.labels_)
# print("Silhouette Score:", silhouette_avg)

# # WCSS Calculation
# wcss = kmeans.inertia_

# Plotting WCSS for different numbers of clusters
rng = range(1, 30)
wcss_values = []
for k in rng:
    kmeans = KMeans(n_clusters=k, random_state=0, init='k-means++', n_init=10)
    kmeans.fit(X)
    wcss_values.append(kmeans.inertia_)
    print(f"{k} completed")

# Plotting the WCSS values
plt.plot(rng, wcss_values)
plt.xlabel('Number of Clusters')
plt.ylabel('WCSS')
plt.xticks(np.arange(min(rng), max(rng)+1, 1))
plt.title('WCSS vs. Number of Clusters')
plt.savefig('wcss_score_cutted_audios.png')
plt.show()
