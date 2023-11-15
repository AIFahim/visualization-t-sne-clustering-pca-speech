from sklearn.cluster import KMeans
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import MinMaxScaler
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
import os
import shutil

# Data Loading
data = pd.read_csv("/home/asif/viz_eda_max_non_speech_regions/clusters_3_cutted_500ms_cluster_0_subcluster_3_cluster_0.csv")


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
# pca = PCA(n_components=2)
# X_pca = pca.fit_transform(X)
# Create a DataFrame with PCA components and cluster assignments  
# df_pca = pd.DataFrame(X_pca, columns=['PCA1', 'PCA2'])
# X = df_pca

# Applying KMeans
kmeans = KMeans(n_clusters=6, random_state=0) 
kmeans.fit(X)

# Adding cluster assignments to the X dataframe
X['Cluster'] = kmeans.labels_

# Dimensionality reduction using t-SNE
tsne = TSNE(n_components=2, random_state=0)
X_tsne = tsne.fit_transform(X)

# print(X.columns)
# assert False

# Create a DataFrame with t-SNE components and cluster assignments
df_tsne = pd.DataFrame(X, columns=['TSNE1' , 'TSNE2']) # ['TSNE1' , 'TSNE2'] ['PCA1', 'PCA2']
df_tsne['level'] = y
df_tsne['Cluster'] = kmeans.labels_


# Create separate folders for each cluster
output_dir = "/home/asif/viz_eda_max_non_speech_regions/clusters_3_cutted_500ms/cluster_0/subcluster_3/cluster_0/subsubcluster_6"
os.makedirs(output_dir, exist_ok=True)

# '''
# Copy the files to the cluster folders
for index, row in df_tsne.iterrows():
    cluster_id = row['Cluster']
    level = row['level']
    file_name = level + ".flac"
    source_path = os.path.join("/home/asif/viz_eda_max_non_speech_regions/cutted_max_non_speech_segments_multipro", file_name)
    destination_folder = os.path.join(output_dir, f"cluster_{cluster_id}")
    os.makedirs(destination_folder, exist_ok=True)
    destination_path = os.path.join(destination_folder, file_name)
    shutil.copy(source_path, destination_path)
# '''

# """
# Scatter plot to visualize the clustering results in 2D
# sns.scatterplot(data=df_tsne, x='PCA1', y='PCA2', hue='Cluster', palette='Set1') # TSNE1 , TSNE2
# plt.title('pca Visualization with KMeans Clustering')
# plt.savefig('pca_plot_with_kmeans_n_6_cutted_500ms.png')
# plt.show()
# """