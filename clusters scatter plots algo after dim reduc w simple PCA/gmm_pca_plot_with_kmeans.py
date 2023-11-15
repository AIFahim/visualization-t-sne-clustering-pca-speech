from sklearn.mixture import GaussianMixture
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import MinMaxScaler
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
import os , shutil


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
# ms = MinMaxScaler()
# X = ms.fit_transform(X)
# X = pd.DataFrame(X, columns=cols)

# Dimensionality reduction using PCA to 2 components
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X)
# Create a DataFrame with PCA components and cluster assignments
df_pca = pd.DataFrame(X_pca, columns=['PCA1', 'PCA2'])
X = df_pca

# Applying Gaussian Mixture Model (GMM)
gmm = GaussianMixture(n_components=6, random_state=0) 
gmm_labels = gmm.fit_predict(X)

# Adding cluster assignments to the X dataframe
X['Cluster'] = gmm_labels

# Dimensionality reduction using t-SNE
# tsne = TSNE(n_components=2, random_state=0)
# X_tsne = tsne.fit_transform(X)

# print(X.columns)
# assert False

# Create a DataFrame with t-SNE components and cluster assignments
df_tsne = pd.DataFrame(X, columns=['PCA1', 'PCA2']) # TSNE1 , TSNE2
df_tsne['level'] = y
df_tsne['Cluster'] = gmm_labels


# Create separate folders for each cluster
output_dir = "/home/asif/viz_eda_max_non_speech_regions/clusters_4_pca_gmm_cutted_500ms"
os.makedirs(output_dir, exist_ok=True)


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


# Plotting the clusters
# plt.figure(figsize=(10, 6))
# sns.scatterplot(x='PCA1', y='PCA2', hue='Cluster', data=df_tsne, palette='Set1')
# plt.title('Gaussian Mixture Model Clustering')
# plt.savefig('pca_plot_with_gmm_cutted_500ms.png')
# plt.show()
