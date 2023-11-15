from sklearn.cluster import KMeans
from sklearn import preprocessing
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA
import multiprocessing
from joblib import Parallel, delayed
# from sklearn.externals import joblib
import joblib
import numpy as np
from sklearn.preprocessing import MinMaxScaler

# Data Loading
data = pd.read_csv("/home/asif/vizualizations/output_features_w_guid.csv")

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

# Applying KMeans
kmeans = KMeans(n_clusters=2, random_state=0) 
kmeans.fit(X)

# Adding cluster assignments to the original data
data['Cluster'] = kmeans.labels_

# Dimensionality reduction using PCA to 2 components
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X)

# Create a DataFrame with PCA components and cluster assignments
df_pca = pd.DataFrame(X_pca, columns=['PCA1', 'PCA2'])
df_pca['Cluster'] = kmeans.labels_

# Visualize the data points with their cluster assignments
sns.scatterplot(data=df_pca, x='PCA1', y='PCA2', hue='Cluster', palette='Set1')
plt.title('KMeans Clustering')
plt.show()