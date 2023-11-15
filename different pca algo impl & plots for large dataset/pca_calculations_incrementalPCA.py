from sklearn import preprocessing
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA, IncrementalPCA
from joblib import Parallel, delayed
# from sklearn.externals import joblib
import joblib
import numpy as np

data = pd.read_csv("/home/asif/vizualizations/output_features_w_guid.csv")

# y = data['label']
print(data.columns)

# assert False
y = data['GUID_name']
X = data.loc[:, data.columns != 'GUID_name']

# Check for missing values in X
print("Missing values in X before dropping rows:\n", X.isnull().sum())

# Remove rows with missing values (NaN)
X = X.dropna(how='any')

# Check for missing values again after dropping rows
print("Missing values in X after dropping rows:\n", X.isnull().sum())

# Check if X is still a valid DataFrame
if not isinstance(X, pd.DataFrame) or X.empty:
    print("No valid DataFrame after dropping rows with missing values. Aborting PCA.")
    exit()

#### NORMALIZE X ####
cols = X.columns
min_max_scaler = preprocessing.MinMaxScaler()
np_scaled = min_max_scaler.fit_transform(X)
X = pd.DataFrame(np_scaled, columns=cols)

# Parallelize PCA computation
n_components = 2

# Incremental PCA
batch_size = 1000  # You can adjust the batch size based on your memory constraints

pca = IncrementalPCA(n_components=n_components)

for chunk in np.array_split(X, len(X) // batch_size):
    pca.partial_fit(chunk)

pca_results = pca.transform(X)

# Create DataFrame for principal components
principalDf = pd.DataFrame(data=pca_results, columns=['principal component 1', 'principal component 2'])

# concatenate with target label
finalDf = pd.concat([principalDf, y], axis=1)

print(pca.explained_variance_ratio_)



plt.figure(figsize = (16, 9))
sns.scatterplot(x = "principal component 1", y = "principal component 2", data = finalDf, hue = "GUID_name", alpha = 0.7,
               s = 100);

plt.title('PCA on Genres', fontsize = 25)
plt.xticks(fontsize = 14)
plt.yticks(fontsize = 10);
plt.xlabel("Principal Component 1", fontsize = 15)
plt.ylabel("Principal Component 2", fontsize = 15)
plt.savefig("PCA Scattert_incrementalPCA.jpg")
