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

def fit_pca(X):
    pca = PCA(n_components=n_components)
    return pca.fit(X)

# Split your data into smaller chunks for parallel processing
num_chunks = 4  # You can adjust this based on the number of cores you want to use
X_chunks = np.array_split(X, num_chunks)

# Parallelize the PCA fitting step using joblib
with joblib.Parallel(n_jobs=-1) as parallel:
    pca_results = parallel(joblib.delayed(fit_pca)(chunk) for chunk in X_chunks)

# Merge results from parallel processing
pca = PCA(n_components=n_components)
pca_results = pca.fit_transform(np.vstack([result.transform(X) for result in pca_results]))

# Create DataFrame for principal components
principalDf = pd.DataFrame(data=pca_results, columns=['principal component 1', 'principal component 2'])

# concatenate with target label
finalDf = pd.concat([principalDf, y], axis=1)

print(pca.explained_variance_ratio_)



plt.figure(figsize = (16, 9))
# def plot_scatter(chunk, ax):
#     sns.scatterplot(x="principal component 1", y="principal component 2", data=chunk, hue="GUID_name", alpha=0.7, s=100, ax=ax)
#     ax.set_title('PCA on Genres', fontsize=25)
#     ax.tick_params(axis='x', labelsize=14)
#     ax.tick_params(axis='y', labelsize=10)
#     ax.set_xlabel("Principal Component 1", fontsize=15)
#     ax.set_ylabel("Principal Component 2", fontsize=15)

# # Create a single figure with multiple subplots
# fig, axes = plt.subplots(nrows=num_chunks, figsize=(16, 9*num_chunks))

# # Split finalDf into smaller chunks for parallel processing
# num_chunks = 4  # You can adjust this based on the number of cores you want to use
# finalDf_chunks = np.array_split(finalDf, num_chunks)

# # Parallelize the plotting step using joblib
# with joblib.Parallel(n_jobs=-1) as parallel:
#     parallel(joblib.delayed(plot_scatter)(chunk, ax) for chunk, ax in zip(finalDf_chunks, axes))

# Split finalDf into smaller chunks for parallel processing
num_chunks = 30  # You can adjust this based on the number of cores you want to use
finalDf_chunks = np.array_split(finalDf, num_chunks)


"""
def plot_scatter(idx):
    chunk = finalDf_chunks[idx]
    print("----- here ------", idx)
    fig, ax = plt.subplots(figsize=(16, 9))
    sns.scatterplot(x="principal component 1", y="principal component 2", data=chunk, hue="GUID_name", alpha=0.7, s=100, ax=ax)
    ax.set_title('PCA on Genres', fontsize=25)
    ax.tick_params(axis='x', labelsize=14)
    ax.tick_params(axis='y', labelsize=10)
    ax.set_xlabel("Principal Component 1", fontsize=15)
    ax.set_ylabel("Principal Component 2", fontsize=15)
    plt.close(fig)  # Close the figure to avoid displaying individual plots
    # plt.savefig(f"PCA_Scatter_chunks_{idx}.jpg") 


    # Create a multiprocessing pool
with multiprocessing.Pool(num_chunks) as pool:
    pool.map(plot_scatter, range(num_chunks))

"""

def plot_scatter(idx, figures_list):
    chunk = finalDf_chunks[idx]
    print("----- here ------", idx)
    fig, ax = plt.subplots(figsize=(16, 9))
    sns.scatterplot(x="principal component 1", y="principal component 2", data=chunk, hue="GUID_name", alpha=0.7, s=100, ax=ax)
    ax.set_title('PCA on Genres', fontsize=25)
    ax.tick_params(axis='x', labelsize=14)
    ax.tick_params(axis='y', labelsize=10)
    ax.set_xlabel("Principal Component 1", fontsize=15)
    ax.set_ylabel("Principal Component 2", fontsize=15)
    figures_list[idx] = fig  # Use idx as the index to store the figure
    plt.close(fig)  # Close the figure to avoid displaying individual plots


# Create a multiprocessing pool
manager = multiprocessing.Manager()
figures_list = manager.list([None] * num_chunks)  # Initialize the list with None values
with multiprocessing.Pool(num_chunks) as pool:
    pool.starmap(plot_scatter, [(idx, figures_list) for idx in range(num_chunks)])

# Combine the individual figures into one
fig, axes = plt.subplots(nrows=num_chunks, figsize=(16, 9*num_chunks))
for i, ax in enumerate(axes):
    ax.axis('off')
    if figures_list[i] is not None:  # Check if a figure exists at the given index
        ax.imshow(figures_list[i].get_axes()[0].get_images()[0].get_array(), aspect='auto')
plt.tight_layout()
plt.savefig("Combined_PCA_Scatter.jpg")  # Save the combined plot


# plt.tight_layout()  # Adjust the spacing between subplots
# plt.savefig("Combined_PCA_Scatter.jpg")  # Save the combined plot