# Visualization of T-SNE Clustering, PCA, and Speech Data Analysis

This repository contains various Python scripts for processing, analyzing, and visualizing speech data. The scripts demonstrate the use of different machine learning techniques such as PCA (Principal Component Analysis), t-SNE (t-Distributed Stochastic Neighbor Embedding), and clustering algorithms like KMeans, DBSCAN, and GMM (Gaussian Mixture Models).

## Directory Structure and File Descriptions

### Audio Preprocessing
- `cut_audios.py`
  - **Description**: Script for cutting audio files by trimming a specified duration from the start and end.
  - **Key Functions**:
    - `cut_audio`: Trims the audio file.
    - `process_audio_files`: Processes all audio files in a folder.
  - **Usage**: Set `input_folder` and `output_folder`, then run.

- `cut_audios_multiprocessing.py`
  - **Description**: Similar to `cut_audios.py`, but uses multiprocessing for efficiency.
  - **Key Functions**:
    - `cut_audio`: Trims the audio file.
    - `process_audio_file`: Multiprocessing wrapper.
    - `process_audio_files`: Processes files using multiprocessing.
  - **Usage**: Configure `input_folder`, `output_folder`, and `cut_time_in_milliseconds`.

### Clusters Scatter Plots after Dimensionality Reduction with PCA
- `dbscan_pca_plot_with_kmeans.py`
  - **Description**: Applies DBSCAN clustering on PCA-reduced data and plots results.
  - **Key Functions**:
    - Data loading and preprocessing.
    - PCA for dimensionality reduction.
    - DBSCAN for clustering.
    - Plotting clusters.
  - **Usage**: Adjust file paths and parameters.

- `gmm_pca_plot_with_kmeans.py`
  - **Description**: Uses GMM for clustering on PCA-reduced data.
  - **Key Functions**:
    - Data loading and preprocessing.
    - PCA for dimensionality reduction.
    - GMM for clustering.
    - Plotting results.
  - **Usage**: Modify file paths and parameters.

- `pca_cluster_plots.py`
  - **Description**: Visualizes clusters using PCA and KMeans.
  - **Key Functions**:
    - Data loading and preprocessing.
    - PCA for dimensionality reduction.
    - KMeans for clustering.
    - Plotting clusters.
  - **Usage**: Configure data paths and parameters.

- `tsne_pca_plot_with_kmeans.py`
  - **Description**: Combines t-SNE with PCA for dimensionality reduction and KMeans clustering.
  - **Key Functions**:
    - Data loading and preprocessing.
    - t-SNE and PCA for dimensionality reduction.
    - KMeans for clustering.
    - Plotting clusters.
  - **Usage**: Adjust file paths and parameters.

### Determining Number of Clusters
- `silhouette_score_dbscan.py`
  - **Description**: Calculates silhouette score for DBSCAN configurations.
  - **Key Functions**:
    - Data loading and preprocessing.
    - Iterating over DBSCAN parameters.
    - Calculating silhouette scores.
  - **Usage**: Run to find optimal DBSCAN parameters.

### Additional Scripts
- `silhouette_score_gmm.py`
  - **Description**: Evaluates silhouette scores for different GMM clustering configurations.
  - **Usage**: Useful for determining the optimal number of clusters in GMM.

- `silhouette_score_kmeans.py`
  - **Description**: Calculates silhouette scores for various KMeans configurations.
  - **Usage**: Assists in finding the best number of clusters for KMeans.

- `spectrogram.py`
  - **Description**: Generates spectrograms from audio files.
  - **Usage**: Useful for visual analysis of audio frequency content.

- `spectrogram_multiprocessing.py`
  - **Description**: Similar to `spectrogram.py`, but uses multiprocessing for efficiency.
  - **Usage**: Efficient for processing large sets of audio files.

- `tsne_cluster_plots.py`
  - **Description**: Visualizes data in lower-dimensional space using t-SNE.
  - **Usage**: Helps in understanding data distribution and clustering.

- `tsne_pca_plot_with_dbscan.py`
  - **Description**: Applies DBSCAN clustering on data reduced with t-SNE and PCA.
  - **Usage**: Useful for visualizing complex data clusters.

- `tsne_pca_plot_with_gmm.py`
  - **Description**: Combines t-SNE and PCA for dimensionality reduction and applies GMM clustering.
  - **Usage**: Effective for visualizing and analyzing speech data clusters.
