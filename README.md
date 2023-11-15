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
  
- `silhouette_score_elbow_plot.py`
  - **Description**: Calculates silhouette scores for different numbers of clusters to find the optimal clustering configuration.
  - **Key Functions**:
    - Data loading and preprocessing.
    - Iterating over different numbers of clusters.
    - Calculating silhouette scores.
    - Plotting silhouette scores against the number of clusters.
  - **Usage**: Run to determine the optimal number of clusters.

- `wcss_score_clusters_plot.py`
  - **Description**: Computes and plots the Within-Cluster Sum of Squares (WCSS) for different numbers of clusters.
  - **Key Functions**:
    - Data loading and preprocessing.
    - PCA for dimensionality reduction.
    - KMeans clustering with varying numbers of clusters.
    - Plotting WCSS values.
  - **Usage**: Useful for identifying the elbow point in clustering.

### Different PCA Algorithm Implementations & Plots for Large Dataset
- `pca_calculations_incrementalPCA.py`
  - **Description**: Implements Incremental PCA for large datasets.
  - **Key Functions**:
    - Data loading and preprocessing.
    - Incremental PCA for handling large datasets.
    - Plotting PCA results.
  - **Usage**: Suitable for datasets that are too large for standard PCA.

- `pca_calculations_parallel_process.py`
  - **Description**: Parallelizes PCA calculations for efficiency.
  - **Key Functions**:
    - Data loading and preprocessing.
    - Parallel processing for PCA.
    - Plotting PCA results.
  - **Usage**: Use when dealing with large datasets and aiming for faster computation.

### Speech Geneva Features Calculation with OpenSMILE
- `speech geneva features cal opensmile.py`
  - **Description**: Extracts speech features using the OpenSMILE tool.
  - **Key Functions**:
    - Processing audio files to extract features.
    - Handling multiple files with multiprocessing.
    - Saving extracted features to a CSV file.
  - **Usage**: Ideal for extracting complex speech features from audio files.


