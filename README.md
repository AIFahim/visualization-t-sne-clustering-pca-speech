# Visualization of T-SNE Clustering, PCA, and Speech Data Analysis
This repository contains various Python scripts for processing, analyzing, and visualizing speech data. The scripts demonstrate the use of different machine learning techniques such as PCA (Principal Component Analysis), t-SNE (t-Distributed Stochastic Neighbor Embedding), and clustering algorithms like KMeans, DBSCAN, and GMM (Gaussian Mixture Models).

## Directory Structure and File Descriptions

## Audio Preprocessing
`cut_audios.py`
- Description: This script is used for cutting audio files. It trims a specified duration from the start and end of audio files in a given directory.
- Key Functions:
cut_audio: Trims the audio file by a specified duration.
process_audio_files: Processes all audio files in a specified folder.
- Usage: Modify the input_folder and output_folder variables and run the script to process audio files.

`cut_audios_multiprocessing.py`
- Description: Similar to cut_audios.py, but utilizes multiprocessing for faster processing.
- Key Functions:
cut_audio: Trims the audio file by a specified duration.
process_audio_file: Wrapper function for multiprocessing.
process_audio_files: Processes all audio files in a specified folder using multiprocessing.
- Usage: Set the input_folder, output_folder, and cut_time_in_milliseconds, then run the script.

## Clusters Scatter Plots after Dimensionality Reduction with PCA
`dbscan_pca_plot_with_kmeans.py`
- Description: Applies DBSCAN clustering on data reduced using PCA and plots the results.
- Key Functions:
Data loading and preprocessing.
PCA for dimensionality reduction.
DBSCAN for clustering.
Saving cluster results and plotting.
- Usage: Adjust file paths and parameters as needed.

`gmm_pca_plot_with_kmeans.py`
- Description: Uses Gaussian Mixture Model for clustering on PCA-reduced data.
- Key Functions:
Data loading and preprocessing.
PCA for dimensionality reduction.
GMM for clustering.
Saving cluster results and plotting.
- Usage: Modify file paths and parameters before running.

`pca_cluster_plots.py`
- Description: Visualizes data points and their cluster assignments using PCA and KMeans.
- Key Functions:
Data loading and preprocessing.
PCA for dimensionality reduction.
KMeans for clustering.
Plotting the clusters.
- Usage: Configure data paths and parameters as required.

`tsne_pca_plot_with_kmeans.py`
- Description: Combines t-SNE with PCA for dimensionality reduction and applies KMeans clustering.
- Key Functions:
Data loading and preprocessing.
t-SNE and PCA for dimensionality reduction.
KMeans for clustering.
Cluster result saving and plotting.
- Usage: Adjust file paths and clustering parameters.
Determining Number of Clusters

`silhouette_score_dbscan.py`
- Description: Calculates the silhouette score for different configurations of DBSCAN to determine the optimal clustering parameters.
- Key Functions:
Data loading and preprocessing.
Iterating over different eps and min_samples values for DBSCAN.
Calculating and comparing silhouette scores.
- Usage: Run the script to find the best DBSCAN parameters.