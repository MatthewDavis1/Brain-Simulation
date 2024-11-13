#!/usr/bin/env python3

import argparse
import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

def load_data(filepath):
    """Load and preprocess the neuron potentials data."""
    df = pd.read_csv(filepath)
    # Remove timestep column for clustering
    X = df.drop('Timestep', axis=1).values
    return df, X

def determine_optimal_clusters(X, output_dir, max_clusters=10):
    """
    Determine optimal number of clusters using silhouette score.
    Returns the optimal k value.
    """
    silhouette_scores = []
    K = range(2, max_clusters + 1)
    
    for k in K:
        kmeans = KMeans(n_clusters=k, random_state=42)
        kmeans.fit(X)
        score = silhouette_score(X, kmeans.labels_)
        silhouette_scores.append(score)
    
    # Get optimal k (adding 2 because we started from k=2)
    optimal_k = silhouette_scores.index(max(silhouette_scores)) + 2
    
    # Plot silhouette scores
    plt.figure(figsize=(10, 6))
    plt.plot(K, silhouette_scores, 'bo-')
    plt.xlabel('Number of Clusters (k)')
    plt.ylabel('Silhouette Score')
    plt.title('Silhouette Score vs Number of Clusters')
    plt.savefig(output_dir / 'silhouette_scores.png')
    plt.close()
    
    return optimal_k

def analyze_brain_states(filepath, output_dir):
    """Main analysis function."""
    # Create output directory if it doesn't exist
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print("Loading data...")
    df, X = load_data(filepath)
    
    # Standardize the features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # Determine optimal number of clusters
    print("Determining optimal number of clusters...")
    optimal_k = determine_optimal_clusters(X_scaled, output_dir)
    print(f"Optimal number of clusters: {optimal_k}")
    
    # Perform k-means clustering
    print("Performing k-means clustering...")
    kmeans = KMeans(n_clusters=optimal_k, random_state=42)
    clusters = kmeans.fit_predict(X_scaled)
    
    # Get cluster centers and transform back to original scale
    centers = scaler.inverse_transform(kmeans.cluster_centers_)
    
    # Calculate cluster frequencies
    cluster_frequencies = np.bincount(clusters)
    significant_clusters = np.argsort(cluster_frequencies)[::-1]
    
    # Determine number of significant modes to show (N)
    # Using square root of optimal_k rounded up
    N = min(int(np.ceil(np.sqrt(optimal_k))), optimal_k)
    
    # Plot the top N significant modes
    fig, axes = plt.subplots(N, 1, figsize=(15, 5*N))
    if N == 1:
        axes = [axes]
    
    for i in range(N):
        cluster_idx = significant_clusters[i]
        frequency = cluster_frequencies[cluster_idx]
        percentage = (frequency / len(clusters)) * 100
        
        # Plot the brain state pattern
        ax = axes[i]
        sns.heatmap(
            centers[cluster_idx].reshape(1, -1),
            ax=ax,
            cmap='RdBu_r',
            center=0,
            cbar_kws={'label': 'Neuron Potential'}
        )
        ax.set_title(f'Brain State {i+1} (Cluster {cluster_idx}): {percentage:.1f}% of timesteps')
        ax.set_xlabel('Neuron ID')
        ax.set_ylabel('State')
    
    plt.tight_layout()
    plt.savefig(output_dir / 'brain_states.png')
    plt.close()
    
    # Save cluster assignments
    results_df = df.copy()
    results_df['Cluster'] = clusters
    results_df.to_csv(output_dir / 'clustered_states.csv', index=False)
    
    print(f"\nAnalysis complete. Generated files in {output_dir}:")
    print("- silhouette_scores.png: Plot showing optimal number of clusters")
    print("- brain_states.png: Visualization of significant brain states")
    print("- clustered_states.csv: Original data with cluster assignments")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Analyze brain states from neuron potential data.')
    parser.add_argument('input_file', type=str, help='Path to the input CSV file containing neuron potentials')
    parser.add_argument('--output-dir', '-o', type=str, default='output',
                      help='Directory to save output files (default: output)')
    args = parser.parse_args()

    # Check if input file exists
    if not Path(args.input_file).exists():
        print(f"Error: {args.input_file} not found!")
        exit(1)
    
    analyze_brain_states(args.input_file, args.output_dir) 