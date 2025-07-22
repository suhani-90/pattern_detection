# discover_patterns.py
import numpy as np
import pandas as pd
import stumpy
import torch
from tslearn.clustering import TimeSeriesKMeans
import matplotlib.pyplot as plt
import time
from kneed import KneeLocator
import warnings
import os
import argparse
from typing import Dict, Any, Optional, Tuple, List

#Configuration
CONFIG = {
    "output_dir": "output/discovery",
    "window_size_range": range(15,61,15),
    "k_range_to_test": range(2, 11),
    "candidate_percentile": 15,
    "kmeans_max_iter_elbow": 5,
    "kmeans_max_iter_final": 15,
    "dtw_window_frac": 0.2, #Sakoe-Chiba window size (Fraction of window size)
    "random_state": 42,
}

#Loading data
def load_and_prepare_data(filepath: str) -> Optional[np.ndarray]:
    print(f"Loading data from '{filepath}'...")
    try:
        df = pd.read_csv(filepath)
        ts = df['close'].astype(np.float64).values
        print(f"Time series length: {len(ts)}")
        return ts
    except (FileNotFoundError, KeyError) as e:
        print(f"ERROR: Could not load data. {e}")
        return None

#Matrix Profile
def calculate_matrix_profile(ts: np.ndarray, window_size: int) -> np.ndarray:
    print(f"Calculating Matrix Profile for window size {window_size}")
    start_time = time.time()
    if torch.cuda.is_available():
        matrix_profile = stumpy.gpu_stump(ts, m=window_size)
    else:
        matrix_profile = stumpy.stump(ts, m=window_size)
    print(f"Matrix Profile computed in {time.time() - start_time:.2f} seconds.")
    return matrix_profile

#Candidate Motif Sleection
def extract_and_normalize_candidates(matrix_profile: np.ndarray, ts: np.ndarray, window_size: int, percentile_threshold:int = 10) -> Tuple[np.ndarray, np.ndarray]:
    print("Selecting and Normalizing Candidates...")
    mp_values = matrix_profile[:, 0].astype(np.float64)
    if not np.any(np.isfinite(mp_values)):
        return np.array([]), np.array([])
    threshold = np.percentile(mp_values[np.isfinite(mp_values)], percentile_threshold)
    print(threshold)
    candidate_indices = np.where(mp_values < threshold)[0]
    print(f"Found {len(candidate_indices)} raw motifs under the threshold (before filtering).")
    filtered_indices, last_idx = [], -np.inf
    for idx in sorted(candidate_indices):
        if idx >= last_idx + window_size:
            filtered_indices.append(idx)
            last_idx = idx
    normalized_candidates, final_original_indices = [], []
    for idx in filtered_indices:
        subsequence = ts[idx : idx + window_size]
        if np.std(subsequence) > 1e-6:
            normalized_candidates.append((subsequence - np.mean(subsequence)) / np.std(subsequence))
            final_original_indices.append(idx)
    X = np.array(normalized_candidates)
    print(f"Identified and Z-Normalized {len(X)} candidates.")
    return X, np.array(final_original_indices)

#Elbow
def find_optimal_k(X: np.ndarray, k_range: range, config: Dict[str, Any]) -> Tuple[Optional[int], Optional[KneeLocator]]:
    print("Finding Optimal K...")
    start_time = time.time()
    if len(X) < max(k_range):
        return None, None

    inertias = [
        TimeSeriesKMeans(n_clusters=k, metric="euclidean", max_iter=config["kmeans_max_iter_elbow"], random_state=config["random_state"], n_jobs=-1).fit(X).inertia_
        for k in k_range
    ]

    try:
        kl = KneeLocator(list(k_range), inertias, curve='convex', direction='decreasing')
        optimal_k = kl.elbow
        if optimal_k:
            print(f"Detected optimal k = {optimal_k}")
            print(f"optimal k in {time.time() - start_time:.2f} seconds.")
        else:
            print("Could not find a clear elbow point.")
        return optimal_k, kl # Return the locator object itself
    except Exception:
        print("An exception occurred during knee detection.")
        return None, None

#visualization
def visualize_and_save_clusters(
    X: np.ndarray, clusters: np.ndarray, centroids: np.ndarray,
    window_size: int, optimal_k: int, output_path: str
):
   
    print(f"Visualizing clusters for w={window_size}, k={optimal_k}")
    fig, axs = plt.subplots(optimal_k, 1, figsize=(14, 4 * optimal_k), sharex=True, squeeze=False)
    fig.suptitle(f'Discovered Pattern Clusters (Window Size = {window_size}, k={optimal_k})', fontsize=20, y=1.0)
    for i in range(optimal_k):
        ax = axs[i, 0]
        cluster_members = X[clusters == i]
        ax.set_title(f"Cluster {i} ({len(cluster_members)} patterns found)")
        for member in cluster_members: ax.plot(member, color='skyblue', alpha=0.3)
        ax.plot(np.ravel(centroids[i]), color='black', linewidth=3, label='Centroid')
        ax.legend()
        ax.grid(True, linestyle='--')
    plt.xlabel("Time Steps in Pattern")
    plt.tight_layout(rect=[0, 0.03, 1, 0.96])
    plt.savefig(output_path, dpi=150)
    plt.close(fig)
    print(f"Saved cluster visualization to {output_path}")



# Main
def main(config: Dict[str, Any]):
    
    parser = argparse.ArgumentParser(description="Discover and save time series patterns.")
    parser.add_argument("--filepath", type=str, required=True, help="Path to the input CSV file.")
    args = parser.parse_args()

    base_filename = os.path.splitext(os.path.basename(args.filepath))[0]
    file_specific_output_dir = os.path.join(config["output_dir"], base_filename)
    os.makedirs(file_specific_output_dir, exist_ok=True)
    print(f"Results will be saved in: {file_specific_output_dir}")

    ts = load_and_prepare_data(args.filepath)
    if ts is None: return

    results_X = []
    results_clusters = []
    results_centroids = []
    results_indices = []
    results_window_sizes = []
    results_k = []

    
    for window_size in config["window_size_range"]:
        print(f"\n{'='*25} PROCESSING WINDOW SIZE: {window_size} {'='*25}")

        matrix_profile = calculate_matrix_profile(ts, window_size)
        X, original_indices = extract_and_normalize_candidates(
            matrix_profile, ts, window_size, config["candidate_percentile"]
        )

        if len(X) < max(config["k_range_to_test"]):
            print("Not enough candidates found. Skipping this window size.")
            continue

        optimal_k, knee_locator = find_optimal_k(X, config["k_range_to_test"], config)

        if knee_locator:
            knee_locator.plot_knee()
            elbow_plot_path = os.path.join(file_specific_output_dir, f"elbow_plot_w{window_size}.png")
            plt.savefig(elbow_plot_path)
            plt.close()
            print(f"Saved elbow plot to {elbow_plot_path}")

        if optimal_k:
            print(f"--- Performing Final Clustering with k={optimal_k} ---")
            start_time = time.time()
            sakoe_chiba_window = int(config["dtw_window_frac"] * window_size)
            print(f"Using constrained DTW with a Sakoe-Chiba window of {sakoe_chiba_window}")

            final_km = TimeSeriesKMeans(
                n_clusters=optimal_k,
                metric="dtw",
                metric_params={"global_constraint": "sakoe_chiba", "sakoe_chiba_radius": sakoe_chiba_window},
                max_iter=config["kmeans_max_iter_final"],
                random_state=config["random_state"],
                n_jobs=-1
            )

            clusters = final_km.fit_predict(X)
            centroids = final_km.cluster_centers_

            results_X.append(X)
            results_clusters.append(clusters)
            results_centroids.append(centroids)
            results_indices.append(original_indices)
            results_window_sizes.append(window_size)
            results_k.append(optimal_k)
             
            window_plot_dir = os.path.join(file_specific_output_dir, f"w_{window_size}_plots")
            os.makedirs(window_plot_dir, exist_ok=True)
            plot_path = os.path.join(window_plot_dir, "cluster_shapes.png")
            visualize_and_save_clusters(X, clusters, centroids, window_size, optimal_k, plot_path)


    if results_window_sizes:
        artifacts_path = os.path.join(file_specific_output_dir, "discovery_artifacts.npz")
        
        num_results = len(results_window_sizes)

        final_X = np.empty(num_results, dtype=object)
        final_clusters = np.empty(num_results, dtype=object)
        final_centroids = np.empty(num_results, dtype=object)
        final_indices = np.empty(num_results, dtype=object)

        for i in range(num_results):
            final_X[i] = results_X[i]
            final_clusters[i] = results_clusters[i]
            final_centroids[i] = results_centroids[i]
            final_indices[i] = results_indices[i]

        np.savez_compressed(
            artifacts_path,
            X_list=final_X,
            clusters_list=final_clusters,
            centroids_list=final_centroids,
            indices_list=final_indices,
            window_sizes=np.array(results_window_sizes),
            k_values=np.array(results_k)
        )
        print(f"Process Completed{time.time() - start_time:.2f} seconds.")

        print(f"\nSuccessfully saved all pattern data to: {artifacts_path}")
    else:
        print("\nNo patterns were discovered across any window size. No artifact file created.")

    print(f"\n{'='*30} DISCOVERY COMPLETE {'='*30}")


if __name__ == "__main__":
    warnings.filterwarnings("ignore", category=UserWarning)
    main(CONFIG)