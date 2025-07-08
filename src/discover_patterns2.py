# discover_patterns_gpu.py 
import numpy as np
import pandas as pd
import stumpy
import torch
import matplotlib.pyplot as plt
import time
from kneed import KneeLocator
import warnings
import os
import argparse
from typing import Dict, Any, Optional, Tuple, List

## NEW: Import TSKMeans from tsai
from tsai.clustering import TSKMeans

# --- Configuration ---
# You can potentially increase the iteration counts as tsai is much faster
CONFIG = {
    "output_dir": "output/discovery", 
    "window_size_range": range(15, 90, 15),
    "k_range_to_test": range(2, 11),
    "candidate_percentile": 10,
    "kmeans_max_iter_elbow": 10,   # Increased, as GPU version is fast
    "kmeans_max_iter_final": 25,   # Increased, as GPU version is fast
    "random_state": 42, # Kept for other potential random operations, though tsai's TSKMeans has deterministic init.
}

# --- Load and Prepare data (No changes needed) ---
def load_and_prepare_data(filepath: str) -> Optional[np.ndarray]:
    print(f"Loading data from '{filepath}'...")
    try:
        df = pd.read_csv(filepath)
        # Using 'close' column, ensure it exists or change it.
        ts = df['close'].astype(np.float64).values
        print(f"Time series length: {len(ts)}")
        return ts
    except (FileNotFoundError, KeyError) as e:
        print(f"ERROR: Could not load data. {e}")
        return None

# --- MATRIX PROFILE (No changes needed) ---
def calculate_matrix_profile(ts: np.ndarray, window_size: int) -> np.ndarray:
    print(f"Calculating Matrix Profile for window size {window_size}")
    start_time = time.time()
    # This already correctly uses the GPU if available
    if torch.cuda.is_available():
        matrix_profile = stumpy.gpu_stump(ts, m=window_size)
    else:
        matrix_profile = stumpy.stump(ts, m=window_size)
    print(f"Matrix Profile computed in {time.time() - start_time:.2f} seconds.")
    return matrix_profile

# --- Extract Motifs (No changes needed) ---
def extract_and_normalize_candidates(matrix_profile: np.ndarray, ts: np.ndarray, window_size: int, percentile_threshold: int = 10) -> Tuple[np.ndarray, np.ndarray]:
    print("Selecting and Normalizing Candidates...")
    mp_values = matrix_profile[:, 0].astype(np.float64)
    if not np.any(np.isfinite(mp_values)):
        return np.array([]), np.array([])
    threshold = np.percentile(mp_values[np.isfinite(mp_values)], percentile_threshold)
    candidate_indices = np.where(mp_values < threshold)[0]
    
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

## MODIFIED: This function now uses tsai for GPU-accelerated DTW
def find_optimal_k_gpu(X: np.ndarray, k_range: range, config: Dict[str, Any]) -> Optional[int]:
    """Finds the optimal number of clusters using the elbow method with GPU-accelerated DTW."""
    print("Finding Optimal K using GPU-accelerated DTW (tsai)...")
    if len(X) < max(k_range):
        print("Not enough candidates for the specified k-range.")
        return None
    
    # tsai expects a 3D PyTorch tensor: (n_samples, n_variables, n_timesteps)
    # Our X is (n_samples, n_timesteps), so we add the variable dimension.
    X_tsai = torch.from_numpy(X).float().unsqueeze(1)
    
    inertias = []
    print("Testing k values for elbow method:")
    for k in k_range:
        start_time = time.time()
        # Use TSKMeans with the fast DTW implementation
        km = TSKMeans(n_clusters=k, dist='dtw_fast', max_iter=config["kmeans_max_iter_elbow"])
        
        # Fit the model. It will automatically use the GPU if available.
        km.fit(X_tsai.to(km.device))
        
        # Inertia is a tensor, move it to CPU and convert to numpy for kneed
        inertias.append(km.inertia_.cpu().numpy())
        print(f"  - k={k} completed in {time.time() - start_time:.2f}s")

    try:
        kl = KneeLocator(list(k_range), inertias, curve='convex', direction='decreasing')
        optimal_k = kl.elbow
        if optimal_k:
            print(f"Detected optimal k = {optimal_k}")
            kl.plot_knee()
            plt.title('Elbow Method Analysis (GPU DTW)')
            plt.xlabel("Number of clusters (k)")
            plt.ylabel("Inertia")
            plt.show()
        else:
            print("Could not find a clear elbow point.")
        return optimal_k
    except Exception as e:
        print(f"ERROR: KneeLocator failed. {e}")
        return None

# --- Visualise Clusters (No changes needed) ---
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

# --- Main Execution Logic ---
def main(config: Dict[str, Any]):
    parser = argparse.ArgumentParser(description="Discover and save time series patterns using GPU acceleration.")
    parser.add_argument("--filepath", type=str, required=True, help="Path to the input CSV file.")
    args = parser.parse_args()

    base_filename = os.path.splitext(os.path.basename(args.filepath))[0]
    file_specific_output_dir = os.path.join(config["output_dir"], base_filename)
    os.makedirs(file_specific_output_dir, exist_ok=True)
    print(f"Results will be saved in: {file_specific_output_dir}")

    ts = load_and_prepare_data(args.filepath)
    if ts is None: return

    # Prepare lists to store results from all window sizes
    all_results = []

    for window_size in config["window_size_range"]:
        print(f"\n{'='*25} PROCESSING WINDOW SIZE: {window_size} {'='*25}")
        
        matrix_profile = calculate_matrix_profile(ts, window_size)
        X, original_indices = extract_and_normalize_candidates(
            matrix_profile, ts, window_size, config["candidate_percentile"]
        )

        if len(X) < max(config["k_range_to_test"]):
            print("Not enough candidates found. Skipping this window size.")
            continue

        ## MODIFIED: Call the new GPU-based function to find k
        optimal_k = find_optimal_k_gpu(X, config["k_range_to_test"], config)

        if optimal_k:
            print(f"--- Performing Final GPU Clustering with k={optimal_k}, metric=DTW ---")
            start_time = time.time()
            
            ## MODIFIED: Use tsai for the final clustering
            # Convert data to a 3D PyTorch tensor
            X_tsai = torch.from_numpy(X).float().unsqueeze(1)

            # Create and run the final TSKMeans model
            final_km = TSKMeans(
                n_clusters=optimal_k, 
                dist='dtw_fast', 
                max_iter=config["kmeans_max_iter_final"]
            )
            clusters_tensor = final_km.fit_predict(X_tsai)
            
            # Convert results back to NumPy for compatibility with the rest of the script
            clusters = clusters_tensor.cpu().numpy()
            centroids = final_km.centroids.cpu().numpy().squeeze(1) # Remove the middle dimension
            
            print(f"Final clustering completed in {time.time() - start_time:.2f} seconds.")

            # Store results
            all_results.append({
                "X": X, "clusters": clusters, "centroids": centroids,
                "indices": original_indices, "window_size": window_size, "k": optimal_k
            })

            # Plotting
            plot_path = os.path.join(file_specific_output_dir, f"w_{window_size}_k_{optimal_k}_clusters.png")
            visualize_and_save_clusters(X, clusters, centroids, window_size, optimal_k, plot_path)
    
    # Save all results to a single file at the end
    if all_results:
        artifacts_path = os.path.join(file_specific_output_dir, "discovery_artifacts.npz")
        np.savez_compressed(
            artifacts_path,
            X_list=np.array([r['X'] for r in all_results], dtype=object),
            clusters_list=np.array([r['clusters'] for r in all_results], dtype=object),
            centroids_list=np.array([r['centroids'] for r in all_results], dtype=object),
            indices_list=np.array([r['indices'] for r in all_results], dtype=object),
            window_sizes=np.array([r['window_size'] for r in all_results]),
            k_values=np.array([r['k'] for r in all_results])
        )
        print(f"\nSuccessfully saved all pattern data to: {artifacts_path}")
    else:
        print("\nNo patterns were discovered across any window size. No artifact file created.")

    print(f"\n{'='*30} DISCOVERY COMPLETE {'='*30}")


if __name__ == "__main__":
    # tsai can be a bit verbose with warnings, this can help
    warnings.filterwarnings("ignore", category=UserWarning) 
    main(CONFIG)