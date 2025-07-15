# compare_across_stocks.py

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
import argparse
import glob
from tslearn.clustering import TimeSeriesKMeans
from kneed import KneeLocator
from typing import Dict, Any, List, Tuple

# Configuration
CONFIG = {
    "discovery_root_dir": "output/discovery",
    "analysis_root_dir": "output/analysis2",
    "comparison_output_dir": "output/comparison",
    "k_range_for_global_clusters": range(2, 8),
    "kmeans_max_iter_elbow": 5,
    "kmeans_max_iter_final": 15,
}

# --- UTILITY FUNCTIONS ---

def find_optimal_k(X: np.ndarray, k_range: range, config: Dict[str, Any], w: int, output_dir: str) -> int:
    """Finds the optimal number of clusters for the global centroids."""
    print(f"Finding optimal K for global patterns (w={w})...")
    if len(X) < max(k_range):
        print(f"Not enough centroids ({len(X)}) to test up to k={max(k_range)}. Skipping.")
        return None
        
    inertias = [
        TimeSeriesKMeans(n_clusters=k, metric="dtw", max_iter=config["kmeans_max_iter_elbow"], random_state=42).fit(X).inertia_
        for k in k_range
    ]
    try:
        kl = KneeLocator(list(k_range), inertias, curve='convex', direction='decreasing')
        optimal_k = kl.elbow
        if optimal_k:
            print(f"Detected optimal global k = {optimal_k}")
            kl.plot_knee()
            plt.title(f"Elbow Plot for Global Patterns (w={w})")
            plt.savefig(os.path.join(output_dir, f"elbow_plot_global_w{w}.png"))
            plt.close()
        return optimal_k
    except Exception as e:
        print(f"Could not find optimal k for w={w}: {e}")
        return None

def visualize_global_clusters(
    global_centroids: np.ndarray,
    all_member_centroids: np.ndarray,
    global_cluster_assignments: np.ndarray,
    all_member_metadata: List[Tuple],
    window_size: int,
    output_path: str
):
    """Visualizes the discovered global pattern clusters."""
    optimal_k = len(global_centroids)
    print(f"Visualizing {optimal_k} global clusters for w={window_size}...")
    
    fig, axs = plt.subplots(optimal_k, 1, figsize=(14, 4 * optimal_k), sharex=True, squeeze=False)
    fig.suptitle(f'Global Pattern Clusters (Window Size = {window_size})', fontsize=20, y=1.0)

    for i in range(optimal_k):
        ax = axs[i, 0]
        mask = (global_cluster_assignments == i)
        cluster_members = all_member_centroids[mask]
        member_metadata = np.array(all_member_metadata)[mask]
        
        num_stocks = len(np.unique([meta[0] for meta in member_metadata]))

        ax.set_title(f"Global Cluster {i} ({len(cluster_members)} total patterns from {num_stocks} stocks)")
        for member in cluster_members:
            ax.plot(member, color='skyblue', alpha=0.5)
        ax.plot(np.ravel(global_centroids[i]), color='black', linewidth=3, label='Global Centroid')
        ax.legend()
        ax.grid(True, linestyle='--')

    plt.xlabel("Time Steps in Pattern")
    plt.tight_layout(rect=[0, 0.03, 1, 0.96])
    plt.savefig(output_path, dpi=150)
    plt.close(fig)
    print(f"Saved global cluster visualization to {output_path}")

def visualize_probability_distribution(
    prob_data: pd.DataFrame,
    global_pattern_map: Dict,
    analysis_type: str,
    horizon: int,
    window_size: int,
    output_path: str
):
    """Visualizes the probability distributions for each global cluster using box plots."""
    num_global_clusters = len(global_pattern_map)
    if num_global_clusters == 0: return

    fig, axs = plt.subplots(num_global_clusters, 1, figsize=(12, 6 * num_global_clusters), squeeze=False, sharey=True)
    fig.suptitle(f"Predictive Power Distribution\nAnalysis: {analysis_type} | Horizon: {horizon} | Window: {window_size}", fontsize=16)

    for i in range(num_global_clusters):
        ax = axs[i, 0]
        
        constituent_patterns = global_pattern_map.get(i, [])
        if not constituent_patterns:
            ax.set_title(f"Global Cluster {i} (No data available)")
            continue

        df_list = []
        for stock, cid in constituent_patterns:
            df_list.append(prob_data[
                (prob_data['stock'] == stock) &
                (prob_data['cluster_id'] == cid)
            ])
        
        if not df_list: continue
        
        cluster_prob_data = pd.concat(df_list)
        
        box_data = [
            cluster_prob_data['prob_up'].dropna(),
            cluster_prob_data['prob_down'].dropna(),
            cluster_prob_data['prob_flat'].dropna()
        ]
        
        labels = [f"Up\n(n={len(box_data[0])})", f"Down\n(n={len(box_data[1])})", f"Flat\n(n={len(box_data[2])})"]
        bp = ax.boxplot(box_data, labels=labels, patch_artist=True, medianprops={'color': 'black'})
        
        colors = ['mediumseagreen', 'lightcoral', 'lightskyblue']
        for patch, color in zip(bp['boxes'], colors):
            patch.set_facecolor(color)
            
        ax.set_title(f"Global Cluster {i}")
        ax.set_ylabel("Probability (%)")
        ax.set_ylim(0, 105)
        ax.grid(True, linestyle='--')

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.savefig(output_path, dpi=150)
    plt.close(fig)
    print(f"  -> Saved probability distribution plot for horizon {horizon}")

def save_global_pattern_summary(
    global_pattern_map: Dict,
    w_prob_data: pd.DataFrame,
    window_size: int,
    output_path: str
):
    """
    Saves a summary of each global cluster, including total pattern count and stock count.
    """
    print("Generating global pattern summary...")
    summary_data = []
    
    for global_cid, constituents in global_pattern_map.items():
        if not constituents:
            continue
            
        stock_names = [item[0] for item in constituents]
        num_unique_stocks = len(set(stock_names))
        
        total_patterns = 0
        for stock, original_cid in constituents:
            # Filter to get the row(s) for the specific stock/cluster
            pattern_rows = w_prob_data[
                (w_prob_data['stock'] == stock) &
                (w_prob_data['cluster_id'] == original_cid)
            ]
            if not pattern_rows.empty:
                # n_patterns is the same for all horizons of a given cluster
                total_patterns += pattern_rows['n_patterns'].iloc[0]

        summary_data.append({
            "window_size": window_size,
            "global_cluster_id": global_cid,
            "total_pattern_count": total_patterns,
            "num_contributing_stocks": num_unique_stocks,
            "contributing_stocks": ", ".join(sorted(list(set(stock_names))))
        })
        
    if summary_data:
        summary_df = pd.DataFrame(summary_data)
        summary_df = summary_df.sort_values(by="global_cluster_id").reset_index(drop=True)
        summary_df.to_csv(output_path, index=False)
        print(f"Saved global pattern summary to {output_path}")

def save_detailed_comparison_csv(
    global_pattern_map: Dict,
    all_probs_df: pd.DataFrame,
    window_size: int,
    output_path: str
):
    """
    Saves a detailed CSV mapping original stock/clusters to global clusters with their probabilities.
    """
    print("Generating detailed comparison CSV...")
    
    reverse_map = {}
    for global_cid, constituents in global_pattern_map.items():
        for stock, original_cid in constituents:
            reverse_map[(stock, original_cid)] = global_cid
            
    w_prob_data = all_probs_df[all_probs_df['window_size'] == window_size].copy()
    
    w_prob_data['global_cluster_id'] = w_prob_data.apply(
        lambda row: reverse_map.get((row['stock'], row['cluster_id']), -1),
        axis=1
    )
    
    cols = ['global_cluster_id', 'stock', 'cluster_id', 'window_size', 'analysis_type', 
            'horizon', 'n_patterns', 'prob_up', 'prob_down', 'prob_flat']
    
    detailed_df = w_prob_data[cols].sort_values(
        by=['global_cluster_id', 'analysis_type', 'horizon', 'stock']
    ).reset_index(drop=True)
    
    detailed_df.to_csv(output_path, index=False)
    print(f"Saved detailed comparison data to {output_path}")


# --- MAIN LOGIC ---

def main(config: Dict[str, Any]):
    # 1. Load all discovery artifacts
    artifact_paths = glob.glob(os.path.join(config["discovery_root_dir"], "*", "discovery_artifacts.npz"))
    if not artifact_paths:
        print(f"No artifact files found in '{config['discovery_root_dir']}'. Exiting.")
        return

    all_artifacts = {}
    for path in artifact_paths:
        stock_name = os.path.basename(os.path.dirname(path))
        all_artifacts[stock_name] = np.load(path, allow_pickle=True)
        print(f"Loaded artifacts for: {stock_name}")

    # 2. Load all analysis probability results
    prob_csv_paths = glob.glob(os.path.join(config["analysis_root_dir"], "*", "*", "w_*", "analysis_probabilities.csv"))
    if not prob_csv_paths:
        print(f"No 'analysis_probabilities.csv' files found in '{config['analysis_root_dir']}'.")
        print("Please run the modified analyze_patterns.py first. Exiting.")
        return
    
    all_probs_df = pd.concat([pd.read_csv(p) for p in prob_csv_paths], ignore_index=True)
    print(f"\nLoaded probability data from {len(prob_csv_paths)} files.")
    
    unique_window_sizes = sorted(list(all_probs_df['window_size'].unique()))
    unique_analysis_types = all_probs_df['analysis_type'].unique()
    
    print(f"Found unique window sizes: {unique_window_sizes}")
    
    # 4. Main loop for cross-stock comparison
    for w in unique_window_sizes:
        print(f"\n{'='*25} PROCESSING GLOBAL WINDOW SIZE: {w} {'='*25}")
        
        all_member_centroids = []
        all_member_centroids_metadata = []

        for stock, artifacts in all_artifacts.items():
            if w in artifacts['window_sizes']:
                idx = list(artifacts['window_sizes']).index(w)
                centroids = artifacts['centroids_list'][idx]
                k_val = artifacts['k_values'][idx]
                for i, centroid in enumerate(centroids):
                    all_member_centroids.append(np.ravel(centroid))
                    all_member_centroids_metadata.append((stock, k_val, i))

        if len(all_member_centroids) < max(config['k_range_for_global_clusters']):
            print(f"Not enough total centroids ({len(all_member_centroids)}) across all stocks for w={w}. Skipping.")
            continue
            
        X_global = np.array(all_member_centroids)
        w_output_dir = os.path.join(config['comparison_output_dir'], f"w_{w}")
        os.makedirs(w_output_dir, exist_ok=True)
        
        optimal_global_k = find_optimal_k(X_global, config['k_range_for_global_clusters'], config, w, w_output_dir)

        if not optimal_global_k:
            continue
            
        global_km = TimeSeriesKMeans(n_clusters=optimal_global_k, metric="dtw", max_iter=config["kmeans_max_iter_final"], random_state=42)
        global_cluster_assignments = global_km.fit_predict(X_global)
        global_centroids = global_km.cluster_centers_

        plot_path = os.path.join(w_output_dir, "global_cluster_shapes.png")
        visualize_global_clusters(global_centroids, X_global, global_cluster_assignments, all_member_centroids_metadata, w, plot_path)
        
        global_pattern_map = {i: [] for i in range(optimal_global_k)}
        for i, meta in enumerate(all_member_centroids_metadata):
            stock, _, original_cid = meta
            global_cid = global_cluster_assignments[i]
            global_pattern_map[global_cid].append((stock, original_cid))

        w_prob_data = all_probs_df[all_probs_df['window_size'] == w].copy()

        summary_path = os.path.join(w_output_dir, "global_pattern_summary.csv")
        save_global_pattern_summary(global_pattern_map, w_prob_data, w, summary_path)

        details_path = os.path.join(w_output_dir, "detailed_comparison.csv")
        save_detailed_comparison_csv(global_pattern_map, all_probs_df, w, details_path)
        
        for analysis_type in unique_analysis_types:
            analysis_prob_data = w_prob_data[w_prob_data['analysis_type'] == analysis_type]
            unique_horizons = sorted(analysis_prob_data['horizon'].unique())
            
            analysis_output_dir = os.path.join(w_output_dir, analysis_type.replace(' ', '_').lower())
            os.makedirs(analysis_output_dir, exist_ok=True)
            
            print(f"\nAnalyzing predictability for '{analysis_type}'...")
            for h in unique_horizons:
                h_prob_data = analysis_prob_data[analysis_prob_data['horizon'] == h]
                plot_path = os.path.join(analysis_output_dir, f"prob_dist_h_{h}.png")
                visualize_probability_distribution(
                    prob_data=h_prob_data,
                    global_pattern_map=global_pattern_map,
                    analysis_type=analysis_type,
                    horizon=h,
                    window_size=w,
                    output_path=plot_path
                )

    print(f"\n{'='*30} COMPARISON COMPLETE {'='*30}")
    print(f"All comparison results saved in: {config['comparison_output_dir']}")


if __name__ == "__main__":
    main(CONFIG)