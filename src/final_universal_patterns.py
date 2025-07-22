# find_universal_patterns.py 
import numpy as np
import pandas as pd
import os
import glob
from sklearn.cluster import DBSCAN
from sklearn.neighbors import NearestNeighbors
from kneed import KneeLocator
import matplotlib.pyplot as plt


# Configuration
DISCOVERY_DIR = "output2/discovery"
ANALYSIS_DIR = "output2/analysis2"  
OUTPUT_DIR = "output/universal_analysis" 

#  DBSCAN's min_samples.
MIN_MEMBERS_IN_GROUP = 2 

#Load data 
def load_data(stock_names, discovery_dir, analysis_dir):
    all_centroids = []
    for stock in stock_names:
        artifact_path = os.path.join(discovery_dir, stock, "discovery_artifacts.npz")
        if not os.path.exists(artifact_path): continue
        artifacts = np.load(artifact_path, allow_pickle=True)
        for i, ws in enumerate(artifacts['window_sizes']):
            for j, centroid in enumerate(artifacts['centroids_list'][i]):
                all_centroids.append({
                    "stock": stock, "window_size": ws, "cluster_id": j, "centroid": centroid.ravel()
                })
    
    all_analyses = []
    for stock in stock_names:
        analysis_path = os.path.join(analysis_dir, stock, "analysis_summary.csv")
        if not os.path.exists(analysis_path): continue
        df = pd.read_csv(analysis_path)
        df['stock'] = stock
        all_analyses.append(df)

    if not all_centroids or not all_analyses: return pd.DataFrame(), pd.DataFrame()
    return pd.DataFrame(all_centroids), pd.concat(all_analyses, ignore_index=True)


# optimal epsilon for DBSCAN 
def find_optimal_eps(data, min_samples, output_dir, window_size):
    nn = NearestNeighbors(n_neighbors=min_samples)
    nn_fit = nn.fit(data)
    distances, _ = nn_fit.kneighbors(data)
    k_distances = np.sort(distances[:, min_samples - 1], axis=0)
    
    try:
        kl = KneeLocator(range(len(k_distances)), k_distances, curve='convex', direction='increasing')
        if kl.knee:
        
            kl.plot_knee()
            plt.xlabel("Points sorted by distance")
            plt.ylabel(f"{min_samples}-th Nearest Neighbor Distance (eps)")
            plt.title(f"DBSCAN Epsilon Knee Plot for w={window_size}")
            knee_plot_path = os.path.join(output_dir, f"w{window_size}_dbscan_knee_plot.png")
            plt.savefig(knee_plot_path, dpi=100)
            plt.close()
            print(f" Auto-detected optimal epsilon (eps) = {k_distances[kl.knee]:.3f}")
            return k_distances[kl.knee]
    except Exception as e:
        print(f"Knee detection failed: {e}. Falling back to percentile method.")
    
    return np.percentile(k_distances, 90)

# Visulization (Meta clusters) 
def visualize_universal_shape(group_df, output_dir):
    if group_df.empty: return

    meta_cluster_id = group_df['meta_cluster'].iloc[0]
    window_size = group_df['window_size'].iloc[0]
    
    plt.figure(figsize=(10, 6))
    avg_centroid = np.mean(np.stack(group_df['centroid'].values), axis=0)

    for _, row in group_df.iterrows():
        plt.plot(row['centroid'], alpha=0.5, label=f"{row['stock']} (c{row['cluster_id']})")
    
    plt.plot(avg_centroid, color='black', linewidth=3, label='Average Shape')
    
    plt.title(f"Universal Pattern Shape (Meta-Cluster {meta_cluster_id}, w={window_size})")
    plt.xlabel("Time Steps"); plt.ylabel("Z-Normalized Value")
    plt.legend(loc='upper right', bbox_to_anchor=(1.2, 1.0))
    plt.grid(True, linestyle='--')
    
    plot_path = os.path.join(output_dir, f"meta_cluster_{meta_cluster_id}_shape.png")
    plt.savefig(plot_path, dpi=120, bbox_inches='tight')
    plt.close()

# Probablity Comparison 
def visualize_performance_comparison(performance_df, meta_cluster_id, horizon, analysis_name, output_dir):
    if performance_df.empty: return

    window_size = performance_df['window_size'].iloc[0]
    
    plt.figure(figsize=(12, 7))
    x_labels = [f"{row['stock']}\n(c{row['cluster_id']})" for _, row in performance_df.iterrows()]
    colors = ['mediumseagreen', 'lightcoral', 'lightskyblue']
    
    performance_df.set_index('stock').plot(y=['prob_up', 'prob_down', 'prob_flat'], kind='bar', ax=plt.gca(), color=colors, width=0.8)

    plt.title(f"Performance of Universal Pattern (Meta-Cluster {meta_cluster_id})\n(w={window_size}, h={horizon}, analysis='{analysis_name}')")
    plt.ylabel("Probability (%)"); plt.xlabel("Member Stock and its Original Cluster ID")
    plt.xticks(ticks=range(len(x_labels)), labels=x_labels, rotation=45, ha="right")
    plt.ylim(0, 105); plt.grid(axis='y', linestyle='--'); plt.legend(["Prob. Up", "Prob. Down", "Prob. Flat"])
    

    analysis_name_safe = analysis_name.replace(' ', '_').lower()
    plot_path = os.path.join(output_dir, f"meta_cluster_{meta_cluster_id}_h{horizon}_{analysis_name_safe}_perf.png")
    plt.savefig(plot_path, dpi=150, bbox_inches='tight')
    plt.close()


if __name__ == "__main__":
    stock_names = [os.path.basename(p) for p in glob.glob(os.path.join(DISCOVERY_DIR, '*')) if os.path.isdir(p)]
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    print("--- Loading All Data ---")
    centroids_df, analysis_df = load_data(stock_names, DISCOVERY_DIR, ANALYSIS_DIR)
    
    if centroids_df.empty or analysis_df.empty:
        print("ERROR: Could not load data. Ensure discovery and analysis have been run for all stocks.")
        exit()

    final_summary_data = []

    for window_size in sorted(centroids_df['window_size'].unique()):
        print(f"\n{'='*20} PROCESSING WINDOW SIZE: {window_size} {'='*20}")
        
        ws_output_dir = os.path.join(OUTPUT_DIR, f"window_{window_size}")
        os.makedirs(ws_output_dir, exist_ok=True)
        
        print("Stage 1: Clustering centroids to find universal shapes...")
        ws_centroids = centroids_df[centroids_df['window_size'] == window_size].copy()
        X = np.stack(ws_centroids['centroid'].values)
        
        if len(X) < MIN_MEMBERS_IN_GROUP:
            print(" Not enough centroids to form groups. Skipping.")
            continue
        
        optimal_eps = find_optimal_eps(X, MIN_MEMBERS_IN_GROUP, ws_output_dir, window_size)
        db = DBSCAN(eps=optimal_eps, min_samples=MIN_MEMBERS_IN_GROUP).fit(X)
        ws_centroids['meta_cluster'] = db.labels_
        
        universal_patterns_df = ws_centroids[ws_centroids['meta_cluster'] != -1].copy()
        
        num_groups = universal_patterns_df['meta_cluster'].nunique()
        if num_groups == 0:
            print(" No universal pattern groups found for this window size.")
            continue
        print(f"Found {num_groups} universal pattern groups.")

        print("Stage 2: Analyzing and visualizing each universal group...")
        full_performance_df = pd.merge(universal_patterns_df, analysis_df, on=['stock', 'window_size', 'cluster_id'])

        for mc_id in sorted(full_performance_df['meta_cluster'].unique()):
            group_df = full_performance_df[full_performance_df['meta_cluster'] == mc_id]
            
            shape_df = group_df.drop_duplicates(subset=['stock', 'cluster_id'])
            visualize_universal_shape(shape_df, ws_output_dir)

        
            if group_df.empty:
                continue
            
    
            analysis_name = group_df['analysis_name'].iloc[0]

            for horizon in sorted(group_df['horizon'].unique()):
                perf_df = group_df[group_df['horizon'] == horizon]
                
        
                visualize_performance_comparison(perf_df, mc_id, horizon, analysis_name, ws_output_dir)
                
                if not perf_df.empty:
                    member_details = ", ".join([f"{row['stock']}(c{row['cluster_id']})" for _, row in perf_df.iterrows()])
                    final_summary_data.append({
                        "window_size": window_size,
                        "meta_cluster_id": mc_id,
                        "analysis_name": analysis_name, 
                        "horizon": horizon,
                        "avg_prob_up": perf_df['prob_up'].mean(),
                        "std_prob_up": perf_df['prob_up'].std(),
                        "avg_prob_down": perf_df['prob_down'].mean(),
                        "std_prob_down": perf_df['prob_down'].std(),
                        "num_members": len(perf_df),
                        "member_details": member_details
                    })
  
    if final_summary_data:
        print(f"\n{'='*30} CREATING FINAL REPORTS {'='*30}")
        report_df = pd.DataFrame(final_summary_data)
        
        report_df = report_df.sort_values(
            by=['num_members', 'avg_prob_up', 'std_prob_up'],
            ascending=[False, False, True]
        ).reset_index(drop=True)
        
        csv_path = os.path.join(OUTPUT_DIR, "universal_patterns_summary.csv")
        report_df.to_csv(csv_path, index=False, float_format='%.2f')
        print(f"Final summary report saved to: {csv_path}")

        txt_report_path = os.path.join(OUTPUT_DIR, "universal_patterns_report.txt")
        with open(txt_report_path, 'w') as f:
            f.write("--- Summary of Top Universal Patterns (Analysis: Average Vs Average) ---\n\n")
            f.write(report_df.to_string())
        print(f"Text summary saved to: {txt_report_path}")

    print(f"\n{'='*30} ANALYSIS COMPLETE {'='*30}")
    print(f"All outputs saved in: {OUTPUT_DIR}")