# find_universal_patterns.py (with Meta-Cluster Visualization and Traceability)
import numpy as np
import pandas as pd
import os
import glob
from sklearn.cluster import DBSCAN
from sklearn.neighbors import NearestNeighbors
from kneed import KneeLocator
import matplotlib.pyplot as plt
import seaborn as sns

# --- Configuration ---
DISCOVERY_DIR = "output/discovery"
ANALYSIS_DIR = "output/analysis2" # Make sure this matches your analysis output folder
OUTPUT_DIR = "output/universal_analysis"
DBSCAN_MIN_SAMPLES = 2
NUM_TOP_PATTERNS_TO_REPORT = 5

# --- Function Definitions (load_all_data, find_optimal_eps are mostly unchanged) ---

def load_all_data(stock_names, discovery_dir, analysis_dir):
    """Load all centroids from discovery and all probability results from analysis."""
    all_centroids_data = []
    all_analyses_dfs = []

    for stock in stock_names:
        artifact_path = os.path.join(discovery_dir, stock, "discovery_artifacts.npz")
        if not os.path.exists(artifact_path): continue
        artifacts = np.load(artifact_path, allow_pickle=True)
        for i, window_size in enumerate(artifacts['window_sizes']):
            centroids = artifacts['centroids_list'][i]
            k_value = artifacts['k_values'][i]
            for j in range(k_value):
                centroid_shape = centroids[j].ravel()
                all_centroids_data.append({
                    "stock": stock, "window_size": window_size, "cluster_id": j, "centroid": centroid_shape
                })
    
    for stock in stock_names:
        analysis_path = os.path.join(analysis_dir, stock, "analysis_summary.csv")
        if not os.path.exists(analysis_path): continue
        df = pd.read_csv(analysis_path)
        df['stock'] = stock
        all_analyses_dfs.append(df)

    if not all_centroids_data or not all_analyses_dfs: return pd.DataFrame(), pd.DataFrame()
    centroids_df = pd.DataFrame(all_centroids_data)
    analysis_df = pd.concat(all_analyses_dfs, ignore_index=True)
    return centroids_df, analysis_df


def find_optimal_eps(data, min_samples):
    """Find the optimal epsilon for DBSCAN using the k-distance graph."""
    nn = NearestNeighbors(n_neighbors=min_samples)
    nn_fit = nn.fit(data)
    distances, _ = nn_fit.kneighbors(data)
    distances = np.sort(distances[:, min_samples - 1], axis=0)
    try:
        kl = KneeLocator(range(len(distances)), distances, curve='convex', direction='increasing')
        if kl.knee: return distances[kl.knee]
    except: pass
    return np.percentile(distances, 85) * 1.2

# <<< NEW FUNCTION: To visualize the shapes in a meta-cluster >>>
def visualize_meta_cluster(df_meta_cluster, output_dir):
    """Plots all member centroids of a universal pattern group."""
    if df_meta_cluster.empty:
        return

    window_size = df_meta_cluster['window_size'].iloc[0]
    meta_cluster_id = df_meta_cluster['meta_cluster'].iloc[0]
    num_members = len(df_meta_cluster)

    plt.figure(figsize=(10, 6))
    
    # Calculate the average shape of the meta-cluster
    avg_centroid = np.mean(np.stack(df_meta_cluster['centroid'].values), axis=0)

    # Plot individual member centroids lightly
    for _, row in df_meta_cluster.iterrows():
        label = f"{row['stock']} (Cluster {row['cluster_id']})"
        plt.plot(row['centroid'], alpha=0.4, label=label)
    
    # Plot the average shape boldly
    plt.plot(avg_centroid, color='black', linewidth=3.5, label='Average Shape (Meta-Centroid)')

    plt.title(f'Universal Pattern Shape: Meta-Cluster {meta_cluster_id}\n(Window Size: {window_size}, {num_members} members)', fontsize=16)
    plt.xlabel("Time Steps in Pattern")
    plt.ylabel("Z-Normalized Value")
    plt.grid(True, linestyle='--', alpha=0.6)
    
    # To avoid overcrowding the legend, we can disable it or show a limited version
    # For now, let's just show the average shape legend
    # plt.legend() # This would be too crowded
    
    # Create a directory for these plots
    meta_cluster_plot_dir = os.path.join(output_dir, f"w_{window_size}_meta_cluster_shapes")
    os.makedirs(meta_cluster_plot_dir, exist_ok=True)
    
    plot_path = os.path.join(meta_cluster_plot_dir, f"meta_cluster_{meta_cluster_id}.png")
    plt.savefig(plot_path, dpi=120, bbox_inches='tight')
    plt.close()
    print(f"  -> Saved meta-cluster shape visualization to: {plot_path}")


# <<< MODIFIED FUNCTION: generate_final_report to include traceability >>>
def generate_final_report(report_df, merged_df):
    report_string = ""
    for analysis_name in sorted(report_df['analysis_name'].unique()):
        analysis_specific_report = report_df[report_df['analysis_name'] == analysis_name]
        top_patterns = analysis_specific_report.head(NUM_TOP_PATTERNS_TO_REPORT)
        
        report_string += f"\n{'='*50}\n"
        report_string += f" DRILL-DOWN REPORT FOR ANALYSIS: {analysis_name.upper()}\n"
        report_string += f"{'='*50}\n\n"

        if top_patterns.empty:
            report_string += "No universal patterns found for this analysis type.\n"
            continue

        for i, (_, pattern) in enumerate(top_patterns.iterrows()):
            ws = int(pattern['window_size'])
            mc = int(pattern['meta_cluster'])
            hz = int(pattern['horizon'])
            
            report_string += f"--- Top Pattern #{i+1} ---\n"
            report_string += f"  - Identifier: window={ws}, meta_cluster={mc}, horizon={hz}\n"
            report_string += f"  - Shape found in {pattern['num_members']} stocks.\n"
            report_string += f"  - Avg Prob Up: {pattern['avg_prob_up']:.1f}% (Consistency/Std Dev: {pattern['std_prob_up']:.1f}%)\n"
            report_string += "  - Member Performance Breakdown (Stock | Original Cluster ID | Probabilities):\n" # Modified header
            
            members = merged_df[
                (merged_df['analysis_name'] == analysis_name) &
                (merged_df['window_size'] == ws) &
                (merged_df['meta_cluster'] == mc) &
                (merged_df['horizon'] == hz)
            ]
            
            for _, member in members.iterrows():
                # <<< THIS LINE ADDS THE TRACEABILITY >>>
                report_string += f"    - {member['stock']:<20} | Cluster {member['cluster_id']:<2} | Prob Up: {member['prob_up']:>5.1f}% | Prob Down: {member['prob_down']:>5.1f}%\n"
            report_string += "\n"

            if i == 0:
                plt.figure(figsize=(12, 7))
                # Create a more informative label for the bar plot
                bar_labels = [f"{row['stock']}\n(Cluster {row['cluster_id']})" for _, row in members.iterrows()]
                sns.barplot(x=bar_labels, y=members['prob_up'], palette='viridis')
                plt.axhline(y=pattern['avg_prob_up'], color='r', linestyle='--', label=f"Average Prob_Up ({pattern['avg_prob_up']:.1f}%)")
                title_an_name = analysis_name.replace(' ', '_').replace('vs', 'vs')
                title = f"Breakdown of #1 Universal Pattern ({title_an_name})\n(w={ws}, meta_cluster={mc}, h={hz})"
                plt.title(title, fontsize=16)
                plt.ylabel("Probability of Price Up (%)"); plt.xlabel("Member Stock and its Original Cluster ID"); plt.ylim(0, 105); plt.legend()
                plt.xticks(rotation=0, ha="center")
                plt.grid(axis='y', linestyle='--', alpha=0.7)
                plot_path = os.path.join(OUTPUT_DIR, f"top_pattern_breakdown_{title_an_name}_w{ws}_h{hz}.png")
                plt.savefig(plot_path, dpi=150, bbox_inches='tight')
                plt.close()
                print(f"Saved visualization for the #1 pattern of '{analysis_name}' to {plot_path}")
                
    return report_string


if __name__ == "__main__":
    stock_names = [os.path.basename(p) for p in glob.glob(os.path.join(DISCOVERY_DIR, '*')) if os.path.isdir(p)]
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    print("--- Stage 0: Loading all discovery and analysis data ---")
    centroids_df, analysis_df = load_all_data(stock_names, DISCOVERY_DIR, ANALYSIS_DIR)
    
    if centroids_df.empty or analysis_df.empty:
        print("\nERROR: Could not load sufficient data. Exiting.")
        exit()

    final_report_dfs = []
    all_merged_data_for_reporting = []

    for window_size in sorted(centroids_df['window_size'].unique()):
        print(f"\n--- Stage 1: Clustering centroids for window size: {window_size} ---")
        ws_centroids_df = centroids_df[centroids_df['window_size'] == window_size].copy()
        X = np.stack(ws_centroids_df['centroid'].values)
        if len(X) < DBSCAN_MIN_SAMPLES: continue
            
        optimal_eps = find_optimal_eps(X, DBSCAN_MIN_SAMPLES)
        db = DBSCAN(eps=optimal_eps, min_samples=DBSCAN_MIN_SAMPLES).fit(X)
        ws_centroids_df['meta_cluster'] = db.labels_
        
        universal_patterns = ws_centroids_df[ws_centroids_df['meta_cluster'] != -1].copy()
        num_meta_clusters = universal_patterns['meta_cluster'].nunique()
        print(f"Found {num_meta_clusters} universal shape groups for window size {window_size}.")

        # <<< NEW BLOCK: Loop through found meta-clusters and visualize them >>>
        if num_meta_clusters > 0:
            print(f"Visualizing the {num_meta_clusters} universal shapes found...")
            for mc_id in sorted(universal_patterns['meta_cluster'].unique()):
                cluster_to_plot = universal_patterns[universal_patterns['meta_cluster'] == mc_id]
                visualize_meta_cluster(cluster_to_plot, OUTPUT_DIR)
        # <<< END OF NEW BLOCK >>>
        
        if universal_patterns.empty: continue
        
        print(f"--- Stage 2: Evaluating performance for window size: {window_size} ---")
        merged_df = pd.merge(universal_patterns, analysis_df, on=['stock', 'window_size', 'cluster_id'])
        all_merged_data_for_reporting.append(merged_df)
        
        # <<< MODIFIED: Add original cluster details to the final report >>>
        # We need to create a string representation of the members for the CSV
        def create_member_string(df_group):
            return ", ".join([f"{row['stock']}(c{row['cluster_id']})" for _, row in df_group.iterrows()])

        report_df = merged_df.groupby(['analysis_name', 'window_size', 'meta_cluster', 'horizon']).agg(
            avg_prob_up=('prob_up', 'mean'), std_prob_up=('prob_up', 'std'),
            avg_prob_down=('prob_down', 'mean'), std_prob_down=('prob_down', 'std'),
            num_members=('stock', 'nunique'),
            # This lambda is a bit more complex now, it captures both stock and cluster_id
            member_details=('stock', lambda x: list(zip(merged_df.loc[x.index, 'stock'], merged_df.loc[x.index, 'cluster_id'])))
        ).reset_index()

        # Reformat the member_details for readability in the CSV
        report_df['member_details'] = report_df['member_details'].apply(
            lambda details: ", ".join([f"{stock}(c{cid})" for stock, cid in sorted(details)])
        )

        report_df = report_df[report_df['num_members'] >= DBSCAN_MIN_SAMPLES].copy()
        if not report_df.empty: final_report_dfs.append(report_df)

    if final_report_dfs:
        print("\n--- Stage 3: Compiling final reports ---")
        full_report_df = pd.concat(final_report_dfs, ignore_index=True)
        # Rename 'member_details' to something more intuitive for the final CSV
        full_report_df = full_report_df.rename(columns={'member_details': 'member_stock_and_cluster_id'})
        
        # Adjust sorting to include the new column
        full_report_df = full_report_df.sort_values(
            by=['analysis_name', 'num_members', 'avg_prob_up', 'std_prob_up'],
            ascending=[True, False, False, True]
        )

        csv_path = os.path.join(OUTPUT_DIR, "universal_patterns_summary_report.csv")
        full_report_df.to_csv(csv_path, index=False)
        print(f"\nSUCCESS: Main summary report saved to {csv_path}")

        full_merged_df = pd.concat(all_merged_data_for_reporting, ignore_index=True)
        drilldown_text = generate_final_report(full_report_df, full_merged_df)
        
        report_path = os.path.join(OUTPUT_DIR, "universal_patterns_drilldown_report.txt")
        with open(report_path, 'w') as f: f.write(drilldown_text)
        print(f"Detailed drill-down reports saved to {report_path}")

        print("\n--- Final Report Summary ---")
        print(drilldown_text)
    else:
        print("\n\nCould not find any universal patterns that met the specified criteria.")