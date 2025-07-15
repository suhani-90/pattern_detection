# analyze_patterns.py (MODIFIED TO SAVE RESULTS)
# THIS IS THE CORRECT VERSION TO USE FOR STEP 2 OF THE WORKFLOW

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
import argparse
import csv # Import CSV module
from typing import Dict, Any, Callable

# Configuration from your original script
CONFIG = {
    # This will create the folder structure you want, e.g., output/analysis2/
    "results_dir": "output/analysis2", 
    "threshold_up": 2.0,
    "threshold_down": -2.0,
}

# Your original calculation functions (unchanged)
def calculate_avg_vs_avg(ts: np.ndarray, idx: int, window_size: int, p_end: int, h_end: int, **kwargs) -> float:
    avg_pattern_price = np.mean(ts[idx : idx + window_size])
    avg_horizon_price = np.mean(ts[p_end + 1 : h_end + 1])
    return avg_horizon_price - avg_pattern_price

def calculate_end_vs_end(ts: np.ndarray, p_end: int, h_end: int, **kwargs) -> float:
    return ts[h_end] - ts[p_end]


# The analysis function is now modified to RETURN the calculated data
def run_analysis(
    ts: np.ndarray, clusters: np.ndarray, original_indices: np.ndarray,
    window_size: int, optimal_k: int, analysis_func: Callable,
    analysis_name: str, output_base_dir: str, config: Dict[str, Any]
) -> list: # <-- IT NOW RETURNS A LIST
    
    horizons = [25, int(0.5 * window_size), int(1.0 * window_size), int(2.0 * window_size)]
    horizons_to_test = sorted(list(set([max(1, h) for h in horizons])))
    
    print(f"\n--- Running Analysis: {analysis_name} for Window Size: {window_size} ---")

    window_output_dir = os.path.join(output_base_dir, f"w_{window_size}")
    os.makedirs(window_output_dir, exist_ok=True)
    
    # <<< NEW: This list will store the data we need to save >>>
    results_data_to_return = [] 

    for horizon in horizons_to_test:
        labels, prob_up_list, prob_down_list, prob_flat_list = [], [], [], []
        
        for i in range(optimal_k):
            indices_for_cluster = original_indices[clusters == i]
            outcomes = [
                analysis_func(ts=ts, idx=idx, window_size=window_size, p_end=idx + window_size - 1, h_end=idx + window_size - 1 + horizon)
                for idx in indices_for_cluster if idx + window_size - 1 + horizon < len(ts)
            ]
            
            if outcomes:
                outcomes_arr = np.array(outcomes)
                n_total = len(outcomes_arr)
                n_up = np.sum(outcomes_arr >= config["threshold_up"])
                n_down = np.sum(outcomes_arr <= config["threshold_down"])
                n_flat = n_total - n_up - n_down
                
                p_up = (n_up / n_total) * 100 if n_total > 0 else 0
                p_down = (n_down / n_total) * 100 if n_total > 0 else 0
                p_flat = (n_flat / n_total) * 100 if n_total > 0 else 0
                
                # <<< NEW: Add the calculated data to our list >>>
                results_data_to_return.append({
                    "analysis_name": analysis_name,
                    "window_size": window_size,
                    "cluster_id": i,
                    "horizon": horizon,
                    "prob_up": p_up,
                    "prob_down": p_down,
                    "prob_flat": p_flat,
                    "num_occurrences": n_total
                })

                labels.append(f"Cluster {i}")
                prob_up_list.append(p_up)
                prob_down_list.append(p_down)
                prob_flat_list.append(p_flat)
        
        # This plotting block is identical to your original code. It still works.
        if labels:
            fig, ax = plt.subplots(figsize=(16, 8))
            x = np.arange(len(labels)); width = 0.25; pos1, pos2, pos3 = x - width, x, x + width
            up_label = f'Prob. Up (>= {config["threshold_up"]})'; down_label = f'Prob. Down (<= {config["threshold_down"]})'; flat_label = 'Prob. Flat'
            title = f'{analysis_name} | Horizon: {horizon} steps (w={window_size}) | Thresholds: {config["threshold_up"]} / {config["threshold_down"]}'
            rects1 = ax.bar(pos1, prob_up_list, width, label=up_label, color='mediumseagreen')
            rects2 = ax.bar(pos2, prob_down_list, width, label=down_label, color='lightcoral')
            rects3 = ax.bar(pos3, prob_flat_list, width, label=flat_label, color='lightskyblue')
            ax.axhline(50, color='royalblue', linestyle='--', label='50% Chance')
            ax.set_ylabel('Probability (%)'); ax.set_ylim(0, 105); ax.set_title(title, fontsize=16)
            ax.set_xticks(x); ax.set_xticklabels(labels, rotation=45, ha="right"); ax.legend()
            ax.bar_label(rects1, fmt='%.1f%%', padding=3); ax.bar_label(rects2, fmt='%.1f%%', padding=3); ax.bar_label(rects3, fmt='%.1f%%', padding=3)
            fig.tight_layout()
            plot_filename = os.path.join(window_output_dir, f"h_{horizon}.png")
            plt.savefig(plot_filename, dpi=150)
            plt.close(fig)
            print(f"  -> Saved plot for horizon {horizon}")
    
    # <<< NEW: Return the list of all data calculated in this function >>>
    return results_data_to_return

# Main function
def main(config: Dict[str, Any]):
    parser = argparse.ArgumentParser(description="Analyze pre-computed time series patterns.")
    parser.add_argument("--filepath", type=str, required=True, help="Path to the original CSV file.")
    parser.add_argument("--discovery_dir", type=str, required=True, help="Path to the discovery_output directory.")
    args = parser.parse_args()

    try: ts = pd.read_csv(args.filepath)['close'].astype(np.float64).values
    except Exception as e: print(f"Failed to load time series from {args.filepath}. Error: {e}"); return

    base_filename = os.path.splitext(os.path.basename(args.filepath))[0]
    file_results_dir = os.path.join(config["results_dir"], base_filename)
    
    artifact_path = os.path.join(args.discovery_dir, base_filename, "discovery_artifacts.npz")
    if not os.path.exists(artifact_path): print(f"Artifact file not found at {artifact_path}."); return
        
    print(f"Loading patterns from {artifact_path}")
    artifacts = np.load(artifact_path, allow_pickle=True)

    analyses_to_run = {"average_vs_average": calculate_avg_vs_avg, "end_price_vs_end_price": calculate_end_vs_end}
   
    # <<< NEW: This list will collect ALL results for this stock before saving >>>
    all_analysis_data_for_csv = []

    for i in range(len(artifacts['window_sizes'])):
        window_size = artifacts['window_sizes'][i]
        k = artifacts['k_values'][i]
        clusters = artifacts['clusters_list'][i]
        indices = artifacts['indices_list'][i]

        for name, func in analyses_to_run.items():
            analysis_output_dir = os.path.join(file_results_dir, name)
            
            # <<< MODIFIED: Capture the data that the function returns >>>
            returned_data = run_analysis(
                ts=ts, clusters=clusters, original_indices=indices, window_size=window_size,
                optimal_k=k, analysis_func=func, analysis_name=name.replace('_', ' ').title(),
                output_base_dir=analysis_output_dir, config=config 
            )
            # Add the results to our master list
            all_analysis_data_for_csv.extend(returned_data)

    # <<< NEW: This whole block saves the crucial summary file >>>
    if all_analysis_data_for_csv:
        # We save it in the top-level folder for this stock's analysis
        output_csv_path = os.path.join(file_results_dir, "analysis_summary.csv")
        os.makedirs(os.path.dirname(output_csv_path), exist_ok=True)
        try:
            with open(output_csv_path, 'w', newline='') as f:
                writer = csv.DictWriter(f, fieldnames=all_analysis_data_for_csv[0].keys())
                writer.writeheader()
                writer.writerows(all_analysis_data_for_csv)
            print(f"\nSuccessfully saved analysis summary to: {output_csv_path}")
        except Exception as e:
            print(f"\nERROR: Could not write summary file. {e}")


    print(f"\n{'='*30} ANALYSIS COMPLETE {'='*30}")
    print(f"All plot results saved in: {file_results_dir}")


if __name__ == "__main__":
    main(CONFIG)