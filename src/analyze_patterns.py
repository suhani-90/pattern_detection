# analyze_patterns.py 

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
import argparse
import csv 
from typing import Dict, Any, Callable

# Configuration 
CONFIG = {
    "results_dir": "output2/analysis_final", 
    "threshold_up": 2.0,
    "threshold_down": -2.0,
}

# AVG VS AVG
def calculate_avg_vs_avg(ts: np.ndarray, idx: int, window_size: int, p_end: int, h_end: int, **kwargs) -> float:
    """
    Calculates the PERCENTAGE difference between the average price in the horizon vs. the pattern.
    This version is robust against empty slices and division by zero.
    """
    # Define the slices for the pattern and the future horizon
    pattern_slice = ts[idx : idx + window_size]
    horizon_slice = ts[p_end + 1 : h_end + 1]

    # --- ROBUSTNESS CHECKS ---
    # Check if either slice is empty. If so, a meaningful comparison is impossible.
    if pattern_slice.size == 0 or horizon_slice.size == 0:
        return 0.0  # Return a neutral value

    # Calculate the average, which might result in `nan` if there are issues
    avg_pattern_price = np.mean(pattern_slice)
    avg_horizon_price = np.mean(horizon_slice)
    
    # Check for NaN values which can occur with problematic data
    if np.isnan(avg_pattern_price) or np.isnan(avg_horizon_price):
        return 0.0 # Return a neutral value

    # Check for division by zero
    if avg_pattern_price == 0:
        return 0.0

    # All checks passed, now calculate the percentage change
    percent_change = ((avg_horizon_price - avg_pattern_price) / avg_pattern_price) * 100
    
    # Final check to ensure the result is a normal number
    if not np.isfinite(percent_change):
        return 0.0 # Handles cases where percent_change becomes inf
        
    return percent_change
# 'end_vs_end' 
# def calculate_end_vs_end(ts: np.ndarray, p_end: int, h_end: int, **kwargs) -> float:
#     return ts[h_end] - ts[p_end]


# Analysis function 
def run_analysis(
    ts: np.ndarray, clusters: np.ndarray, original_indices: np.ndarray,
    window_size: int, optimal_k: int, analysis_func: Callable,
    analysis_name: str, output_base_dir: str, config: Dict[str, Any]
) -> list:
    
    horizons = [25, int(0.5 * window_size), int(1.0 * window_size), int(2.0 * window_size)]
    horizons_to_test = sorted(list(set([max(1, h) for h in horizons])))
    
    print(f"\n--- Running Analysis: {analysis_name} for Window Size: {window_size} ---")

    window_output_dir = os.path.join(output_base_dir, f"w_{window_size}")
    os.makedirs(window_output_dir, exist_ok=True)
    
    results_data_to_return = [] 

    for horizon in horizons_to_test:
        print(f"  Analyzing horizon: {horizon}...") # Added for clarity
        labels, prob_up_list, prob_down_list, prob_flat_list = [], [], [], []
        
        for i in range(optimal_k):
            indices_for_cluster = original_indices[clusters == i]
            
            # The calculation step
            outcomes = [
                analysis_func(ts=ts, idx=idx, window_size=window_size, p_end=idx + window_size - 1, h_end=idx + window_size - 1 + horizon)
                for idx in indices_for_cluster if idx + window_size - 1 + horizon < len(ts)
            ]
            
            # --- DIAGNOSTIC PRINT ---
            # If you still have issues, this line will show you the exact values being calculated.
            # print(f"    Cluster {i}, w={window_size}, h={horizon}: Found {len(outcomes)} outcomes. Sample: {outcomes[:5]}")

            if outcomes:
                outcomes_arr = np.array(outcomes)
                n_total = len(outcomes_arr)
                
                # This calculation should now be completely safe.
                n_up = np.sum(outcomes_arr >= config["threshold_up"])
                n_down = np.sum(outcomes_arr <= config["threshold_down"])
                n_flat = n_total - n_up - n_down
                
                p_up = (n_up / n_total) * 100 if n_total > 0 else 0
                p_down = (n_down / n_total) * 100 if n_total > 0 else 0
                p_flat = (n_flat / n_total) * 100 if n_total > 0 else 0
                
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
        
        if labels:
            fig, ax = plt.subplots(figsize=(16, 8))
            x = np.arange(len(labels)); width = 0.25; pos1, pos2, pos3 = x - width, x, x + width
            up_label = f'Prob. Up (>= {config["threshold_up"]}%)'; down_label = f'Prob. Down (<= {config["threshold_down"]}%)'; flat_label = 'Prob. Flat'
            title = f'{analysis_name} | Horizon: {horizon} steps (w={window_size}) | Thresholds: {config["threshold_up"]}% / {config["threshold_down"]}%'
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
    
    return results_data_to_return

# Main function
# Main function
def main(config: Dict[str, Any]):
    parser = argparse.ArgumentParser(description="Analyze pre-computed time series patterns.")
    # The --filepath is the path to the original CSV, e.g., "data/HDFCBANK.csv"
    parser.add_argument("--filepath", type=str, required=True, help="Path to the original CSV file.")
    # The --discovery_dir should be the top-level discovery output, e.g., "output/discovery"
    parser.add_argument("--discovery_dir", type=str, required=True, help="Path to the main discovery output directory.")
    args = parser.parse_args()

    # --- 1. Load Data ---
    try: 
        ts = pd.read_csv(args.filepath)['close'].astype(np.float64).values
    except Exception as e: 
        print(f"ERROR: Failed to load time series from {args.filepath}. Error: {e}")
        return

    # --- 2. Set up Correct Directory Structure ---
    # Extract just the filename without extension, e.g., "HDFCBANK"
    base_filename = os.path.splitext(os.path.basename(args.filepath))[0]

    # This is the path to the specific artifact file for this stock
    # e.g., "output/discovery/HDFCBANK/discovery_artifacts.npz"
    artifact_path = os.path.join(args.discovery_dir, base_filename, "discovery_artifacts.npz")
    if not os.path.exists(artifact_path): 
        print(f"ERROR: Artifact file not found at {artifact_path}.")
        return

    # This will be the main results directory for THIS stock's analysis
    # e.g., "output/analysis/HDFCBANK"
    stock_analysis_dir = os.path.join(config["results_dir"], base_filename)
    os.makedirs(stock_analysis_dir, exist_ok=True)
    print(f"Analysis results for '{base_filename}' will be saved in: {stock_analysis_dir}")
        
    print(f"Loading patterns from {artifact_path}...")
    artifacts = np.load(artifact_path, allow_pickle=True)

    # --- 3. Define and Run Analyses ---
    analyses_to_run = {
        "average_vs_average": calculate_avg_vs_avg
        # You can add more analysis types here in the future
    }
   
    all_analysis_data_for_csv = []

    # Loop through each discovery result (one per window size from the artifact file)
    for i in range(len(artifacts['window_sizes'])):
        window_size = artifacts['window_sizes'][i]
        k = artifacts['k_values'][i]
        clusters = artifacts['clusters_list'][i]
        indices = artifacts['indices_list'][i]

        # Loop through each type of analysis we want to run
        for name, func in analyses_to_run.items():
            # This directory is for the specific analysis type, inside the stock's folder
            # e.g., "output/analysis/HDFCBANK/average_vs_average"
            analysis_specific_output_dir = os.path.join(stock_analysis_dir, name)
            
            # Pass this specific, fully-formed path to the run_analysis function
            returned_data = run_analysis(
                ts=ts, 
                clusters=clusters, 
                original_indices=indices, 
                window_size=window_size,
                optimal_k=k, 
                analysis_func=func, 
                analysis_name=name.replace('_', ' ').title(),
                output_base_dir=analysis_specific_output_dir, # This is the critical change
                config=config 
            )
            all_analysis_data_for_csv.extend(returned_data)

    # --- 4. Save Summary CSV ---
    if all_analysis_data_for_csv:
        # Save the summary CSV inside the main directory for this stock
        output_csv_path = os.path.join(stock_analysis_dir, "analysis_summary.csv")
        try:
            with open(output_csv_path, 'w', newline='') as f:
                writer = csv.DictWriter(f, fieldnames=all_analysis_data_for_csv[0].keys())
                writer.writeheader()
                writer.writerows(all_analysis_data_for_csv)
            print(f"\nSuccessfully saved analysis summary to: {output_csv_path}")
        except Exception as e:
            print(f"\nERROR: Could not write summary file. {e}")

    print(f"\n{'='*30} ANALYSIS COMPLETE {'='*30}")
if __name__ == "__main__":
    main(CONFIG)