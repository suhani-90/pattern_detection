# analyze_patterns.py 

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
import argparse
from typing import Dict, Any, Callable

#configuration
CONFIG = {
    "results_dir": "output/analysis" 
}

# 3 types of analysis

# 1.PCT Change
def calculate_pct_change(ts: np.ndarray, p_end: int, h_end: int, **kwargs) -> float:
    price_pattern_end = ts[p_end]
    if price_pattern_end > 1e-9:
        return ((ts[h_end] - price_pattern_end) / price_pattern_end) * 100
    return 0.0

# 2 AVG Change
def calculate_avg_vs_avg(ts: np.ndarray, idx: int, window_size: int, p_end: int, h_end: int, **kwargs) -> float:
    avg_pattern_price = np.mean(ts[idx : idx + window_size])
    avg_horizon_price = np.mean(ts[p_end + 1 : h_end + 1])
    return avg_horizon_price - avg_pattern_price

# 3 Simple Close price change
def calculate_end_vs_end(ts: np.ndarray, p_end: int, h_end: int, **kwargs) -> float:
    return ts[h_end] - ts[p_end]


def run_analysis(
    ts: np.ndarray, clusters: np.ndarray, original_indices: np.ndarray,
    window_size: int, optimal_k: int, analysis_func: Callable,
    analysis_name: str, output_base_dir: str
):
    horizons = [25, int(0.5 * window_size), int(1.0 * window_size), int(2.0 * window_size)]
    horizons_to_test = sorted(list(set([max(1, h) for h in horizons])))
    
    print(f"\n--- Running Analysis: {analysis_name} for Window Size: {window_size} ---")

    window_output_dir = os.path.join(output_base_dir, f"w_{window_size}")
    os.makedirs(window_output_dir, exist_ok=True)

    for horizon in horizons_to_test:
        labels, prob_up_list, prob_down_list = [], [], []
        for i in range(optimal_k):
            indices_for_cluster = original_indices[clusters == i]
            outcomes = [
                analysis_func(ts=ts, idx=idx, window_size=window_size, p_end=idx + window_size - 1, h_end=idx + window_size - 1 + horizon)
                for idx in indices_for_cluster if idx + window_size - 1 + horizon < len(ts)
            ]
            if outcomes:
                outcomes_arr = np.array(outcomes)
                n_up, n_down, n_total = np.sum(outcomes_arr > 0), np.sum(outcomes_arr < 0), len(outcomes_arr)
                p_up, p_down = (n_up / n_total) * 100, (n_down / n_total) * 100
                labels.append(f"Cluster {i}"); prob_up_list.append(p_up); prob_down_list.append(p_down)
        
        if labels:
            fig, ax = plt.subplots(figsize=(14, 7))
            x, width = np.arange(len(labels)), 0.35
            rects1 = ax.bar(x - width/2, prob_up_list, width, label='Prob. Up/Positive', color='mediumseagreen')
            rects2 = ax.bar(x + width/2, prob_down_list, width, label='Prob. Down/Negative', color='lightcoral')
            ax.axhline(50, color='royalblue', linestyle='--', label='50% Chance')
            ax.set_ylabel('Probability (%)'); ax.set_ylim(0, 105)
            ax.set_title(f'{analysis_name} | Horizon: {horizon} steps (Window={window_size})')
            ax.set_xticks(x); ax.set_xticklabels(labels, rotation=45, ha="right")
            ax.legend(); ax.bar_label(rects1, fmt='%.1f%%'); ax.bar_label(rects2, fmt='%.1f%%')
            fig.tight_layout()
            plot_filename = os.path.join(window_output_dir, f"h_{horizon}.png")
            plt.savefig(plot_filename, dpi=150)
            plt.close(fig)
            print(f"  -> Saved plot for horizon {horizon}")


# Main
def main(config: Dict[str, Any]):
    parser = argparse.ArgumentParser(description="Analyze pre-computed time series patterns.")
    parser.add_argument("--filepath", type=str, required=True, help="Path to the original CSV file.")
    parser.add_argument("--discovery_dir", type=str, required=True, help="Path to the discovery_output directory.")
    args = parser.parse_args()

    try:
        ts = pd.read_csv(args.filepath)['close'].astype(np.float64).values
    except Exception as e:
        print(f"Failed to load original time series from {args.filepath}. Error: {e}")
        return

    base_filename = os.path.splitext(os.path.basename(args.filepath))[0]
    file_results_dir = os.path.join(config["results_dir"], base_filename)
    
   
    artifact_path = os.path.join(args.discovery_dir, base_filename, "discovery_artifacts.npz")
    if not os.path.exists(artifact_path):
        print(f"Artifact file not found at {artifact_path}. Please run discover_patterns.py first.")
        return
        
    print(f"Loading patterns from {artifact_path}")
    artifacts = np.load(artifact_path, allow_pickle=True)

   
    analyses_to_run = {
        "percentage_change": calculate_pct_change,
        "average_vs_average": calculate_avg_vs_avg,
        "end_price_vs_end_price": calculate_end_vs_end
    }

   
    num_sets = len(artifacts['window_sizes'])
    print(f"Found {num_sets} sets of patterns to analyze.")

    for i in range(num_sets):
        window_size = artifacts['window_sizes'][i]
        k = artifacts['k_values'][i]
        clusters = artifacts['clusters_list'][i]
        indices = artifacts['indices_list'][i]

        for name, func in analyses_to_run.items():
            analysis_output_dir = os.path.join(file_results_dir, name)
            run_analysis(
                ts=ts,
                clusters=clusters,
                original_indices=indices,
                window_size=window_size,
                optimal_k=k,
                analysis_func=func,
                analysis_name=name.replace('_', ' ').title(),
                output_base_dir=analysis_output_dir
            )

    print(f"\n{'='*30} ANALYSIS COMPLETE {'='*30}")
    print(f"All results saved in: {file_results_dir}")


if __name__ == "__main__":
    main(CONFIG)