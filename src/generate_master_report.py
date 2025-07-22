# generate_master_report.py 
import numpy as np
import pandas as pd
import os
import argparse
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List
import glob 

# --- Configuration ---
CONFIG = {
    "data_dir": "data",
    "discovery_dir": "output2/discovery",
    "output_dir": "output2/HOPE_Simplified_Final", # New, clean output folder
    "timestamp_col": "timestamp",
}

# --- HELPER FUNCTIONS ---

def load_data_with_datetime(filepath: str, timestamp_col: str) -> pd.DataFrame:
    """Loads data and creates a DataFrame guaranteed to be aligned with the discovery script."""
    print(f"Loading data from {filepath}...")
    try:
        raw_df = pd.read_csv(filepath)
        close_prices = raw_df['close'].astype(np.float64).values
        timestamps = pd.to_datetime(raw_df[timestamp_col], utc=True)
        aligned_df = pd.DataFrame(data={'close': close_prices}, index=timestamps)
        aligned_df.dropna(subset=['close'], inplace=True)
        return aligned_df
    except Exception as e:
        print(f"ERROR: Could not load data from '{filepath}'. Error: {e}")
        return None

def plot_time_series(counts_series: pd.Series, title: str, ylabel: str, plot_type: str, output_path: str):
    """A generic function to plot line or bar charts for time series data."""
    if counts_series.empty: return
    plt.figure(figsize=(20, 7))
    if plot_type == 'line':
        counts_series.plot(kind='line', color='royalblue', alpha=0.8)
    else: # bar chart
        # --- MODIFICATION 1: Format x-axis labels for bar charts ---
        # Create a temporary series with the index formatted as 'YYYY-MM-DD' strings for cleaner labels.
        plot_series = counts_series.copy()
        plot_series.index = plot_series.index.strftime('%Y-%m-%d')
        plot_series.plot(kind='bar', color='darkcyan', width=0.8)
        # --- END MODIFICATION 1 ---

        if len(counts_series) > 50:
             ax = plt.gca()
             ax.xaxis.set_major_locator(plt.MaxNLocator(50))
        plt.xticks(rotation=45, ha="right")
    plt.title(title, fontsize=16); plt.ylabel(ylabel); plt.xlabel("Date")
    plt.grid(True, axis='y', linestyle='--', alpha=0.7); plt.tight_layout()
    plt.savefig(output_path, dpi=150); plt.close()

def analyze_seasonality(weekly_counts: pd.Series, output_path: str, title: str) -> List[str]:
    """Creates a monthly boxplot and returns seasonality conclusions."""
    if weekly_counts.empty: return ["No data for seasonality analysis."]
    df = weekly_counts.to_frame(name='count'); df['month'] = df.index.month
    plt.figure(figsize=(12, 7))
    sns.boxplot(x='month', y='count', data=df, palette='viridis')
    plt.title(title); plt.xlabel("Month of the Year"); plt.ylabel("Weekly Occurrences")
    plt.grid(axis='y', linestyle='--', alpha=0.7); plt.savefig(output_path); plt.close()
    monthly_avg = df.groupby('month')['count'].mean().sort_values(ascending=False)
    busiest_month = monthly_avg.index[0]
    quietest_month = monthly_avg.index[-1]
    return [f"Most active in month {busiest_month}, least active in month {quietest_month}."]


# --- MAIN SCRIPT LOGIC ---
def main(config: Dict):
    stock_names = [os.path.basename(p) for p in glob.glob(os.path.join(config["discovery_dir"], '*')) if os.path.isdir(p)]
    if not stock_names:
        print(f"No stock subdirectories found in {config['discovery_dir']}. Exiting."); return
    print(f"Found {len(stock_names)} stocks to analyze: {stock_names}")

    for stock in stock_names:
        print(f"\n\n{'='*25} GENERATING MASTER REPORT FOR: {stock.upper()} {'='*25}")
        ts_df = load_data_with_datetime(os.path.join(config["data_dir"], f"{stock}.csv"), config["timestamp_col"])
        if ts_df is None: continue
        artifact_path = os.path.join(config["discovery_dir"], stock, "discovery_artifacts.npz")
        if not os.path.exists(artifact_path): continue
        artifacts = np.load(artifact_path, allow_pickle=True)
        
        stock_output_dir = os.path.join(config["output_dir"], stock)
        os.makedirs(stock_output_dir, exist_ok=True)

        for i, window_size in enumerate(artifacts['window_sizes']):
            k, indices, clusters = artifacts['k_values'][i], artifacts['indices_list'][i], artifacts['clusters_list'][i]
            window_report_dir = os.path.join(stock_output_dir, f"w_{window_size}")
            os.makedirs(window_report_dir, exist_ok=True)
            report_path = os.path.join(window_report_dir, f'master_report_w{window_size}.txt')
            patterns_df = pd.DataFrame({'cluster_id': clusters, 'timestamp': ts_df.index[indices]}).set_index('timestamp')
            
            # --- MODIFICATION 2: Prepare for evenness analysis ---
            cluster_evenness_metrics = {}
            
            with open(report_path, 'w') as f:
                f.write(f"MASTER ANALYSIS REPORT FOR STOCK: {stock.upper()} | WINDOW SIZE: {window_size}\n")
                f.write("="*80 + "\n")

                for cluster_id in range(k):
                    f.write(f"\n--- Cluster {cluster_id} Profile ---\n")
                    cluster_patterns_df = patterns_df[patterns_df['cluster_id'] == cluster_id]
                    if cluster_patterns_df.empty:
                        f.write("  This cluster has no occurrences.\n\n"); continue
                    
                    cluster_plots_dir = os.path.join(window_report_dir, f"cluster_{cluster_id}_plots")
                    os.makedirs(cluster_plots_dir, exist_ok=True)

                    daily_counts = cluster_patterns_df.groupby(pd.Grouper(freq='D')).size()
                    weekly_counts = cluster_patterns_df.groupby(pd.Grouper(freq='W-MON')).size()

                    # --- MODIFICATION 2: Calculate and store evenness metric for each cluster ---
                    if not weekly_counts.empty:
                        yearly_cvs = []
                        for year in weekly_counts.index.year.unique():
                            counts_this_year = weekly_counts[weekly_counts.index.year == year]
                            # CV is std/mean. Requires >1 data point and non-zero mean.
                            if len(counts_this_year) > 1 and counts_this_year.mean() > 0:
                                cv = counts_this_year.std() / counts_this_year.mean()
                                yearly_cvs.append(cv)
                        
                        if yearly_cvs:
                            average_cv = np.mean(yearly_cvs)
                            total_occurrences = weekly_counts.sum()
                            cluster_evenness_metrics[cluster_id] = {'avg_cv': average_cv, 'total_count': total_occurrences}
                    # --- END ---
                    
                    # 1. Generate Daily Occurrence Line Chart
                    line_plot_path = os.path.join(cluster_plots_dir, '1_daily_occurrence_trend.png')
                    plot_time_series(daily_counts, f'Daily Occurrences for C{cluster_id} ({stock})', 'Count', 'line', line_plot_path)

                    # 2. Generate Yearly Breakdown Bar Charts
                    yearly_dir = os.path.join(cluster_plots_dir, '2_yearly_weekly_breakdown')
                    os.makedirs(yearly_dir, exist_ok=True)
                    for year in weekly_counts.index.year.unique():
                        plot_time_series(weekly_counts[weekly_counts.index.year == year], f'Weekly Occurrences in {year} for C{cluster_id}', 'Count', 'bar', os.path.join(yearly_dir, f'weekly_freq_{year}.png'))
                    
                    # 3. Generate Seasonality Box Plot
                    boxplot_path = os.path.join(cluster_plots_dir, '3_monthly_seasonality.png')
                    seasonality_conclusions = analyze_seasonality(weekly_counts, boxplot_path, f'Monthly Seasonality of C{cluster_id}')
                    
                    # Write the final, simplified text report
                    f.write("  1. Occurrence Trend: See plots in folder -> " + f"/{os.path.basename(cluster_plots_dir)}/\n")
                    f.write("     - This includes the full daily trend line chart and per-year weekly bar charts.\n\n")
                    f.write("  2. Seasonality (Monthly): " + ', '.join(seasonality_conclusions) + "\n")
                    f.write(f"     - See boxplot: /{os.path.basename(cluster_plots_dir)}/3_monthly_seasonality.png\n")
                
                # --- MODIFICATION 2: Add even distribution summary to the report ---
                f.write("\n\n" + "="*80 + "\n")
                f.write("EVEN DISTRIBUTION ANALYSIS (ACROSS YEARS)\n")
                f.write("="*80 + "\n")
                
                # Filter out clusters with too few occurrences to be meaningful
                min_occurrences_for_evenness = max(10, window_size)
                filtered_metrics = {cid: data for cid, data in cluster_evenness_metrics.items() 
                                    if data['total_count'] >= min_occurrences_for_evenness}

                if filtered_metrics:
                    # Sort clusters by their average CV (lower is more even/stable)
                    sorted_clusters = sorted(filtered_metrics.items(), key=lambda item: item[1]['avg_cv'])
                    
                    f.write(f"This analysis identifies clusters that appear most consistently over time, without major spikes.\n")
                    f.write(f"(Based on the average Coefficient of Variation (CV) of weekly counts per year. Lower CV is more even.)\n")
                    f.write(f"(Only clusters with at least {min_occurrences_for_evenness} occurrences are considered.)\n\n")
                    f.write("Top Most Evenly Distributed Clusters:\n")
                    
                    # List top 3 (or fewer if not available)
                    for rank, (cid, data) in enumerate(sorted_clusters[:3]):
                        f.write(f"  {rank+1}. Cluster {cid} (Avg CV: {data['avg_cv']:.2f}, Total Occurrences: {data['total_count']})\n")
                else:
                    f.write("  No clusters met the minimum occurrence threshold for this analysis.\n")
                # --- END MODIFICATION 2 ---

            print(f"  -> Generated reports and plots for w={window_size}")
            print(f"-> Master text report for {stock} w={window_size} saved.")

    print(f"\n{'='*30} ALL MASTER REPORTS COMPLETE {'='*30}")

if __name__ == "__main__":
    main(CONFIG)