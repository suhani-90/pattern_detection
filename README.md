# pattern_detection
discovery and analyzing pattern in olhcv data
# Time Series Pattern Discovery and Analysis Pipeline

This project provides a two-step pipeline to first discover recurring patterns in a time series and then analyze their predictive power using various metrics.

## Project Structure

- `discover_patterns.py`: Script to perform heavy computation (Matrix Profile, Clustering) and save the discovered patterns.
- `analyze_patterns.py`: Script to load saved patterns and run multiple trend analyses.
- `requirements.txt`: Python package dependencies.
- `discovery_output/`: Directory where the consolidated pattern data file and cluster plots are saved.
- `analysis_results/`: Directory where the final analysis plots are saved, organized by analysis type.

## Prerequisites

Install the required Python packages:
```bash
pip install -r requirements.txt
```

## Workflow

The process is split into two distinct steps. You must run Step 1 before you can run Step 2.

### Step 1: Discover and Save Patterns

This step reads a raw time series CSV, finds patterns for different window sizes, clusters them, and saves all results into a single file.

**Usage:**
```bash
python discover_patterns.py --filepath /path/to/your/data.csv
```

### Step 2: Analyze Saved Patterns

This step loads the single `discovery_artifacts.npz` file and performs three different types of post-pattern trend analysis for every set of patterns found within it.

**Usage:**
```bash
python analyze_patterns.py --filepath /path/to/your/data.csv --discovery_dir /path/to/discovery_output
```

This will generate the same detailed results folder as before, correctly organized by analysis type and window size.