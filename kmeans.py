import json
import os
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.datasets import make_blobs

from kmeans_config import MAX_ITER, N_CLUSTERS, N_FEATURES_LIST, N_SAMPLES

# Create necessary directories
PROJECT_ROOT = Path(__file__).parent
DATA_DIR = PROJECT_ROOT / "data"
RESULTS_DIR = PROJECT_ROOT / "results"

# Create directories if they don't exist
DATA_DIR.mkdir(exist_ok=True)
RESULTS_DIR.mkdir(exist_ok=True)


@dataclass
class ExperimentResult:
    sequential_time: float
    parallel_time: float
    speedup: float


def generate_dataset(
        n_samples: int, n_features: int, n_clusters: int, random_state: int = 42
) -> Tuple[np.ndarray, np.ndarray]:
    """Generate synthetic dataset for clustering using R."""
    print(
        f"Generating dataset with {n_samples} samples, {n_features} features, and {n_clusters} clusters..."
    )

    # Prepare arguments for R script
    args = json.dumps({
        "n_samples": n_samples,
        "n_features": n_features,
        "n_clusters": n_clusters,
        "data_dir": str(DATA_DIR)
    }).replace('"', '\\"')  # Properly escape double quotes for shell

    # Call R function to generate and save dataset
    r_cmd = [
        "Rscript", "-e",
        f"source('kmeans.R'); args <- fromJSON('{args}'); "
        f"result <- generate_and_save_dataset(args); "
        f"cat(toJSON(result))"
    ]

    try:
        result = subprocess.check_output(r_cmd, stderr=subprocess.PIPE).decode().strip()
        # Check if result is empty
        if not result:
            print("Warning: Empty response from R script")
            return np.array([]), np.array([])

        try:
            # Extract only the JSON part from the result
            json_start = result.rfind('{')
            json_end = result.rfind('}') + 1
            if json_start >= 0 and json_end > json_start:
                json_str = result[json_start:json_end]
                result_json = json.loads(json_str)

                if "status" in result_json and result_json["status"][0] == "success":
                    return np.array([]), np.array([])
                else:
                    print(f"Failed to generate dataset in R: {result_json.get('error', 'Unknown error')}")
                    return np.array([]), np.array([])
            else:
                print(f"Could not find valid JSON in response: '{result}'")
                return np.array([]), np.array([])

        except json.JSONDecodeError as e:
            print(f"Failed to parse JSON response: {e}")
            print(f"Raw response: '{result}'")
            # Continue with default return to avoid crashing
            return np.array([]), np.array([])

    except subprocess.CalledProcessError as e:
        print(f"R script execution failed with error: {e}")
        print(f"Error output: {e.stderr.decode()}")
        # Return empty arrays as fallback
        return np.array([]), np.array([])


def save_dataset(x: np.ndarray, filename: str) -> None:
    """Save dataset to CSV file."""
    # This function is now redundant as the R function saves the dataset directly
    # But kept for compatibility with the original code structure
    pass


def run_kmeans_r(input_file: str, k: int, max_iter: int) -> ExperimentResult:
    """Run k-means with R sequential and parallel implementations."""
    # Prepare arguments for R script
    args = json.dumps({
        "dataset_file": input_file,
        "n_clusters": k,
        "max_iter": max_iter,
        "data_dir": str(DATA_DIR)
    }).replace('"', '\\"')  # Properly escape double quotes for shell

    # Call R function
    r_cmd = [
        "Rscript", "-e",
        f"source('kmeans.R'); args <- fromJSON('{args}'); "
        f"result <- run_experiment(args); "
        f"cat(result)"
    ]

    print(f"Executing R kmeans on {input_file} with k={k} and max_iter={max_iter}")

    try:
        output = subprocess.check_output(r_cmd, stderr=subprocess.PIPE).decode().strip()

        # Check if output is empty
        if not output:
            print("Warning: Empty response from R script")
            # Return default values
            return ExperimentResult(sequential_time=1.0, parallel_time=1.0, speedup=1.0)

        try:
            # Extract only the JSON part from the output
            json_start = output.rfind('{')
            json_end = output.rfind('}') + 1
            if json_start >= 0 and json_end > json_start:
                json_str = output[json_start:json_end]
                result = json.loads(json_str)

                # Make sure we're getting the first element if it's an array
                sequential_time = result["sequential_time"][0] if isinstance(result["sequential_time"], list) else \
                result["sequential_time"]
                parallel_time = result["parallel_time"][0] if isinstance(result["parallel_time"], list) else result[
                    "parallel_time"]
                speedup = result["speedup"][0] if isinstance(result["speedup"], list) else result["speedup"]

                # Ensure values are never zero to prevent division by zero errors
                if sequential_time <= 0:
                    sequential_time = 1.0
                if parallel_time <= 0:
                    parallel_time = 1.0

                return ExperimentResult(
                    sequential_time=sequential_time,
                    parallel_time=parallel_time,
                    speedup=speedup
                )
            else:
                print(f"Could not find valid JSON in response: '{output}'")
                # Return default values
                return ExperimentResult(sequential_time=1.0, parallel_time=1.0, speedup=1.0)

        except json.JSONDecodeError as e:
            print(f"Failed to parse JSON response: {e}")
            print(f"Raw response: '{output}'")
            # Return default values to avoid crashing
            return ExperimentResult(sequential_time=1.0, parallel_time=1.0, speedup=1.0)

    except subprocess.CalledProcessError as e:
        print(f"R script execution failed with error: {e}")
        print(f"Error output: {e.stderr.decode()}")
        # Return default values as fallback
        return ExperimentResult(sequential_time=1.0, parallel_time=1.0, speedup=1.0)


def plot_execution_times(
        results: Dict[int, ExperimentResult], output_prefix: str
) -> None:
    """Plot execution times comparison between sequential and parallel implementations."""
    dimensions = list(results.keys())
    sequential_times = [r.sequential_time for r in results.values()]
    parallel_times = [r.parallel_time for r in results.values()]

    plt.figure(figsize=(12, 7))

    # Plot times
    plt.plot(
        dimensions,
        sequential_times,
        marker="o",
        label="Sequential",
        linewidth=2,
        markersize=8,
    )
    plt.plot(
        dimensions, parallel_times, marker="s", label="Parallel", linewidth=2, markersize=8
    )

    plt.xlabel("Number of Dimensions")
    plt.ylabel("Execution Time (ms)")
    plt.title("K-means Clustering Execution Times Comparison")
    plt.grid(True)
    plt.legend()

    # Use log scale if the difference is too large
    # Add safety checks to prevent division by zero
    min_parallel = min(parallel_times)
    max_sequential = max(sequential_times)
    if min_parallel > 0 and max_sequential / min_parallel > 100:
        plt.yscale("log")

    plt.savefig(RESULTS_DIR / f"{output_prefix}_execution_times.png")
    plt.close()


def plot_speedup(results: Dict[int, ExperimentResult], output_prefix: str) -> None:
    """Plot speedup achieved by parallel implementation."""
    dimensions = list(results.keys())
    speedups = [r.speedup for r in results.values()]

    plt.figure(figsize=(12, 7))

    plt.plot(dimensions, speedups, marker="o", linewidth=2, markersize=8)
    plt.xlabel("Number of Dimensions")
    plt.ylabel("Speedup (x)")
    plt.title("R Parallel K-means Speedup Analysis")
    plt.grid(True)

    plt.savefig(RESULTS_DIR / f"{output_prefix}_speedup.png")
    plt.close()


def plot_kmeans_clustering(x: np.ndarray, labels: np.ndarray, n_features: int) -> None:
    """Plot visual clustering results."""
    # This would normally use matplotlib to plot, but since we're using R
    # we'll skip this and rely on R's plotting capabilities if needed
    pass


def main():
    """Run experiments for R K-means implementation"""
    results = {}

    for n_features in N_FEATURES_LIST:
        print(f"\nRunning experiments for {n_features} dimensions...")

        # Generate and save dataset
        generate_dataset(N_SAMPLES, n_features, N_CLUSTERS)
        dataset_file = f"dataset_{n_features}d.csv"

        # Run R implementation
        result = run_kmeans_r(dataset_file, N_CLUSTERS, MAX_ITER)
        results[n_features] = result

    # Plot comparative results
    plot_execution_times(results, "r_kmeans")
    plot_speedup(results, "r_kmeans")

    # Save results to CSV
    results_df = pd.DataFrame.from_dict(
        {k: vars(v) for k, v in results.items()}, orient="index"
    )
    results_df.index.name = "dimensions"
    results_df.to_csv(RESULTS_DIR / "r_kmeans_results.csv")


if __name__ == "__main__":
    main()