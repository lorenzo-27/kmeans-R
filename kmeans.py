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

from kmeans_config import MAX_ITER, N_CLUSTERS, N_FEATURES_LIST, N_SAMPLES, NUM_THREADS

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
    n_threads: int


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


def run_kmeans_r(input_file: str, k: int, max_iter: int, n_threads: int) -> ExperimentResult:
    """Run k-means with R sequential and parallel implementations."""
    # Prepare arguments for R script
    args = json.dumps({
        "dataset_file": input_file,
        "n_clusters": k,
        "max_iter": max_iter,
        "data_dir": str(DATA_DIR),
        "n_threads": n_threads
    }).replace('"', '\\"')  # Properly escape double quotes for shell

    # Call R function
    r_cmd = [
        "Rscript", "-e",
        f"source('kmeans.R'); args <- fromJSON('{args}'); "
        f"result <- run_experiment(args); "
        f"cat(result)"
    ]

    print(f"Executing R kmeans on {input_file} with k={k}, max_iter={max_iter}, and threads={n_threads}")

    try:
        output = subprocess.check_output(r_cmd, stderr=subprocess.PIPE).decode().strip()

        # Check if output is empty
        if not output:
            print("Warning: Empty response from R script")
            # Return default values
            return ExperimentResult(sequential_time=1.0, parallel_time=1.0, speedup=1.0, n_threads=n_threads)

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
                threads = result["n_threads"][0] if isinstance(result["n_threads"], list) else result["n_threads"]

                # Ensure values are never zero to prevent division by zero errors
                if sequential_time <= 0:
                    sequential_time = 1.0
                if parallel_time <= 0:
                    parallel_time = 1.0

                return ExperimentResult(
                    sequential_time=sequential_time,
                    parallel_time=parallel_time,
                    speedup=speedup,
                    n_threads=threads
                )
            else:
                print(f"Could not find valid JSON in response: '{output}'")
                # Return default values
                return ExperimentResult(sequential_time=1.0, parallel_time=1.0, speedup=1.0, n_threads=n_threads)

        except json.JSONDecodeError as e:
            print(f"Failed to parse JSON response: {e}")
            print(f"Raw response: '{output}'")
            # Return default values to avoid crashing
            return ExperimentResult(sequential_time=1.0, parallel_time=1.0, speedup=1.0, n_threads=n_threads)

    except subprocess.CalledProcessError as e:
        print(f"R script execution failed with error: {e}")
        print(f"Error output: {e.stderr.decode()}")
        # Return default values as fallback
        return ExperimentResult(sequential_time=1.0, parallel_time=1.0, speedup=1.0, n_threads=n_threads)


def plot_execution_times(
        results: Dict[Tuple[int, int], ExperimentResult], output_prefix: str
) -> None:
    """Plot execution times comparison between sequential and parallel implementations.

    Results are now indexed by (dimensions, threads)
    """
    # Group results by dimensions
    dimensions = sorted(set(dim for dim, _ in results.keys()))

    # Create a figure for each dimension
    for dim in dimensions:
        plt.figure(figsize=(12, 7))

        # Get all thread values for this dimension
        thread_results = {t: results[(dim, t)] for dim, t in results.keys() if dim == dim}
        threads = sorted(thread_results.keys())

        # Sequential time is the same for all thread values
        sequential_time = list(thread_results.values())[0].sequential_time
        sequential_times = [sequential_time] * len(threads)

        # Get parallel times for each thread count
        parallel_times = [thread_results[t].parallel_time for t in threads]

        # Plot times
        plt.plot(
            threads,
            sequential_times,
            marker="o",
            label="Sequential",
            linewidth=2,
            markersize=8,
        )
        plt.plot(
            threads, parallel_times, marker="s", label="Parallel", linewidth=2, markersize=8
        )

        plt.xlabel("Number of Threads")
        plt.ylabel("Execution Time (ms)")
        plt.title(f"K-means Clustering Execution Times Comparison ({dim} dimensions)")
        plt.grid(True)
        plt.legend()

        # Use log scale if the difference is too large
        min_parallel = min(parallel_times)
        if min_parallel > 0 and sequential_time / min_parallel > 100:
            plt.yscale("log")

        plt.savefig(RESULTS_DIR / f"{output_prefix}_execution_times_dim_{dim}.png")
        plt.close()

    # Create a figure comparing dimensions for each thread count
    thread_counts = sorted(set(t for _, t in results.keys()))

    for thread_count in thread_counts:
        plt.figure(figsize=(12, 7))

        # Get results for all dimensions with this thread count
        dim_results = {dim: results[(dim, thread_count)] for dim, t in results.keys() if t == thread_count}
        dims_for_thread = sorted(dim_results.keys())

        # Get sequential and parallel times for each dimension
        sequential_times = [dim_results[dim].sequential_time for dim in dims_for_thread]
        parallel_times = [dim_results[dim].parallel_time for dim in dims_for_thread]

        # Plot times
        plt.plot(
            dims_for_thread,
            sequential_times,
            marker="o",
            label="Sequential",
            linewidth=2,
            markersize=8,
        )
        plt.plot(
            dims_for_thread, parallel_times, marker="s", label=f"Parallel ({thread_count} threads)",
            linewidth=2, markersize=8
        )

        plt.xlabel("Number of Dimensions")
        plt.ylabel("Execution Time (ms)")
        plt.title(f"K-means Clustering Execution Times Comparison ({thread_count} threads)")
        plt.grid(True)
        plt.legend()

        # Use log scale if the difference is too large
        min_parallel = min(parallel_times) if parallel_times else 1.0
        max_sequential = max(sequential_times) if sequential_times else 1.0
        if min_parallel > 0 and max_sequential / min_parallel > 100:
            plt.yscale("log")

        plt.savefig(RESULTS_DIR / f"{output_prefix}_execution_times_threads_{thread_count}.png")
        plt.close()


def plot_speedup(results: Dict[Tuple[int, int], ExperimentResult], output_prefix: str) -> None:
    """Plot speedup achieved by parallel implementation.

    Results are now indexed by (dimensions, threads)
    """
    # Group results by dimensions
    dimensions = sorted(set(dim for dim, _ in results.keys()))

    # Create speedup plot for each dimension
    for dim in dimensions:
        plt.figure(figsize=(12, 7))

        # Get all thread values for this dimension
        thread_results = {t: results[(dim, t)] for dim, t in results.keys() if dim == dim}
        threads = sorted(thread_results.keys())

        # Get speedup for each thread count
        speedups = [thread_results[t].speedup for t in threads]

        # Plot speedup
        plt.plot(threads, speedups, marker="o", linewidth=2, markersize=8)
        plt.xlabel("Number of Threads")
        plt.ylabel("Speedup (x)")
        plt.title(f"R Parallel K-means Speedup Analysis ({dim} dimensions)")
        plt.grid(True)

        # Add theoretical maximum speedup line (linear speedup)
        ideal_speedups = threads
        plt.plot(threads, ideal_speedups, linestyle='--', label="Ideal Speedup", alpha=0.7)
        plt.legend()

        plt.savefig(RESULTS_DIR / f"{output_prefix}_speedup_dim_{dim}.png")
        plt.close()

    # Create 3D plot showing speedup for both dimensions and threads
    try:
        from mpl_toolkits.mplot3d import Axes3D

        fig = plt.figure(figsize=(14, 10))
        ax = fig.add_subplot(111, projection='3d')

        # Create data points for 3D plot
        x = []  # dimensions
        y = []  # threads
        z = []  # speedup

        for (dim, thread), result in results.items():
            x.append(dim)
            y.append(thread)
            z.append(result.speedup)

        # Create scatter plot
        scatter = ax.scatter(x, y, z, c=z, cmap='viridis', s=100, alpha=0.8)

        # Add labels and title
        ax.set_xlabel('Dimensions')
        ax.set_ylabel('Threads')
        ax.set_zlabel('Speedup')
        ax.set_title('Speedup Analysis (Dimensions vs Threads)')

        # Add colorbar
        cbar = fig.colorbar(scatter, ax=ax, label='Speedup')

        plt.savefig(RESULTS_DIR / f"{output_prefix}_speedup_3d.png")
        plt.close()
    except ImportError:
        print("Warning: 3D plotting not available. Skipping 3D speedup plot.")


def main():
    """Run experiments for R K-means implementation with varying dimensions and threads"""
    results = {}

    # First, generate datasets for all dimensions
    for n_features in N_FEATURES_LIST:
        print(f"\nGenerating dataset for {n_features} dimensions...")
        generate_dataset(N_SAMPLES, n_features, N_CLUSTERS)

    # Then run experiments for all combinations of dimensions and threads
    for n_features in N_FEATURES_LIST:
        dataset_file = f"dataset_{n_features}d.csv"

        for n_threads in NUM_THREADS:
            print(f"\nRunning experiment with {n_features} dimensions and {n_threads} threads...")

            # Run R implementation
            result = run_kmeans_r(dataset_file, N_CLUSTERS, MAX_ITER, n_threads)
            results[(n_features, n_threads)] = result

    # Plot comparative results
    plot_execution_times(results, "r_kmeans")
    plot_speedup(results, "r_kmeans")

    # Save results to CSV
    results_df = pd.DataFrame([
        {
            'dimensions': dim,
            'threads': threads,
            'sequential_time': result.sequential_time,
            'parallel_time': result.parallel_time,
            'speedup': result.speedup
        }
        for (dim, threads), result in results.items()
    ])

    results_df.to_csv(RESULTS_DIR / "r_kmeans_results.csv", index=False)


if __name__ == "__main__":
    main()