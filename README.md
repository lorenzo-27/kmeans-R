# Why R Might Not Be the Best Choice for Parallel Programming

This repository presents a comprehensive analysis of R's parallel computing capabilities through a K-Means clustering implementation. The project compares sequential versus parallel execution using R's `doParallel` and `foreach` packages, demonstrating the limitations and challenges of parallel programming in R compared to lower-level languages like C++ with OpenMP.

## Prerequisites

The project requires the following R packages and system components:

- **R (version 4.0 or higher)**
- **Required R packages:**
  - `parallel`
  - `doParallel` 
  - `foreach`
  - `ggplot2`
  - `plotly`
  - `MASS`
  - `tidyverse`
  - `jsonlite`
  - `conflicted`
- **Python (version 3.7 or higher)** for orchestration and visualization
- **Required Python packages:**
  - `matplotlib`
  - `numpy`
  - `pandas`
  - `scikit-learn`

## Setup and Usage

1. **Clone the repository:**
   ```bash
   git clone <repository-url>
   cd r-kmeans-analysis
   ```

2. **Install R dependencies:**
   ```r
   install.packages(c("parallel", "doParallel", "foreach", "ggplot2", 
                      "plotly", "MASS", "tidyverse", "jsonlite", "conflicted"))
   ```

3. **Install Python dependencies:**
   ```bash
   pip install matplotlib numpy pandas scikit-learn
   ```

4. **Configure the algorithm parameters:**
   - Open `kmeans_config.py`
   - Adjust the clustering parameters according to your requirements:
     ```python
     N_SAMPLES = 100000
     N_FEATURES_LIST = [2, 3, 4, 8, 16, 32, 64, 128]
     N_CLUSTERS = 32
     MAX_ITER = 30
     NUM_THREADS = [1, 2, 4]
     ```

5. **Run the analysis:**
   ```bash
   python kmeans.py
   ```

<details>
  <summary>System-Specific Considerations</summary>
  
  **Windows Users:**
  - Ensure Rscript is available in your PATH
  - Consider using WSL for better R parallel performance
  
  **macOS Users:**
  - Some parallel operations may be limited due to macOS's fork() restrictions
  - Consider adjusting the number of cores used in the analysis
  
  **Linux Users:**
  - Generally provides the best performance for R parallel operations
  - Ensure sufficient memory is available for large datasets
</details>

> [!NOTE]
> Upon execution, the program automatically creates two directories:
> - `data/`: Contains generated synthetic datasets
> - `results/`: Stores performance plots, speedup analysis, and CSV results

## Documentation
For a comprehensive understanding of the implementation and performance analysis, please refer to our detailed technical report available <a href="https://github.com/lorenzo-27/kmeans-R/blob/master/TR3-kmeans.pdf">here</a>. The report includes:
- Implementation details
- Performance benchmarks
- Experimental results and analysis

## License

This project is licensed under the <a href="https://github.com/lorenzo-27/kmeans-R/blob/master/LICENSE" target="_blank">MIT</a> License.
