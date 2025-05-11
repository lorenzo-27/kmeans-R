library(parallel)
library(doParallel)
library(foreach)
library(ggplot2)
library(plotly)
library(MASS)
library(tidyverse)
library(jsonlite)
library(conflicted)

#' Generate synthetic dataset for clustering
#'
#' @param n_samples Number of samples to generate
#' @param n_features Number of features for each sample
#' @param n_clusters Number of clusters
#' @param random_state Random seed for reproducibility
#' @return A list containing the data matrix and cluster assignments
generate_dataset <- function(n_samples, n_features, n_clusters, random_state = 42) {
  cat(sprintf("Generating dataset with %d samples, %d features, and %d clusters...\n", 
              n_samples, n_features, n_clusters))
  
  set.seed(random_state)
  
  # Create cluster centers randomly
  centers <- matrix(rnorm(n_features * n_clusters), nrow = n_clusters)
  
  # Assign samples to clusters
  y <- sample(1:n_clusters, n_samples, replace = TRUE)
  
  # Generate data points around cluster centers
  x <- matrix(0, nrow = n_samples, ncol = n_features)
  for (i in 1:n_samples) {
    cluster_center <- centers[y[i], ]
    x[i, ] <- cluster_center + rnorm(n_features, sd = 1.0)
  }
  
  return(list(x = x, y = y))
}

#' Save dataset to CSV file
#'
#' @param x Data matrix
#' @param filename Output filename
#' @param dir Directory to save the file
save_dataset <- function(x, filename, dir) {
  filepath <- file.path(dir, filename)
  cat(sprintf("Saving dataset to '%s'...\n", filepath))
  write.csv(x, filepath, row.names = FALSE)
}

#' K-means implementation - sequential version
#'
#' @param x Data matrix
#' @param k Number of clusters
#' @param max_iter Maximum number of iterations
#' @return List containing cluster assignments and centroids
kmeans_sequential <- function(x, k, max_iter) {
  n <- nrow(x)
  p <- ncol(x)
  
  # Random initialization of centroids by selecting k random data points
  set.seed(42)
  centroid_indices <- sample(1:n, k)
  centroids <- x[centroid_indices, , drop = FALSE]
  
  # Initialize cluster assignments
  assignments <- rep(0, n)
  
  for (iter in 1:max_iter) {
    # Assignment step
    for (i in 1:n) {
      min_dist <- Inf
      best_cluster <- 0
      
      # Find the closest centroid
      for (j in 1:k) {
        dist <- sum((x[i, ] - centroids[j, ])^2)
        if (dist < min_dist) {
          min_dist <- dist
          best_cluster <- j
        }
      }
      assignments[i] <- best_cluster
    }
    
    # Update step
    new_centroids <- matrix(0, nrow = k, ncol = p)
    counts <- rep(0, k)
    
    for (i in 1:n) {
      cluster <- assignments[i]
      counts[cluster] <- counts[cluster] + 1
      new_centroids[cluster, ] <- new_centroids[cluster, ] + x[i, ]
    }
    
    # Calculate new centroids
    for (j in 1:k) {
      if (counts[j] > 0) {
        centroids[j, ] <- new_centroids[j, ] / counts[j]
      }
    }
  }
  
  return(list(assignments = assignments, centroids = centroids))
}

#' K-means implementation - parallel version using foreach
#'
#' @param x Data matrix
#' @param k Number of clusters
#' @param max_iter Maximum number of iterations
#' @param n_cores Number of cores to use
#' @return List containing cluster assignments and centroids
kmeans_parallel <- function(x, k, max_iter, n_cores = detectCores() - 1) {
  n <- nrow(x)
  p <- ncol(x)
  
  # Register parallel backend
  cl <- makeCluster(n_cores)
  registerDoParallel(cl)
  
  # Random initialization of centroids
  set.seed(42)
  centroid_indices <- sample(1:n, k)
  centroids <- x[centroid_indices, , drop = FALSE]
  
  # Initialize cluster assignments
  assignments <- rep(0, n)
  
  for (iter in 1:max_iter) {
    # Assignment step in parallel
    # Split data into chunks for parallel processing
    chunk_size <- ceiling(n / n_cores)
    
    # Process each chunk in parallel
    results <- foreach(chunk = 1:n_cores, .combine = c) %dopar% {
      start_idx <- (chunk - 1) * chunk_size + 1
      end_idx <- min(chunk * chunk_size, n)
      
      if (start_idx > n) return(numeric(0))
      
      chunk_assignments <- rep(0, end_idx - start_idx + 1)
      
      for (i in start_idx:end_idx) {
        min_dist <- Inf
        best_cluster <- 0
        
        for (j in 1:k) {
          dist <- sum((x[i, ] - centroids[j, ])^2)
          if (dist < min_dist) {
            min_dist <- dist
            best_cluster <- j
          }
        }
        chunk_assignments[i - start_idx + 1] <- best_cluster
      }
      
      return(chunk_assignments)
    }
    
    assignments <- results
    
    # Update step - can also be parallelized for high dimensional data
    new_centroids <- matrix(0, nrow = k, ncol = p)
    counts <- rep(0, k)
    
    for (i in 1:n) {
      cluster <- assignments[i]
      counts[cluster] <- counts[cluster] + 1
      new_centroids[cluster, ] <- new_centroids[cluster, ] + x[i, ]
    }
    
    # Calculate new centroids
    for (j in 1:k) {
      if (counts[j] > 0) {
        centroids[j, ] <- new_centroids[j, ] / counts[j]
      }
    }
  }
  
  # Stop cluster
  stopCluster(cl)
  
  return(list(assignments = assignments, centroids = centroids))
}

#' Run k-means with both sequential and parallel implementations
#'
#' @param input_file Input CSV file with data
#' @param k Number of clusters
#' @param max_iter Maximum number of iterations
#' @param data_dir Directory containing the input file
#' @return List with execution times and speedup information
run_kmeans <- function(input_file, k, max_iter, data_dir) {
  input_path <- file.path(data_dir, input_file)
  cat(sprintf("Loading data from %s\n", input_path))
  
  # Load data
  data <- read.csv(input_path)
  x <- as.matrix(data)
  
  # Run sequential k-means
  cat("Running sequential k-means...\n")
  seq_start <- Sys.time()
  seq_result <- kmeans_sequential(x, k, max_iter)
  seq_end <- Sys.time()
  sequential_time <- as.numeric(difftime(seq_end, seq_start, units = "secs")) * 1000  # Convert to ms
  
  # Run parallel k-means
  cat("Running parallel k-means...\n")
  n_cores <- detectCores() - 1
  cat(sprintf("Using %d cores for parallel execution\n", n_cores))
  
  par_start <- Sys.time()
  par_result <- kmeans_parallel(x, k, max_iter, n_cores)
  par_end <- Sys.time()
  parallel_time <- as.numeric(difftime(par_end, par_start, units = "secs")) * 1000  # Convert to ms
  
  # Calculate speedup
  speedup <- sequential_time / parallel_time
  
  # Print results
  cat(sprintf("Sequential execution time: %.2f ms\n", sequential_time))
  cat(sprintf("Parallel execution time: %.2f ms\n", parallel_time))
  cat(sprintf("Speedup: %.2fx\n", speedup))
  
  # Export the results to a JSON file that Python can read
  result <- list(
    sequential_time = sequential_time,
    parallel_time = parallel_time,
    speedup = speedup
  )
  
  return(result)
}

#' Main function to run a single experiment and return the results
#'
#' @param args List of arguments including dataset_file, n_clusters, max_iter, data_dir
#' @return JSON string with results
run_experiment <- function(args) {
  result <- run_kmeans(args$dataset_file, args$n_clusters, args$max_iter, args$data_dir)
  return(toJSON(result))
}

#' Generate a dataset based on input parameters
#'
#' @param args List of arguments including n_samples, n_features, n_clusters, data_dir
#' @return List with data generation status
generate_and_save_dataset <- function(args) {
  # Generate dataset
  dataset <- generate_dataset(args$n_samples, args$n_features, args$n_clusters)
  
  # Save dataset
  dataset_file <- sprintf("dataset_%dd.csv", args$n_features)
  save_dataset(dataset$x, dataset_file, args$data_dir)
  
  return(list(status = "success", filename = dataset_file))
}
