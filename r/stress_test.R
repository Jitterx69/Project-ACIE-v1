#!/usr/bin/env Rscript

# ACIEr Endurance Test
# Stress tests the package with repeated analysis loops.

# Add local lib path
.libPaths(c(file.path(getwd(), "r-libs"), .libPaths()))

library(ACIEr)
library(ggplot2)

LOG_FILE <- "stress_test.log"

log_msg <- function(msg) {
  cat(paste0("[", Sys.time(), "] ", msg, "\n"), file = LOG_FILE, append = TRUE)
  message(msg)
}

# Configuration
N_ITERATIONS <- 100
N_SAMPLES <- 1000

log_msg("Starting ACIEr Endurance Test...")
log_msg(paste("Iterations:", N_ITERATIONS))
log_msg(paste("Samples per iteration:", N_SAMPLES))

# Mock Data Generation (if file not present)
log_msg("Generating test data...")
data <- data.frame(
  mass = rnorm(N_SAMPLES, 10, 2),
  metallicity = runif(N_SAMPLES, 0, 0.05),
  cluster_id = sample(1:10, N_SAMPLES, replace = TRUE),
  ra = runif(N_SAMPLES, 0, 360),
  dec = runif(N_SAMPLES, -90, 90)
)
write.csv(data, "stress_test_data.csv", row.names = FALSE)

# Main Loop
start_time <- Sys.time()

for (i in 1:N_ITERATIONS) {
  tryCatch({
    log_msg(paste("Iteration", i, "started"))
    
    # 1. Causal Discovery
    cpdag <- discover_causal_structure(data[, c("mass", "metallicity")], alpha = 0.05)
    
    # 2. Hierarchical Modeling
    model <- fit_hierarchical_model(data, "mass ~ metallicity + (1|cluster_id)")
    
    # 3. Spatial Analysis
    pcf <- calculate_spatial_correlation(data$ra, data$dec, r_max = 2.0, n_bins = 10)
    
    # Check memory usage
    mem <- gc()
    log_msg(paste("Iteration", i, "completed. Memory usage:", mem[2,2], "MB"))
    
  }, error = function(e) {
    log_msg(paste("ERROR in iteration", i, ":", e$message))
  })
}

end_time <- Sys.time()
duration <- difftime(end_time, start_time, units = "secs")
log_msg(paste("Endurance test completed in", round(duration, 2), "seconds"))
