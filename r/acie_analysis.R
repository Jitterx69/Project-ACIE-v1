# ACIE R Statistical Analysis Package
# Advanced Causal Discovery and Evaluation

library(reticulate)
library(ggplot2)
library(tidyverse)
library(pcalg)
library(bnlearn)

#' Load ACIE Python Model
#'
#' @param model_path Path to ACIE checkpoint
#' @return Python model object
#' @export
load_acie_model <- function(model_path) {
  py <- import_main()
  
  # Import ACIE modules
  train_module <- import("acie.training.train")
  
  # Load model
  model <- train_module$ACIELightningModule$load_from_checkpoint(model_path)
  model$eval()
  
  return(model)
}

#' Causal Discovery from Observational Data
#'
#' Uses PC algorithm to discover causal structure
#'
#' @param data Observational data matrix
#' @param alpha Significance level
#' @return Estimated CPDAG
#' @export
discover_causal_structure <- function(data, alpha = 0.05) {
  suffStat <- list(C = cor(data), n = nrow(data))
  
  # PC algorithm
  pc_result <- pc(
    suffStat = suffStat,
    indepTest = gaussCItest,
    alpha = alpha,
    labels = colnames(data),
    verbose = FALSE
  )
  
  return(pc_result)
}

#' Compute Counterfactual Evaluation Metrics
#'
#' @param predictions Predicted counterfactuals
#' @param ground_truth True counterfactuals
#' @return List of evaluation metrics
#' @export
evaluate_counterfactuals <- function(predictions, ground_truth) {
  # Ensure matrices
  predictions <- as.matrix(predictions)
  ground_truth <- as.matrix(ground_truth)
  
  # MSE
  mse <- mean((predictions - ground_truth)^2)
  
  # MAE
  mae <- mean(abs(predictions - ground_truth))
  
  # R-squared per variable
  r2_per_var <- sapply(1:ncol(predictions), function(i) {
    ss_res <- sum((predictions[, i] - ground_truth[, i])^2)
    ss_tot <- sum((ground_truth[, i] - mean(ground_truth[, i]))^2)
    1 - ss_res / ss_tot
  })
  
  # Correlation
  cor_per_var <- sapply(1:ncol(predictions), function(i) {
    cor(predictions[, i], ground_truth[, i])
  })
  
  list(
    mse = mse,
    mae = mae,
    r2_mean = mean(r2_per_var),
    r2_per_var = r2_per_var,
    correlation_mean = mean(cor_per_var),
    correlation_per_var = cor_per_var
  )
}

#' Statistical Hypothesis Testing for Intervention Effects
#'
#' Tests if interventions have significant causal effects
#'
#' @param factual_data Factual observations
#' @param counterfactual_data Counterfactual predictions
#' @param alpha Significance level
#' @return Data frame with test results
#' @export
test_intervention_effects <- function(factual_data, counterfactual_data, alpha = 0.05) {
  n_vars <- ncol(factual_data)
  
  results <- map_dfr(1:n_vars, function(i) {
    # Paired t-test
    test_result <- t.test(
      factual_data[, i],
      counterfactual_data[, i],
      paired = TRUE
    )
    
    tibble(
      variable = i,
      mean_diff = test_result$estimate,
      t_statistic = test_result$statistic,
      p_value = test_result$p.value,
      significant = test_result$p.value < alpha,
      ci_lower = test_result$conf.int[1],
      ci_upper = test_result$conf.int[2]
    )
  })
  
  # Adjust for multiple testing
  results$p_value_adjusted <- p.adjust(results$p_value, method = "BH")
  results$significant_adjusted <- results$p_value_adjusted < alpha
  
  return(results)
}

#' Visualize Counterfactual Predictions
#'
#' @param factual Factual observations
#' @param counterfactual Counterfactual predictions
#' @param sample_indices Indices to plot
#' @return ggplot object
#' @export
plot_counterfactual_comparison <- function(factual, counterfactual, sample_indices = 1:5) {
  # Prepare data
  plot_data <- map_dfr(sample_indices, function(i) {
bind_rows(
      tibble(
        variable = 1:length(factual[i, ]),
        value = factual[i, ],
        type = "Factual",
        sample = i
      ),
      tibble(
        variable = 1:length(counterfactual[i, ]),
        value = counterfactual[i, ],
        type = "Counterfactual",
        sample = i
      )
    )
  })
  
  # Plot
  ggplot(plot_data, aes(x = variable, y = value, color = type, group = type)) +
    geom_line(alpha = 0.7) +
    facet_wrap(~sample, scales = "free_y") +
    labs(
      title = "Factual vs Counterfactual Predictions",
      x = "Variable Index",
      y = "Value",
      color = "Type"
    ) +
    theme_minimal() +
    theme(legend.position = "bottom")
}

#' Causal Effect Estimation
#'
#' Estimate Average Treatment Effect (ATE)
#'
#' @param model ACIE model
#' @param observations Factual observations
#' @param intervention Intervention specification
#' @return ATE estimate with confidence interval
#' @export
estimate_ate <- function(model, observations, intervention) {
  # Get Python functions
  torch <- import("torch")
  
  # Convert to tensor
  obs_tensor <- torch$tensor(observations, dtype = torch$float32)
  
  # Get ACIE engine
  engine <- model$get_acie_engine()
  
  # Compute counterfactuals
  counterfactuals <- engine$intervene(obs_tensor, intervention)
  counterfactuals_np <- counterfactuals$cpu()$numpy()
  
  # ATE = mean(Y_do(X) - Y)
  ate <- mean(counterfactuals_np - observations)
  
  # Bootstrap confidence interval
  n_bootstrap <- 1000
  boot_ates <- replicate(n_bootstrap, {
    idx <- sample(1:nrow(observations), replace = TRUE)
    mean(counterfactuals_np[idx, ] - observations[idx, ])
  })
  
  ci <- quantile(boot_ates, c(0.025, 0.975))
  
  list(
    ate = ate,
    ci_lower = ci[1],
    ci_upper = ci[2],
    bootstrap_ates = boot_ates
  )
}
