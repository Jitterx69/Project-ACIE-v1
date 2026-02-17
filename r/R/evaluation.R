#' Compute Counterfactual Evaluation Metrics
#'
#' Calculates metrics to assess the quality of counterfactual predictions against ground truth.
#'
#' @param predictions Numeric matrix of predicted values.
#' @param ground_truth Numeric matrix of true values.
#' @return A list containing MSE, MAE, R-squared, and Correlation metrics.
#' @export
evaluate_counterfactuals <- function(predictions, ground_truth) {
  predictions <- as.matrix(predictions)
  ground_truth <- as.matrix(ground_truth)
  
  mse <- mean((predictions - ground_truth)^2)
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
    correlation_mean = mean(cor_per_var, na.rm = TRUE),
    correlation_per_var = cor_per_var
  )
}

#' Statistical Hypothesis Testing for Intervention Effects
#'
#' Performs paired t-tests to determine if interventions caused significant shifts.
#'
#' @param factual_data Matrix of factual observations.
#' @param counterfactual_data Matrix of counterfactual predictions.
#' @param alpha Significance level (default 0.05).
#' @return A data frame comprising test statistics and p-values for each variable.
#' @import dplyr
#' @importFrom stats t.test p.adjust
#' @export
test_intervention_effects <- function(factual_data, counterfactual_data, alpha = 0.05) {
  n_vars <- ncol(factual_data)
  
  results <- purrr::map_dfr(1:n_vars, function(i) {
    test_result <- t.test(
      factual_data[, i],
      counterfactual_data[, i],
      paired = TRUE
    )
    
    tibble::tibble(
      variable = i,
      mean_diff = test_result$estimate,
      t_statistic = test_result$statistic,
      p_value = test_result$p.value,
      significant = test_result$p.value < alpha,
      ci_lower = test_result$conf.int[1],
      ci_upper = test_result$conf.int[2]
    )
  })
  
  results$p_value_adjusted <- p.adjust(results$p_value, method = "BH")
  results$significant_adjusted <- results$p_value_adjusted < alpha
  
  return(results)
}

#' Causal Effect Estimation (ATE)
#'
#' Estimates the Average Treatment Effect of an intervention.
#'
#' @param model Loaded ACIE model object.
#' @param observations Numeric matrix of observations.
#' @param intervention List specifying the intervention (e.g., list(mass=2.0)).
#' @return List containing ATE and bootstrap confidence intervals.
#' @import reticulate
#' @export
estimate_ate <- function(model, observations, intervention) {
  torch <- reticulate::import("torch")
  obs_tensor <- torch$tensor(observations, dtype = torch$float32)
  
  engine <- model$get_acie_engine()
  counterfactuals <- engine$intervene(obs_tensor, intervention)
  
  # Ensure we copy data from GPU/Tensor to R
  counterfactuals_np <- counterfactuals$cpu()$numpy()
  
  ate <- mean(counterfactuals_np - observations)
  
  # Bootstrap
  n_bootstrap <- 100
  boot_ates <- replicate(n_bootstrap, {
    idx <- sample(1:nrow(observations), replace = TRUE)
    mean(counterfactuals_np[idx, ] - observations[idx, ])
  })
  
  ci <- quantile(boot_ates, c(0.025, 0.975))
  
  list(
    ate = ate,
    ci_lower = ci[1],
    ci_upper = ci[2]
  )
}
