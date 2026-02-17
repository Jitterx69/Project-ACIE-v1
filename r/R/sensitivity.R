#' Sensitivity Analysis
#'
#' Tests the robustness of counterfactual predictions to small perturbations in input.
#'
#' @param model Loaded ACIE model object.
#' @param observation Single observation vector (numeric).
#' @param intervention Intervention list.
#' @param feature_idx Index of feature to perturb.
#' @param epsilon Perturbation magnitude (default 0.01).
#' @param n_steps Number of steps around the observation value (default 10).
#' @return Data frame of perturbation results.
#' @import reticulate
#' @export
sensitivity_analysis <- function(model, observation, intervention, feature_idx, epsilon = 0.01, n_steps = 10) {
  
  torch <- reticulate::import("torch")
  engine <- model$get_acie_engine()
  
  # Generate perturbed inputs
  shifts <- seq(-epsilon, epsilon, length.out = n_steps)
  base_val <- observation[feature_idx]
  
  results <- purrr::map_dfr(shifts, function(delta) {
    perturbed_obs <- observation
    perturbed_obs[feature_idx] <- base_val + delta
    
    # Run inference
    obs_tensor <- torch$tensor(matrix(perturbed_obs, nrow = 1), dtype = torch$float32)
    cf <- engine$intervene(obs_tensor, intervention)
    cf_np <- cf$cpu()$numpy()
    
    # Calculate output mean shift
    output_mean <- mean(cf_np)
    
    tibble::tibble(
      perturbation = delta,
      input_value = base_val + delta,
      output_mean = output_mean
    )
  })
  
  return(results)
}
