#' Residual Diagnostics
#'
#' Checks VAE reconstruction residuals for normality using Q-Q plots and Shapiro-Wilk test.
#'
#' @param model Loaded ACIE model object.
#' @param data Matrix of observations.
#' @return List containing Shapiro-Wilk test result and a ggplot object.
#' @import ggplot2
#' @import stats
#' @export
check_residuals <- function(model, data) {
  
  torch <- reticulate::import("torch")
  data_tensor <- torch$tensor(as.matrix(data), dtype = torch$float32)
  engine <- model$get_acie_engine()
  
  # Forward pass (get reconstruction)
  # O = f_O(P, U_O). In VAE, Decoder(Encoder(x))
  with(torch$no_grad(), {
    latent_dist <- engine$inference_model$encode(data_tensor)
    z <- latent_dist$mean
    reconstruction <- engine$inference_model$decode(z)$cpu()$numpy()
  })
  
  residuals <- as.vector(as.matrix(data) - reconstruction)
  
  # Subsample if too large for Shapiro test (limit 5000)
  if (length(residuals) > 5000) {
    check_residuals <- sample(residuals, 5000)
  } else {
    check_residuals <- residuals
  }
  
  shapiro_res <- stats::shapiro.test(check_residuals)
  
  # Plot
  p <- ggplot(data.frame(res = residuals), aes(sample = res)) +
    stat_qq() +
    stat_qq_line(color = "red") +
    labs(title = "Normal Q-Q Plot of Reconstruction Residuals",
         subtitle = paste("Shapiro-Wilk p-value:", format.pval(shapiro_res$p.value))) +
    theme_minimal()
  
  return(list(
    shapiro_test = shapiro_res,
    plot = p
  ))
}
