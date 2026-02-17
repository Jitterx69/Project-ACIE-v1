#' Causal Discovery from Observational Data
#'
#' Uses the PC algorithm to discover the causal structure (CPDAG) from data.
#'
#' @param data A numeric matrix or data frame of observations.
#' @param alpha Significance level for conditional independence tests (default 0.05).
#' @return An object of class `pcAlgo` representing the estimated CPDAG.
#' @import pcalg
#' @export
discover_causal_structure <- function(data, alpha = 0.05) {
  # Check data
  if (any(is.na(data))) {
    warning("Data contains NAs. This may cause errors in PC algorithm.")
  }
  
  # Correlation matrix
  C <- cor(data, use = "pairwise.complete.obs")
  n <- nrow(data)
  suffStat <- list(C = C, n = n)
  
  # PC algorithm (Gaussian)
  pc_result <- pcalg::pc(
    suffStat = suffStat,
    indepTest = pcalg::gaussCItest,
    alpha = alpha,
    labels = colnames(data),
    verbose = FALSE
  )
  
  return(pc_result)
}
