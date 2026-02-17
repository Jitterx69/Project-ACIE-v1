#' Fit Hierarchical Model
#'
#' Fits a Linear Mixed-Effects Model to analyze galaxy properties nested within clusters.
#'
#' @param data Data frame containing observations.
#' @param formula String or formula object (e.g., "mass ~ metallicity + (1|cluster_id)").
#' @param family Distribution family (default "gaussian").
#' @return An object of class `lmerMod`.
#' @import lme4
#' @export
fit_hierarchical_model <- function(data, formula, family = "gaussian") {
  
  if (family == "gaussian") {
    model <- lme4::lmer(as.formula(formula), data = data)
  } else {
    model <- lme4::glmer(as.formula(formula), data = data, family = family)
  }
  
  return(model)
}

#' Summarize Random Effects
#'
#' Extracts and summarizes the random effects (e.g., cluster-specific deviations).
#'
#' @param model Fitted mixed-effects model.
#' @return Data frame of random effects.
#' @import lme4
#' @export
get_random_effects <- function(model) {
  re <- lme4::ranef(model)
  as.data.frame(re)
}
