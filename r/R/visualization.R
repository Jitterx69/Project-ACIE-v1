#' Visualize Counterfactual Predictions
#'
#' Plots factual vs counterfactual trajectories for selected samples.
#'
#' @param factual Matrix of factual observations.
#' @param counterfactual Matrix of counterfactual predictions.
#' @param sample_indices Vector of indices to visualize (default 1:5).
#' @return A ggplot object.
#' @import ggplot2
#' @import dplyr
#' @import tidyr
#' @export
plot_counterfactual_comparison <- function(factual, counterfactual, sample_indices = 1:5) {
  
  plot_data <- purrr::map_dfr(sample_indices, function(i) {
    dplyr::bind_rows(
      tibble::tibble(
        variable = 1:length(factual[i, ]),
        value = factual[i, ],
        type = "Factual",
        sample = as.character(i)
      ),
      tibble::tibble(
        variable = 1:length(counterfactual[i, ]),
        value = counterfactual[i, ],
        type = "Counterfactual",
        sample = as.character(i)
      )
    )
  })
  
  ggplot(plot_data, aes(x = variable, y = value, color = type, group = type)) +
    geom_line(alpha = 0.7, size = 1) +
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
