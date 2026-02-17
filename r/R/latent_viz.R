#' Visualize Latent Space
#'
#' Projects high-dimensional latent vectors into 2D using PCA or t-SNE.
#'
#' @param model Loaded ACIE model object.
#' @param data Matrix of observations to encode.
#' @param labels Optional vector of class labels for coloring points.
#' @param method "pca" or "tsne" (default "pca").
#' @return A ggplot object.
#' @import ggplot2
#' @import Rtsne
#' @export
plot_latent_space <- function(model, data, labels = NULL, method = "pca") {
  
  # Encode data to get latent vectors
  torch <- reticulate::import("torch")
  data_tensor <- torch$tensor(as.matrix(data), dtype = torch$float32)
  
  engine <- model$get_acie_engine()
  
  # Forward pass through encoder
  with(torch$no_grad(), {
    latent_dist <- engine$inference_model$encode(data_tensor)
    z <- latent_dist$mean$cpu()$numpy()
  })
  
  # Dimensionality Reduction
  if (method == "pca") {
    pca_res <- prcomp(z, scale. = TRUE)
    plot_df <- data.frame(
      Dim1 = pca_res$x[, 1],
      Dim2 = pca_res$x[, 2]
    )
    title_text <- "Latent Space (PCA)"
  } else if (method == "tsne") {
    # Perplexity must be < (nrow(z) - 1) / 3
    perp <- min(30, floor((nrow(z) - 1) / 3))
    tsne_res <- Rtsne::Rtsne(z, perplexity = perp, verbose = FALSE)
    plot_df <- data.frame(
      Dim1 = tsne_res$Y[, 1],
      Dim2 = tsne_res$Y[, 2]
    )
    title_text <- "Latent Space (t-SNE)"
  } else {
    stop("Method must be 'pca' or 'tsne'")
  }
  
  # Add labels if provided
  if (!is.null(labels)) {
    plot_df$Label <- as.factor(labels)
    p <- ggplot(plot_df, aes(x = Dim1, y = Dim2, color = Label))
  } else {
    p <- ggplot(plot_df, aes(x = Dim1, y = Dim2))
  }
  
  p + geom_point(alpha = 0.6) +
    labs(title = title_text, x = "Dimension 1", y = "Dimension 2") +
    theme_minimal()
}
