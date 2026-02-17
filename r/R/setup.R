#' Load ACIE Python Model
#'
#' Initializes the Python environment and loads a trained ACIE checkpoint.
#'
#' @param model_path Path to the .ckpt file.
#' @return A Python object representing the loaded LightningModule in eval mode.
#' @import reticulate
#' @export
load_acie_model <- function(model_path) {
  # Ensure Python is initialized
  if (!reticulate::py_available()) {
    # Try to find python in venv if not initialized
    venv_path <- file.path(getwd(), "venv")
    if (dir.exists(venv_path)) {
      reticulate::use_virtualenv(venv_path, required = FALSE)
    }
  }
  
  sys <- reticulate::import("sys")
  # Add project root to python path if needed
  project_root <- normalizePath(file.path(getwd(), ".."), mustWork = FALSE)
  if (!project_root %in% sys$path) {
    sys$path$append(project_root)
  }

  train_module <- reticulate::import("acie.training.train")
  
  if (!file.exists(model_path)) {
    stop(paste("Model checklist not found at:", model_path))
  }
  
  model <- train_module$ACIELightningModule$load_from_checkpoint(model_path)
  model$eval()
  
  return(model)
}
