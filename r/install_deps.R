# Install core dependencies for ACIEr
pkgs <- c(
  "reticulate", 
  "ggplot2", 
  "dplyr", 
  "tidyr", 
  "purrr", 
  "pcalg", 
  "bnlearn", 
  "Rtsne", 
  "plotly", 
  "shiny", 
  "shinydashboard", 
  "lme4", 
  "spatstat", 
  "testthat", 
  "rmarkdown",
  "knitr"
)

# Filter missing
missing_pkgs <- pkgs[!(pkgs %in% installed.packages()[,"Package"])]

# Handle library paths - Use local project library
lib_loc <- file.path(getwd(), "r-libs")
if (!dir.exists(lib_loc)) {
  dir.create(lib_loc, recursive = TRUE, showWarnings = FALSE)
}
.libPaths(c(lib_loc, .libPaths()))
message("Installing to local library: ", lib_loc)

if(length(missing_pkgs)) {
  message("Installing missing packages: ", paste(missing_pkgs, collapse=", "))
  install.packages(missing_pkgs, repos="https://cloud.r-project.org", lib=lib_loc)
} else {
  message("All dependencies installed.")
}
