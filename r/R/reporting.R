#' Generate Analysis Report
#'
#' Renders a parameterized RMarkdown report summarizing the analysis.
#'
#' @param output_file Path to save the HTML/PDF report.
#' @param model_path Path to the ACIE model checkpoint.
#' @param data_path Path to the observation CSV.
#' @param title Title of the report.
#' @return The output file path.
#' @import rmarkdown
#' @export
generate_report <- function(output_file, model_path, data_path, title = "ACIE Analysis Report") {
  
  template_path <- system.file("rmd", "analysis_report.Rmd", package = "ACIEr")
  
  if (template_path == "") {
    # Fallback for dev mode
    template_path <- file.path(getwd(), "inst", "rmd", "analysis_report.Rmd")
  }
  
  if (!file.exists(template_path)) {
    stop("Report template not found.")
  }
  
  rmarkdown::render(
    input = template_path,
    output_file = output_file,
    params = list(
      model_path = model_path,
      data_path = data_path,
      title = title
    ),
    clean = TRUE
  )
  
  return(output_file)
}
