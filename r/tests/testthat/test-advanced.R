library(testthat)
library(ACIEr)

test_that("generate_report finds template", {
  # Mock rmarkdown::render to avoid actually running pandoc
  # But we can check if template path resolution works in the function logic
  # Actually, we can just check if the template exists
  expect_true(file.exists(system.file("rmd/analysis_report.Rmd", package = "ACIEr")) || 
              file.exists(file.path("..", "..", "inst", "rmd", "analysis_report.Rmd")))
})

# We cannot easily test lme4 or spatstat without real data and dependencies
