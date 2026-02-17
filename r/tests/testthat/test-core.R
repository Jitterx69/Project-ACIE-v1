library(testthat)
library(ACIEr)

# Mock Python model for testing
mock_model <- list(
  get_acie_engine = function() {
    list(
      inference_model = list(
        encode = function(x) {
          list(mean = torch$tensor(matrix(rnorm(10*2), ncol=2)))
        },
        decode = function(z) {
          torch$tensor(matrix(rnorm(10*5), ncol=5))
        }
      ),
      intervene = function(obs, intervention) {
        torch$tensor(obs + 0.1) # Mock intervention
      }
    )
  }
)

test_that("load_acie_model handles missing paths gracefully", {
  expect_error(load_acie_model("non_existent_path.ckpt"))
})

test_that("evaluate_counterfactuals computes metrics correctly", {
  pred <- matrix(1:10, ncol=2)
  truth <- matrix(1:10, ncol=2)
  
  res <- evaluate_counterfactuals(pred, truth)
  
  expect_equal(res$mse, 0)
  expect_equal(res$mae, 0)
  expect_equal(res$correlation_mean, 1)
})

test_that("discover_causal_structure validates input", {
  data_na <- matrix(c(1, NA, 3, 4), ncol=2)
  expect_warning(discover_causal_structure(data_na))
})

# Note: We cannot test Python-dependent functions without a live Python environment
# mocking reticulate is complex here.
