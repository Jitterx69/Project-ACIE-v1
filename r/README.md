# ACIEr: Advanced Causal Inference Engine for R

An R interface to the ACIE system, providing advanced statistical analysis, causal discovery, and visualization tools.

## Installation

### Prerequisites
- Python 3.9+ with `acie` installed
- R 4.2+

### Install from Source
```r
# Install devtools if needed
install.packages("devtools")

# Install ACIEr
devtools::install("r/")
```

## Features

### 1. Causal Discovery
Infer causal structure from observational data using the PC algorithm.
```r
library(ACIEr)

data <- read.csv("observations.csv")
cpdag <- discover_causal_structure(data, alpha = 0.01)
plot(cpdag)
```

### 2. Latent Space Visualization
Project high-dimensional physical states into 2D.
```r
model <- load_acie_model("cie_final.ckpt")
plot_latent_space(model, data, method = "tsne")
```

### 3. Sensitivity Analysis
Test robustness of counterfactuals.
```r
results <- sensitivity_analysis(
  model, 
  observation, 
  intervention = list(mass = 2.0),
  feature_idx = 1,
  epsilon = 0.05
)
```

### 4. Hierarchical Modeling
Fit mixed-effects models for nested data (e.g., galaxies in clusters).
```r
model <- fit_hierarchical_model(data, "mass ~ metallicity + (1|cluster_id)")
summary(model)
```

### 5. Spatial Analysis
Analyze galaxy clustering via 2-Point Correlation Function.
```r
pcf <- calculate_spatial_correlation(data$ra, data$dec)
plot(pcf$r, pcf$xi, type="l")
```

### 6. Automated Reporting
Generate comprehensive HTML analysis reports.
```r
generate_report("report.html", "model.ckpt", "data.csv")
```

### 7. Shiny Dashboard
Interactive dashboard for model monitoring.
```r
shiny::runApp(system.file("shiny", package = "ACIEr"))
```
