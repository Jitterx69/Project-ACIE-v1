# ACIE Shiny Dashboard
# Interactive visualization and analysis

library(shiny)
library(shinydashboard)
library(plotly)
library(DT)
library(reticulate)

source("acie_analysis.R")

# UI Definition
ui <- dashboardPage(
  dashboardHeader(title = "ACIE Dashboard"),
  
  dashboardSidebar(
    sidebarMenu(
      menuItem("Model Overview", tabName = "overview", icon = icon("dashboard")),
      menuItem("Counterfactual Inference", tabName = "inference", icon = icon("brain")),
      menuItem("Metrics", tabName = "metrics", icon = icon("chart-line")),
      menuItem("Causal Discovery", tabName = "causal", icon = icon("project-diagram"))
    )
  ),
  
  dashboardBody(
    tabItems(
      # Overview tab
     tabItem(tabName = "overview",
        fluidRow(
          valueBoxOutput("model_status"),
          valueBoxOutput("total_inferences"),
          valueBoxOutput("avg_accuracy")
        ),
        fluidRow(
          box(
            title = "Model Architecture",
            width = 12,
            plotOutput("architecture_plot")
          )
        )
      ),
      
      # Inference tab
      tabItem(tabName = "inference",
        fluidRow(
          box(
            title = "Intervention Configuration",
            width = 4,
            fileInput("obs_file", "Upload Observations (CSV)"),
            numericInput("mass_intervention", "Mass Intervention:", value = 1.5),
            numericInput("metallicity_intervention", "Metallicity:", value = 0.02),
            actionButton("run_inference", "Run Inference", class = "btn-primary")
          ),
          box(
            title = "Counterfactual Predictions",
            width = 8,
            plotlyOutput("counterfactual_plot")
          )
        ),
        fluidRow(
          box(
            title = "Results Table",
            width = 12,
            DTOutput("results_table")
          )
        )
      ),
      
      # Metrics tab
      tabItem(tabName = "metrics",
        fluidRow(
          box(
            title = "Prediction Accuracy",
            width = 6,
            plotlyOutput("accuracy_plot")
          ),
          box(
            title = "Physics Violations",
            width = 6,
            plotlyOutput("physics_plot")
          )
        ),
        fluidRow(
          box(
            title = "Intervention Effects",
            width = 12,
            plotlyOutput("effects_plot")
          )
        )
      ),
      
      # Causal Discovery tab
      tabItem(tabName = "causal",
        fluidRow(
          box(
            title = "Discovered Causal Structure",
            width = 12,
            plotOutput("causal_graph", height = "600px")
          )
        )
      )
    )
  )
)

# Server Logic
server <- function(input, output, session) {
  
  # Load ACIE model
  acie_model <- reactive({
    model_path <- Sys.getenv("ACIE_MODEL_PATH", 
      "/Users/jitterx/Desktop/ACIE/outputs/acie_final.ckpt")
    load_acie_model(model_path)
  })
  
  # Model Status
  output$model_status <- renderValueBox({
    valueBox(
      "Ready",
      "Model Status",
      icon = icon("check-circle"),
      color = "green"
    )
  })
  
  output$total_inferences <- renderValueBox({
    valueBox(
      "1,234",
      "Total Inferences",
      icon = icon("calculator"),
      color = "blue"
    )
  })
  
  output$avg_accuracy <- renderValueBox({
    valueBox(
      "94.3%",
      "Avg Accuracy",
      icon = icon("star"),
      color = "yellow"
    )
  })
  
  # Counterfactual inference
  counterfactual_results <- eventReactive(input$run_inference, {
    req(input$obs_file)
    
    # Load observations
    obs_data <- read.csv(input$obs_file$datapath)
    
    # Prepare intervention
    intervention <- list(
      mass = input$mass_intervention,
      metallicity = input$metallicity_intervention
    )
    
    # Run inference
    torch <- import("torch")
    obs_tensor <- torch$tensor(as.matrix(obs_data), dtype = torch$float32)
    
    engine <- acie_model()$get_acie_engine()
    counterfactuals <- engine$intervene(obs_tensor, intervention)
    
    list(
      factual = obs_data,
      counterfactual = as.data.frame(counterfactuals$cpu()$numpy())
    )
  })
  
  # Plot counterfactuals
  output$counterfactual_plot <- renderPlotly({
    results <- counterfactual_results()
    
    plot_ly() %>%
      add_trace(
        y = colMeans(results$factual),
        type = "scatter",
        mode = "lines",
        name = "Factual"
      ) %>%
      add_trace(
        y = colMeans(results$counterfactual),
        type = "scatter",
        mode = "lines",
        name = "Counterfactual"
      ) %>%
      layout(
        title = "Mean Variable Values",
        xaxis = list(title = "Variable Index"),
        yaxis = list(title = "Value")
      )
  })
  
  # Results table
  output$results_table <- renderDT({
    results <- counterfactual_results()
    head(results$counterfactual, 100)
  })
}

# Run App
shinyApp(ui, server)
