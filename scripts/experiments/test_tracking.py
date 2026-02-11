"""
Test script for MLflow experiment tracking
"""

from acie.tracking.experiment import ExperimentTracker
import torch
import torch.nn as nn
import random
import time

# Define a dummy model
class SimpleModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc = nn.Linear(10, 1)
        
    def forward(self, x):
        return self.fc(x)

def run_experiment():
    print("🧪 Starting test experiment...")
    
    tracker = ExperimentTracker("ACIE_Test_Experiment")
    
    with tracker.start_run(run_name=f"test_run_{int(time.time())}"):
        # Log params
        print("Logging parameters...")
        tracker.log_params({
            "learning_rate": 0.01,
            "batch_size": 32,
            "model_type": "SimpleModel"
        })
        
        # Log metrics
        print("Logging metrics...")
        for epoch in range(5):
            loss = 1.0 / (epoch + 1) + random.random() * 0.1
            tracker.log_metrics({"loss": loss, "accuracy": 0.2 * epoch}, step=epoch)
            
        # Log model
        print("Logging model...")
        model = SimpleModel()
        tracker.log_model(model, "simple_model")
        
    print("✅ Experiment complete!")

if __name__ == "__main__":
    run_experiment()
