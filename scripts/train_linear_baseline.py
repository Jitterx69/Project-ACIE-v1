#!/usr/bin/env python3
"""
Train Logistic Regression Baseline.

Demonstrates using a simple linear model (Logistic Regression) to predict
a binary property from the observational data.

Task: Predict if a latent variable (P_0) is positive based on observations (O).
This is a "Propensity-like" task even if strictly predictive here.
"""

import sys
import torch
import torch.nn as nn
import torch.optim as optim
from pathlib import Path
from torch.utils.data import DataLoader, TensorDataset
import numpy as np

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from acie.models.propensity import LogisticPropensityModel

def train_logistic_regression():
    print("="*60)
    print("Training Logistic Regression Baseline")
    print("="*60)
    
    # 1. Create Synthetic Data (for demonstration)
    # In real usage, load from CSV
    print("Generating synthetic data...")
    N_SAMPLES = 1000
    OBS_DIM = 20  # Reduced for demo clarity
    
    # True relationship: y usually 1 if sum(x) > 0
    X = torch.randn(N_SAMPLES, OBS_DIM)
    coefficients = torch.randn(OBS_DIM, 1)
    
    # Generate labels with some noise (simulating probabilistic nature)
    logits = X @ coefficients
    probs_true = torch.sigmoid(logits)
    y = torch.bernoulli(probs_true)
    
    # Split
    dataset = TensorDataset(X, y)
    loader = DataLoader(dataset, batch_size=32, shuffle=True)
    
    # 2. Initialize Model
    model = LogisticPropensityModel(input_dim=OBS_DIM)
    optimizer = optim.SGD(model.parameters(), lr=0.1) # SGD is standard for Logistic Regression
    criterion = nn.BCELoss() # Binary Cross Entropy
    
    print(f"Model: Logistic Regression (1 layer, {OBS_DIM} inputs)")
    
    # 3. Train
    model.train()
    for epoch in range(20):
        total_loss = 0
        correct = 0
        total = 0
        
        for bx, by in loader:
            optimizer.zero_grad()
            
            # Forward
            preds = model(bx)
            loss = criterion(preds, by)
            
            # Backward
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            
            # Accuracy
            predictions = (preds >= 0.5).float()
            correct += (predictions == by).sum().item()
            total += by.size(0)
            
        acc = correct / total
        if epoch % 5 == 0:
            print(f"Epoch {epoch}: Loss = {total_loss/len(loader):.4f}, Acc = {acc:.2%}")

    print("="*60)
    print("Final Model Weights (First 5):")
    print(model.linear.weight.data[0][:5])
    print("\nTraining Complete.")
    print("This simple model serves as a baseline for causal inference tasks.")

if __name__ == "__main__":
    train_logistic_regression()
