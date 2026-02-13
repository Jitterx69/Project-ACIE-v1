"""
Convolutional Neural Network (CNN) for Image Classification.
Part of the Secure Image Pipeline.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

class SecureImageCNN(nn.Module):
    """
    Simple CNN for classifying astronomical images.
    Input: [Batch, 3, 64, 64] images
    Output: Class probabilities
    """
    
    def __init__(self, num_classes: int = 2):
        super().__init__()
        
        # Conv Block 1
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(32)
        self.pool = nn.MaxPool2d(2, 2)
        
        # Conv Block 2
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(64)
        
        # Conv Block 3
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm2d(128)
        
        # Fully Connected
        # Input image 64x64 -> pool -> 32x32 -> pool -> 16x16 -> pool -> 8x8
        # Final feature map: 128 * 8 * 8
        self.fc1 = nn.Linear(128 * 8 * 8, 512)
        self.dropout = nn.Dropout(0.5)
        self.fc2 = nn.Linear(512, num_classes)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Block 1
        x = self.pool(F.relu(self.bn1(self.conv1(x))))
        
        # Block 2
        x = self.pool(F.relu(self.bn2(self.conv2(x))))
        
        # Block 3
        x = self.pool(F.relu(self.bn3(self.conv3(x))))
        
        # Flatten
        x = x.view(-1, 128 * 8 * 8)
        
        # Classifier
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        
        return F.softmax(x, dim=1)
