import torch
import torch.nn as nn
from torchvision import models

class FogEstimator(nn.Module):
    def __init__(self):
        super().__init__()
        try:
            # Try newer PyTorch weights API
            self.backbone = models.resnet18(weights=None)
        except Exception:
            # Fallback to older PyTorch pretrained flag
            self.backbone = models.resnet18(pretrained=False)
            
        num_features = self.backbone.fc.in_features
        # Predict a single scalar in range [0, 1] using Sigmoid
        self.backbone.fc = nn.Sequential(
            nn.Linear(num_features, 128),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, 1),
            nn.Sigmoid()
        )
        
    def forward(self, x):
        return self.backbone(x).squeeze(-1)
