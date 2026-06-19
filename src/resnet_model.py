"""
ResNet18 model for speech deepfake detection using Mel spectrograms.

Uses pretrained ResNet18 from torchvision with:
- Mel spectrogram inputs (converted to 3-channel format)
- Dropout and BatchNorm regularization
- 2-class binary classification (real/fake)
"""

import torch
import torch.nn as nn
import torchvision.models as models


class ResNet18MelModel(nn.Module):
    """
    ResNet18 model adapted for Mel spectrogram input.
    
    The model takes single-channel Mel spectrograms and converts them
    to 3-channel format (by repeating the channel) to work with pretrained
    ResNet18. The final classification layer outputs 2 classes (real/fake).
    """
    
    def __init__(self, pretrained=True, dropout_rate=0.5):
        """
        Initialize ResNet18 model.
        
        Args:
            pretrained: Whether to load pretrained ImageNet weights
            dropout_rate: Dropout probability for regularization
        """
        super().__init__()
        
        # Load pretrained ResNet18
        self.resnet = models.resnet18(pretrained=pretrained)
        
        # ===========================
        # MODIFY INPUT LAYER
        # ===========================
        # ResNet expects 3-channel input, but we'll receive 1-channel
        # The conversion happens in forward()
        
        # ===========================
        # ADD DROPOUT TO RESIDUAL BLOCKS
        # ===========================
        self.dropout = nn.Dropout(dropout_rate)
        
        # ===========================
        # MODIFY FINAL CLASSIFICATION LAYER
        # ===========================
        # ResNet18's final layer outputs 1000 classes (ImageNet)
        # We need 2 classes (real/fake)
        num_ftrs = self.resnet.fc.in_features
        
        self.resnet.fc = nn.Sequential(
            nn.Dropout(dropout_rate),
            nn.Linear(num_ftrs, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(512, 2)  # Binary classification
        )
    
    def forward(self, x):
        """
        Forward pass.
        
        Args:
            x: Input tensor of shape (batch, 1, height, width)
               Single-channel Mel spectrogram
        
        Returns:
            Logits of shape (batch, 2) for binary classification
        """
        # Convert 1-channel to 3-channel by repetition
        # Input: (batch, 1, height, width)
        # Output: (batch, 3, height, width)
        batch_size = x.size(0)
        x = x.repeat(1, 3, 1, 1)
        
        # Apply ResNet with dropout
        x = self.resnet(x)
        
        return x
