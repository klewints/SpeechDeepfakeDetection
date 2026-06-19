"""
Wav2Vec2 model for speech deepfake detection with partial unfreezing.

Features:
- Pretrained facebook/wav2vec2-base model
- Partial unfreezing of transformer layers
- Discriminative learning rates
- Dropout and regularization
"""

import torch
import torch.nn as nn
from transformers import Wav2Vec2Model, Wav2Vec2FeatureExtractor


class Wav2Vec2DeepfakeModel(nn.Module):
    """
    Wav2Vec2 model adapted for deepfake detection.
    
    Uses pretrained Wav2Vec2 with:
    - Partial unfreezing of last transformer layers
    - Custom classification head
    - Dropout regularization
    """
    
    def __init__(self, unfreeze_layers=4, dropout_rate=0.3, num_classes=2):
        """
        Initialize Wav2Vec2 model.
        
        Args:
            unfreeze_layers: Number of transformer layers to unfreeze
            dropout_rate: Dropout probability
            num_classes: Number of output classes (2 for binary)
        """
        super().__init__()
        
        # ===========================
        # LOAD PRETRAINED WAV2VEC2
        # ===========================
        self.wav2vec2 = Wav2Vec2Model.from_pretrained(
            "facebook/wav2vec2-base"
        )
        
        # ===========================
        # FREEZE MOST PARAMETERS
        # ===========================
        for param in self.wav2vec2.parameters():
            param.requires_grad = False
        
        # ===========================
        # UNFREEZE LAST TRANSFORMER LAYERS
        # ===========================
        # The transformer has 12 layers total (for base model)
        # We unfreeze the last `unfreeze_layers` layers
        transformer_layers = self.wav2vec2.encoder.layers
        
        # Calculate which layers to unfreeze
        total_layers = len(transformer_layers)
        freeze_until = total_layers - unfreeze_layers
        
        for i, layer in enumerate(transformer_layers):
            if i >= freeze_until:
                for param in layer.parameters():
                    param.requires_grad = True
        
        # Unfreeze feature projection layer
        for param in self.wav2vec2.feature_extractor.parameters():
            param.requires_grad = False  # Keep fixed
        
        # Unfreeze feature projection
        for param in self.wav2vec2.feature_projection.parameters():
            param.requires_grad = True
        
        # ===========================
        # CLASSIFICATION HEAD
        # ===========================
        wav2vec2_output_dim = self.wav2vec2.config.hidden_size
        
        self.classifier = nn.Sequential(
            nn.Dropout(dropout_rate),
            nn.Linear(wav2vec2_output_dim, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(128, num_classes)
        )
        
        self.dropout = nn.Dropout(dropout_rate)
    
    def forward(self, input_values, attention_mask=None):
        """
        Forward pass.
        
        Args:
            input_values: Raw audio waveform (batch, seq_len)
            attention_mask: Attention mask for padding
        
        Returns:
            Logits (batch, num_classes)
        """
        # Extract features with Wav2Vec2
        outputs = self.wav2vec2(
            input_values,
            attention_mask=attention_mask,
            output_hidden_states=False
        )
        
        # Get last hidden states (batch, seq_len, hidden_size)
        hidden_states = outputs.last_hidden_state
        
        # Global average pooling over sequence length
        # (batch, seq_len, hidden_size) -> (batch, hidden_size)
        pooled = torch.mean(hidden_states, dim=1)
        
        # Apply dropout
        pooled = self.dropout(pooled)
        
        # Classification head
        logits = self.classifier(pooled)
        
        return logits
