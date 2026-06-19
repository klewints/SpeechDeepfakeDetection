import torch
import torch.nn as nn
from transformers import Wav2Vec2Model


class Wav2Vec2Classifier(nn.Module):
    """
    Wav2Vec2-based binary classifier for deepfake detection.
    
    Architecture:
    - Pretrained Wav2Vec2Model (facebook/wav2vec2-base)
    - Mean pooling over time dimension
    - Classification head: Linear → ReLU → Dropout → Linear(2)
    """

    def __init__(self, model_name="facebook/wav2vec2-base", freeze_encoder=True):
        super().__init__()

        # Load pretrained Wav2Vec2 encoder
        self.wav2vec2 = Wav2Vec2Model.from_pretrained(model_name)

        # Optionally freeze the encoder to save memory/training time
        if freeze_encoder:
            for param in self.wav2vec2.parameters():
                param.requires_grad = False

        # Get feature dimension from the model
        self.feature_dim = self.wav2vec2.config.hidden_size

        # Classification head
        self.classifier = nn.Sequential(
            nn.Linear(self.feature_dim, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, 2)  # Binary classification: [real, fake]
        )

    def forward(self, x):
        """
        Forward pass.
        
        Args:
            x: Tensor of shape (batch_size, seq_len)
               Raw waveform audio signals
        
        Returns:
            logits: Tensor of shape (batch_size, 2)
                    Logits for binary classification
        """
        # Extract features from Wav2Vec2
        # x shape: (batch_size, seq_len)
        outputs = self.wav2vec2(x)
        hidden_states = outputs.last_hidden_state
        # hidden_states shape: (batch_size, time_steps, hidden_size)

        # Mean pooling over time dimension
        pooled = hidden_states.mean(dim=1)
        # pooled shape: (batch_size, hidden_size)

        # Pass through classifier
        logits = self.classifier(pooled)
        # logits shape: (batch_size, 2)

        return logits
