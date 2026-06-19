import torch
import torch.nn as nn

class CNNModel(nn.Module):

    def __init__(self):

        super().__init__()

        self.features = nn.Sequential(

            # Block 1
            nn.Conv2d(
                1,
                16,
                kernel_size=3,
                padding=1
            ),

            nn.ReLU(),

            nn.MaxPool2d(2),

            # Block 2
            nn.Conv2d(
                16,
                32,
                kernel_size=3,
                padding=1
            ),

            nn.ReLU(),

            nn.MaxPool2d(2)
        )

        # Adaptive pooling fixes shape problems
        self.pool = nn.AdaptiveAvgPool2d((8, 8))

        self.classifier = nn.Sequential(

            nn.Flatten(),

            nn.Linear(
                32 * 8 * 8,
                128
            ),

            nn.ReLU(),

            nn.Linear(
                128,
                2
            )
        )

    def forward(self, x):

        x = self.features(x)

        x = self.pool(x)

        x = self.classifier(x)

        return x