
import os
import torch
from torch import nn
import torch.nn.functional as F


class DiscriminatorMLP(nn.Module):
    """Discriminator model for source domain."""

    def __init__(self, input_dims, hidden_dims, output_dims):
        """Init discriminator."""
        super(DiscriminatorMLP, self).__init__()

        self.restored = False

        self.layer = nn.Sequential(
            nn.Linear(input_dims, hidden_dims),
            nn.ReLU(),
            nn.Linear(hidden_dims, hidden_dims),
            nn.ReLU(),
            nn.Linear(hidden_dims, output_dims),
            nn.LogSoftmax()
        )

    def forward(self, input):
        """Forward the discriminator."""
        out = self.layer(input)
        return out
    
"""LeNet model for ADDA."""

class LeNetEncoderMLP(nn.Module):
    """LeNet encoder model for ADDA."""

    def __init__(self, input_dims, hidden_dims, output_dims):
        """Init LeNet encoder."""
        super(LeNetEncoderMLP, self).__init__()

        self.restored = False

        self.encoder = nn.Sequential(
            nn.Linear(input_dims, hidden_dims),
            nn.ReLU(),
            nn.Linear(hidden_dims, hidden_dims),
            nn.ReLU(),
            nn.Linear(hidden_dims, output_dims),
            nn.ReLU()
        )

    def forward(self, input):
        """Forward the LeNet."""
        feat = self.encoder(input.view(input.size(0), -1))
        return feat


class LeNetRegressorMLP(nn.Module):
    """LeNet classifier model for ADDA."""

    def __init__(self, input_dims):
        """Init LeNet encoder."""
        super(LeNetRegressorMLP, self).__init__()
        self.fc2 = nn.Linear(input_dims, 1)
        self.restored = False

    def forward(self, feat):
        """Forward the LeNet classifier."""
        out = F.dropout(F.relu(feat), training=self.training)
        out = F.sigmoid(self.fc2(out))
        return out