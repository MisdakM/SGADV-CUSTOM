import torch

import torch.nn as nn

class SimpleAutoencoder(nn.Module):
    def __init__(self, input_channels: int = 3, height: int = 250, width: int = 250, hidden_size: int = 128):
        super(SimpleAutoencoder, self).__init__()
        self.input_channels = input_channels
        self.height = height
        self.width = width
        self.hidden_size = hidden_size
        self.flattened_size = input_channels * height * width

        # Encoder
        self.encoder = nn.Sequential(
            nn.Linear(self.flattened_size, hidden_size),
            nn.ReLU(True)
        )
        # Decoder
        self.decoder = nn.Sequential(
            nn.Linear(hidden_size, self.flattened_size),
            nn.Sigmoid()  # To ensure the output values are between 0 and 1
        )

    def forward(self, x):
        # Flatten the input
        x = x.reshape(x.size(0), -1)
        # Encode
        encoded = self.encoder(x)
        # Decode
        decoded = self.decoder(encoded)
        # Reshape back to image dimensions
        decoded = decoded.view(x.size(0), self.input_channels, self.height, self.width)
        return decoded
