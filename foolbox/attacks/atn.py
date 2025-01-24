import torch.nn as nn

class SimpleAutoencoder(nn.Module):
    def __init__(self, input_size: int, hidden_size: int):
        super(SimpleAutoencoder, self).__init__()
        # Encoder
        self.encoder = nn.Linear(input_size, hidden_size)
        # Decoder
        self.decoder = nn.Linear(hidden_size, input_size)

    def forward(self, x):
        # Encode
        encoded = self.encoder(x)
        # Decode
        decoded = self.decoder(encoded)
        return decoded
