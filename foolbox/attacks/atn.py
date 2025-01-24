import torch.nn as nn


class SimpleATN(nn.Module):
    def __init__(self, input_size: int):
        super(SimpleATN, self).__init__()
        # A simple fully connected layer
        self.fc = nn.Linear(input_size, input_size)

    def forward(self, x):
        return self.fc(x)
