import torch
import torch.nn as nn

class LSTMAutoencoder(nn.Module):
    def __init__(self, input_dim, hidden_dim, sequence_length):
        super(LSTMAutoencoder, self).__init__()
        self.sequence_length = sequence_length
        self.encoder = nn.LSTM(input_dim, hidden_dim, batch_first=True)
        self.decoder = nn.LSTM(hidden_dim, input_dim, batch_first=True)

    def forward(self, x):
        _, (hidden, _) = self.encoder(x)
        decoded, _ = self.decoder(hidden.repeat(self.sequence_length, 1, 1).transpose(0, 1))
        return decoded