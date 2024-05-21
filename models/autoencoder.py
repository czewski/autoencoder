import torch.nn as nn
import torch.nn.functional as F
import torch
import torch.nn.init as weight_init

class AutoEncoder(nn.Module):
    def __init__(self, input_dim=20, hidden_dim=50):
        super(AutoEncoder, self).__init__()
        
        self.encoder = nn.LSTM(input_size=input_dim, hidden_size=hidden_dim, batch_first=True)
        self.decoder = nn.LSTM(input_size=hidden_dim, hidden_size=hidden_dim, batch_first=True)
        self.output_layer = nn.Linear(hidden_dim, input_dim)
        self.relu = nn.ReLU()

        self.initialize_weights()

    def initialize_weights(self):
        for layer in self.modules():
            if isinstance(layer, nn.Linear):
                nn.init.xavier_uniform_(layer.weight)
                nn.init.constant_(layer.bias, 0.0)

    def forward(self, x):
        encoded, _ = self.encoder(x)
        encoded = self.relu(encoded)

        decoded, _ = self.decoder(encoded)
        decoded = self.output_layer(decoded)
        
        return decoded
