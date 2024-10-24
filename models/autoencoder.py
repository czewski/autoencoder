import torch.nn as nn
import torch.nn.functional as F
import torch
import torch.nn.init as weight_init
import torchvision

class AutoEncoder(nn.Module):
    def __init__(self, input_dim=20, hidden_dim=16, output_dim=20):
        super(AutoEncoder, self).__init__()
        
        # self.encoder = nn.LSTM(input_size=input_dim, hidden_size=hidden_dim, batch_first=True)
        # self.decoder = nn.LSTM(input_size=hidden_dim, hidden_size=hidden_dim, batch_first=True)
        # self.output_layer = nn.Linear(hidden_dim, output_dim)
        # self.relu = nn.ReLU()

        self.encoder = torchvision.ops.MLP(in_channels=input_dim, hidden_channels=[hidden_dim], activation_layer=nn.ReLU)
        self.decoder = torchvision.ops.MLP(in_channels=hidden_dim, hidden_channels=[input_dim], activation_layer=nn.ReLU)

        self.initialize_weights()

    def initialize_weights(self):
        for layer in self.modules():
            if isinstance(layer, nn.Linear):
                nn.init.xavier_uniform_(layer.weight)
                nn.init.constant_(layer.bias, 0.0)

    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        
        return decoded
