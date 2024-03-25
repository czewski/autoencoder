import torch.nn as nn
import torch.nn.functional as F
import torch
import torch.nn.init as weight_init

class AutoEncoder(nn.Module):
    def __init__(self, num_items, embedding_dim=100, hidden_dim=100):
        super(AutoEncoder, self).__init__()
        
        #Method 3
        # Stack encoders and decoders
        encoder_modules, decoder_modules = [], []
        encoder_modules.append(nn.Linear(19,100))
        #encoder_modules.append(nn.Linear(256,100))
        decoder_modules.append(nn.Linear(100,19))
        #decoder_modules.append(nn.Linear(256,19))
        self.encoder = nn.ModuleList(encoder_modules)
        self.decoder = nn.ModuleList(decoder_modules)

        # Initialize weights
        self.encoder.apply(self.weight_init)
        self.decoder.apply(self.weight_init)

        #Method 2
        # Encoder layers
        # self.encoder = nn.Sequential(
        #     nn.Linear(19, 256),
        #     nn.ReLU(),
        #     nn.Linear(256, hidden_dim),
        #     nn.ReLU()
        # )

        # # Decoder layers
        # self.decoder = nn.Sequential(
        #     nn.Linear(hidden_dim, 256),
        #     nn.ReLU(),
        #     nn.Linear(256, 19),
        #     nn.Softmax() 
        # )

        #Method 1
        # self.encoder = nn.Linear(19, hidden_dim)  # Adjusted encoder input size
        # self.decoder = nn.Linear(hidden_dim, 19)  # Adjusted decoder output size

        #Initialize weights to ONE
        # for layer in self.encoder:
        #     if isinstance(layer, nn.Linear):
        #         weight_init.constant_(layer.weight, 1)
        #         weight_init.constant_(layer.bias, 0)  
        
        # #Initialize weights to ONE
        # for layer in self.decoder:
        #     if isinstance(layer, nn.Linear):
        #         weight_init.constant_(layer.weight, 1)
        #         weight_init.constant_(layer.bias, 0)  
    
    def weight_init(self, m):
        if isinstance(m, nn.Linear):
            nn.init.kaiming_normal_(m.weight)
            m.bias.data.zero_()



    def forward(self, x):
        #Method 3
        #x = F.normalize(x)
        #x = self.input_dropout(x)
        
        for i, layer in enumerate(self.encoder):
            x = layer(x)
            if i != len(self.encoder) - 1:
                x = F.relu(x)
                #x = torch.tanh(x)

        for i, layer in enumerate(self.decoder):
            x = layer(x)
            if i != len(self.decoder) - 1:
                x = F.relu(x)
                #x = torch.tanh(x)
        
        #Method 2
        # x = self.encoder(x)
        # x = self.decoder(x)

        #Method 1
        # encoded = F.relu(self.encoder(input))  # Pass through encoder with ReLU activation
        # #decoded = torch.sigmoid(self.decoder(encoded))  # Pass through decoder with sigmoid activation
        # decoded = torch.softmax(self.decoder(encoded), dim=1)  # Pass through decoder with sigmoid activation
        return x