import torch.nn as nn
import torch.nn.functional as F
import torch
import torch.nn.init as weight_init

class VariationalAutoEncoder(nn.Module):
    def __init__(self, input_dim=20, hidden_dim=50, latent_dim=10, output_dim=20):
        super(VariationalAutoEncoder, self).__init__()
        
        # Encoder
        self.encoder_lstm = nn.LSTM(input_size=input_dim, hidden_size=hidden_dim, batch_first=True)
        self.fc_mu = nn.Linear(hidden_dim, latent_dim)
        self.fc_logvar = nn.Linear(hidden_dim, latent_dim)
        
        # Decoder
        self.decoder_lstm = nn.LSTM(input_size=latent_dim, hidden_size=hidden_dim, batch_first=True)
        self.output_layer = nn.Linear(hidden_dim, output_dim)
        
        self.relu = nn.ReLU()
        
        self.initialize_weights()

    def initialize_weights(self):
        for layer in self.modules():
            if isinstance(layer, nn.Linear):
                nn.init.xavier_uniform_(layer.weight)
                nn.init.constant_(layer.bias, 0.0)

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward(self, x):
        # Encoding
        h, _ = self.encoder_lstm(x)
        
        # Debug print to check dimensions
        #print(f"Shape of h after LSTM: {h.shape}")

        if len(h.shape) == 3:
            h = self.relu(h)

            # Get the final time step output
            h_last = h[:, -1, :]
        elif len(h.shape) == 2:
            h_last = h  # if LSTM output is already 2D, use it as is
        else:
            raise ValueError(f"Unexpected tensor shape: {h.shape}")

        # Compute mu and logvar
        mu = self.fc_mu(h_last)
        logvar = self.fc_logvar(h_last)

        # Reparameterization trick
        z = self.reparameterize(mu, logvar)

        # Decoding
        z = z.unsqueeze(1).repeat(1, x.size(1), 1)
        decoded, _ = self.decoder_lstm(z)
        decoded = self.output_layer(decoded)

        return decoded, mu, logvar

    def loss_function(self, recon_x, x, mu, logvar):
        # Reshape x to match the shape of recon_x
        x = x.unsqueeze(1).repeat(1, recon_x.size(1), 1)
        
        # Reconstruction loss
        recon_loss = F.mse_loss(recon_x, x, reduction='sum')
        
        # KL divergence loss
        kld_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
        
        return recon_loss + kld_loss