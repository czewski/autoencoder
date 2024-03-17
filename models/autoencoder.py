import torch.nn as nn
import torch.nn.functional as F
import torch

class AutoEncoder(nn.Module):
    def __init__(self, num_items, embedding_dim=100, hidden_dim=200):
        super(AutoEncoder, self).__init__()
        self.encoder = nn.Linear(19, hidden_dim)  # Adjusted encoder input size
        self.decoder = nn.Linear(hidden_dim, 19)  # Adjusted decoder output size
        
    def forward(self, input):
        encoded = F.relu(self.encoder(input))  # Pass through encoder with ReLU activation
        #decoded = torch.sigmoid(self.decoder(encoded))  # Pass through decoder with sigmoid activation
        decoded = torch.sigmoid(self.decoder(encoded))  # Pass through decoder with sigmoid activation
        
        return decoded


# Definir o modelo de autoencoder com grus
# class AutoEncoder(nn.Module):
#     def __init__(self, num_items, embedding_dim=100, hidden_dim=200, batch_size=50):
#         super(AutoEncoder, self).__init__()
#         #self.embedding = nn.Embedding(num_items + 1, embedding_dim)
#         #self.encoder = nn.GRU(embedding_dim, hidden_dim, batch_first=True)
#         self.encoder = nn.Linear(19, 50)
#         self.decoder = nn.Linear(19, batch_size)
#         #self.decoder = nn.Linear(hidden_dim, num_items)
        
#     def forward(self, input):
#         #embedded = self.embedding(input)
#         #_, hidden = self.encoder(embedded)
#         input = input.to(torch.float32).transpose(0, 1)
#         _, hidden = self.encoder(input)
#         output = self.decoder(hidden.squeeze(0))
#         logits = F.sigmoid(output)
#         return logits