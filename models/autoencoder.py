import torch.nn as nn
import torch.nn.functional as F

# Definir o modelo de autoencoder com grus
class AutoEncoder(nn.Module):
    def __init__(self, num_items, embedding_dim=100, hidden_dim=200):
        super(AutoEncoder, self).__init__()
        self.embedding = nn.Embedding(num_items + 1, embedding_dim)
        self.encoder = nn.GRU(embedding_dim, hidden_dim, batch_first=True)
        self.decoder = nn.Linear(hidden_dim, num_items)
        
    def forward(self, input):
        embedded = self.embedding(input)
        _, hidden = self.encoder(embedded)
        output = self.decoder(hidden.squeeze(0))
        logits = F.sigmoid(output)
        return logits