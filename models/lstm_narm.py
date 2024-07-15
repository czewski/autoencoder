import torch.nn as nn
import torch.nn.functional as F
import torch

class LSTM(nn.Module):
    def __init__(self, n_items, hidden_size, embedding_dim, batch_size, n_layers = 1):
        super(LSTM, self).__init__()

        self.n_items = n_items
        self.hidden_size = hidden_size
        self.batch_size = batch_size
        self.n_layers = n_layers
        self.embedding_dim = embedding_dim

        self.emb = nn.Embedding(self.n_items, self.embedding_dim, padding_idx = 0)
        self.emb_dropout = nn.Dropout(0.25)

        self.fc1 = nn.LSTM(embedding_dim*19, hidden_size, batch_first=True) #
        self.fc2 = nn.Linear(hidden_size, n_items)


    def init_hidden(self, batch_size):
        return torch.zeros((self.n_layers, batch_size, self.hidden_size), requires_grad=True).to(self.device)

    def forward(self, seq, lengths):
        embs = self.emb_dropout(self.emb(seq))
        embs = embs.permute(1, 0, 2)  # Change to (batch_size, sequence_length, embedding_dim)
        embs = embs.contiguous().view(embs.size(0), -1)  # Flatten to (batch_size, sequence_length * embedding_dim)

        x, _ = self.fc1(embs)
        y = torch.relu(x)
        output = self.fc2(y)

        return output
