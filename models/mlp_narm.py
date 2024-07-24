import torch.nn as nn
import torch.nn.functional as F
import torch
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

#import torch.nn.init as weight_init
#import torchvision

class MLP(nn.Module):
    def __init__(self, n_items, hidden_size, embedding_dim, batch_size, n_layers = 1):
        super(MLP, self).__init__()

        self.n_items = n_items
        self.hidden_size = hidden_size
        self.batch_size = batch_size
        self.n_layers = n_layers
        self.embedding_dim = embedding_dim

        self.emb = nn.Embedding(self.n_items, self.embedding_dim, padding_idx = 0)
        self.emb_dropout = nn.Dropout(0.25)

        
        self.fc1 = nn.Linear(embedding_dim*19, hidden_size)
        self.fc2 = nn.Linear(hidden_size, n_items)


#self.rnn = nn.LSTM(input_size=embedding_dim, hidden_size=hidden_size, batch_first=True)
        # self.fc1 = nn.Linear(embedding_matrix.size(1) * input_dim, hidden_dim)
        # self.fc2 = nn.Linear(hidden_dim, 100)

        #self.encoder = torchvision.ops.MLP(in_channels=input_dim, hidden_channels=[hidden_dim, 1], activation_layer=nn.ReLU)
        #self.decoder = torchvision.ops.MLP(in_channels=hidden_dim, hidden_channels=[input_dim], activation_layer=nn.ReLU)

        #self.initialize_weights()

    def init_hidden(self, batch_size):
        return torch.zeros((self.n_layers, batch_size, self.hidden_size), requires_grad=True).to(self.device)

    def forward(self, seq, lengths):
        embs = self.emb_dropout(self.emb(seq))
        embs = embs.permute(1, 0, 2)  # Change to (batch_size, sequence_length, embedding_dim)
        embs = embs.contiguous().view(embs.size(0), -1)  # Flatten to (batch_size, sequence_length * embedding_dim)
    
        y = torch.relu(self.fc1(embs))
        output = self.fc2(y)

        return output
