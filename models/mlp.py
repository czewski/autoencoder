import torch.nn as nn
import torch.nn.functional as F
import torch
import torch.nn.init as weight_init
import torchvision

class MLP(nn.Module):
    def __init__(self, embedding_matrix, input_dim, hidden_dim):
        super(MLP, self).__init__()

        # self.embedding = nn.Embedding(107311, 20)
        # self.embedding.weight = nn.Parameter(embedding_matrix)
        # self.embedding.weight.requires_grad = False  # Freeze the embedding layer

        self.embedding = nn.Embedding.from_pretrained(embedding_matrix, freeze=True)
        
        self.rnn = nn.LSTM(input_size=embedding_matrix.size(1) * input_dim, hidden_size=hidden_dim, batch_first=True)
        self.fc1 = nn.Linear(hidden_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, 1)

        # self.fc1 = nn.Linear(embedding_matrix.size(1) * input_dim, hidden_dim)
        # self.fc2 = nn.Linear(hidden_dim, 1)

        #self.encoder = torchvision.ops.MLP(in_channels=input_dim, hidden_channels=[hidden_dim, 1], activation_layer=nn.ReLU)
        #self.decoder = torchvision.ops.MLP(in_channels=hidden_dim, hidden_channels=[input_dim], activation_layer=nn.ReLU)

        self.initialize_weights()

    def initialize_weights(self):
        for layer in self.modules():
            if isinstance(layer, nn.Linear):
                nn.init.xavier_uniform_(layer.weight)
                nn.init.constant_(layer.bias, 0.0)

    def forward(self, x):
        item_embeddings = self.embedding(x)
        item_embeddings = item_embeddings.view(item_embeddings.size(0), -1)  # Flatten (batch_size, sequence_length * embedding_dim)

        x1 = self.rnn(item_embeddings)

        y = torch.relu(self.fc1(x1))
        output = self.fc2(y)

        return output.squeeze()
