import torch.nn as nn
import torch.nn.functional as F
import torch
import torch.nn.init as weight_init
import torchvision

class MLP(nn.Module):
    def __init__(self, embedding_matrix, input_dim=5, hidden_dim=20):
        super(MLP, self).__init__()

        self.embedding = nn.Embedding(107312, 20)
        self.embedding.weight = nn.Parameter(embedding_matrix)
        self.embedding.weight.requires_grad = False  # Freeze the embedding layer

        self.fc1 = nn.Linear(20, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, 1)  # Output layer with 1 output unit

        #self.encoder = torchvision.ops.MLP(in_channels=input_dim, hidden_channels=[hidden_dim, 1], activation_layer=nn.ReLU)
        #self.decoder = torchvision.ops.MLP(in_channels=hidden_dim, hidden_channels=[input_dim], activation_layer=nn.ReLU)

#        self.initialize_weights()

    # def initialize_weights(self):
    #     for layer in self.modules():
    #         if isinstance(layer, nn.Linear):
    #             nn.init.xavier_uniform_(layer.weight)
    #             nn.init.constant_(layer.bias, 0.0)

    def forward(self, x):
        print(x.size())
        item_embeddings = self.embedding(x)
        print(item_embeddings.size())
        y = torch.relu(self.fc1(item_embeddings))  # Using ReLU activation in hidden layer
        output = self.fc2(y)  # Linear output (no activation function)
        
        return output
