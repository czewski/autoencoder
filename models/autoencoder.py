import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
import torch.nn.parallel
import torch.utils.data
from torch.autograd import Variable
import torch.nn.init as weight_init

class Encoder(nn.Module):
    def __init__(self, L, activation_fn='sigmoid', drop_prob=0.0):
        super(Encoder, self).__init__()
        # layers = self.create_nn_structure(L)
        # self.num_layers = len(L)
        # print(layers)
        # print("-"*10)

        self._drop_prob = drop_prob
        if drop_prob > 0.0:
            self.dropout = nn.Dropout(drop_prob)        

        input_size = 50
        latent_size = 10
        
        # Não teria que ser module list? já que to linkando as camadas depois?
        # Ou quem sabe separar uma por uma, ou em encode/decode        
        self.linears = nn.Sequential(
            nn.Linear(in_features=input_size, out_features=19),
            nn.Linear(in_features=19, out_features=latent_size),
            nn.Linear(latent_size, 19),
            nn.Linear(19, input_size),
        )
        
    def forward(self, seq, lengths):
        #print(seq.size())
        #x = x.to(self.linears[0].weight.dtype)
        x = seq.float()

        for i,layer in enumerate(self.linears):
            if i <= len(self.linears)-2:
                act_fn = F.relu #nn.Sigmoid() 
                x = act_fn(layer(x))

                if self._drop_prob > 0.0 and i <= int(self.num_layers/2): 
                    x = self.dropout(x)
        
        # No activation on the last decoding layer
        x = self.linears[-1](x)

        # Aplicar softmax na última camada
        prob_matrix = F.softmax(x, dim=1)
        #print(prob_matrix.size())
        return prob_matrix


    def create_nn_structure(self, L):
        max_ind = len(L)-1
        layers = []

        for i,v in enumerate(L):
            if i < max_ind:
                #still have i+1 available, create layer tuple
                layer = [v,L[i+1]]
                layers.append(layer)

        #then inverse the layers for decoder size
        encoder_layers = layers[:]
        for l in encoder_layers[::-1]:
            decoder_layer = l[::-1]
            layers.append(decoder_layer)
        return layers