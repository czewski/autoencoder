import torch.nn as nn
import torch.nn.functional as F
import torch
import torch.nn.init as weight_init
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

class AutoEncoderEmbedding(nn.Module):
    def __init__(self, device, hidden_dim=150, embedding_dim=150, n_itens=43098):
        super(AutoEncoderEmbedding, self).__init__()
        self.device = device
        self.n_itens = n_itens
        self.hidden_size = hidden_dim

        self.embedding = nn.Embedding(n_itens, embedding_dim, padding_idx=0) #Actually if padding idx really consider the zeros, that will help a lot
        self.emb_dropout = nn.Dropout(p=0.25)
        
        self.encoder = nn.LSTM(input_size=embedding_dim, hidden_size=hidden_dim, batch_first=False)
        self.decoder = nn.LSTM(input_size=hidden_dim, hidden_size=hidden_dim, batch_first=False)
        self.output_layer = nn.Linear(embedding_dim, hidden_dim) ##Bias false? why would i want to have no bias
        
        self.relu = nn.ReLU()
        self.initialize_weights()

    def initialize_weights(self):
        for layer in self.modules():
            if isinstance(layer, nn.Linear):
                nn.init.xavier_uniform_(layer.weight)
                nn.init.constant_(layer.bias, 0.0)

    def forward(self, seq, lengths):
        #hidden = self.init_hidden(seq.size(1))
        #seq = seq.transpose(0,1)
        # print("hidden.size()")
        # print(hidden.size())

        #print(seq.size())
        embs = self.emb_dropout(self.embedding(seq))
        embs = pack_padded_sequence(embs, lengths)

        # Encode
        enc_out, hidden = self.encoder(embs)
        enc_out, lengths = pad_packed_sequence(enc_out)
        # print("enc_out.size()")
        # print(enc_out.size())

        # Decode
        dec_out, hidden = self.decoder(enc_out)
        
        # Scores
        item_embs = self.embedding(torch.arange(self.n_itens).to(self.device))
        # print("dec_out.size()")
        # print(dec_out.size())

        # print("self.output_layer(item_embs)")
        # print(self.output_layer(item_embs).permute(1, 0).size())


        # print("sum")
        # print(torch.sum(dec_out, 1).size())

        scores = torch.matmul(torch.sum(dec_out.permute(1,0,2), 1), self.output_layer(item_embs).permute(1, 0))

        return scores

    def init_hidden(self, batch_size):
        #self.n_layers, b_s, self.hidden_size
        return torch.zeros((1, batch_size), requires_grad=True).to(self.device)
        