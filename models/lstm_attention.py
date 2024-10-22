import torch.nn as nn
import torch.nn.functional as F
import torch
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
import torch.nn.init as init
import math

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class MultiHeadAttention(nn.Module):
    def __init__(self, hidden_size, num_heads):
        super(MultiHeadAttention, self).__init__()
        self.num_heads = num_heads
        self.hidden_size = hidden_size
        self.head_dim = hidden_size // num_heads
        print(self.head_dim)
        print(num_heads)

        assert self.head_dim * num_heads == hidden_size, "hidden_size must be divisible by num_heads"

        self.query = nn.Linear(hidden_size, hidden_size).to(device) 
        self.key = nn.Linear(hidden_size, hidden_size).to(device) 
        self.value = nn.Linear(hidden_size, hidden_size).to(device) 
        self.fc_out = nn.Linear(hidden_size, hidden_size).to(device) 
        self.softmax = nn.Softmax(dim=-1)
           
    def forward(self, lstm_output, padding_mask):
        batch_size = lstm_output.size(0)

        ## Linear projections
        queries = self.query(lstm_output)
        keys = self.key(lstm_output)
        values = self.value(lstm_output)
        queries = queries.reshape(batch_size * self.num_heads, -1, self.head_dim)
        keys = keys.reshape(batch_size * self.num_heads, -1, self.head_dim)
        values = values.reshape(batch_size * self.num_heads, -1, self.head_dim)

        ## Alignment Functions
        score = torch.bmm(queries, keys.transpose(-1, -2))/(self.hidden_size**0.5) 

        if padding_mask is not None:
            score = score.masked_fill(padding_mask.unsqueeze(1).unsqueeze(2) == 0, float('-inf'))

        attention = self.softmax(score) 
        weighted_values = torch.matmul(attention, values)  # (batch_size, num_heads, seq_len, head_dim)
        weighted_values = weighted_values.transpose(1, 2).contiguous().view(batch_size, -1, self.hidden_size)
        output = self.fc_out(weighted_values)
        return output

class PositionalEncoding(nn.Module):
    def __init__(self, embedding_dim, max_len=100):
        super(PositionalEncoding, self).__init__()
        self.encoding = torch.zeros(max_len, embedding_dim)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, embedding_dim, 2).float() * (-math.log(10000.0) / embedding_dim))
        self.encoding[:, 0::2] = torch.sin(position * div_term)
        self.encoding[:, 1::2] = torch.cos(position * div_term)
        self.encoding = self.encoding.unsqueeze(0)

    def forward(self, x):
        seq_len = x.size(1)
        return x + self.encoding[:, :seq_len, :].to(x.device)
    
class LSTMAttentionModel(nn.Module): 
    def __init__(self, n_items, hidden_size, embedding_dim, batch_size, pos_enc, num_heads, n_layers=1, drop_prob=0.25, max_len=5000):
      super(LSTMAttentionModel, self).__init__()
      self.batch_size = batch_size
      self.output_size = n_items
      self.input_dim = embedding_dim
      self.hidden_size = hidden_size
      self.pos_enc = pos_enc
      self.num_heads = num_heads

      ## Embeddings
      self.embedding = nn.Embedding(n_items, embedding_dim, padding_idx=0)
      self.dropout = nn.Dropout(drop_prob)

      # Positional Encoding
      if self.pos_enc == "True":
        self.positional_encoding = PositionalEncoding(embedding_dim, max_len)

      ## RNN
      self.lstm = nn.LSTM(embedding_dim, hidden_size, n_layers, batch_first = False)  ##Em algum momento testei com emb*19 e batch_first = true
      
      ## Multi-Head Attention
      self.multihead_attention = MultiHeadAttention(hidden_size, num_heads)

      ## Linear layer to map from hidden size to embedding size
      self.embedding_to_hidden = nn.Linear(embedding_dim, hidden_size)
      self.hidden_to_embedding = nn.Linear(hidden_size, embedding_dim)

      self._initialize_weights()

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    init.constant_(m.bias, 0)
            elif isinstance(m, nn.LSTM):
                for param in m.parameters():
                    if len(param.shape) >= 2:  # Weights (2D matrices)
                        init.xavier_uniform_(param)
                    else:  # Biases (1D vectors)
                        init.constant_(param, 0)
      
    def forward(self, x, lengths):
      x = x.long()
      embs = self.dropout(self.embedding(x))

      if self.pos_enc == "True": 
        embs = self.positional_encoding(embs)  # Apply positional encoding
        embs = self.embedding_to_hidden(embs) 
      else: #Pack pad
        embs = pack_padded_sequence(embs, lengths)
        embs, _ = self.lstm(embs) # _ = (final_hidden_state, final_cell_state)
        embs, lengths = pad_packed_sequence(embs)
        
      embs = embs.permute(1, 0, 2) # Change dimensions to: (batch_size, sequence_length, embedding_dim)
      padding_mask = (torch.sum(embs, dim=-1) != 0) 
      attn_output = self.multihead_attention(embs, padding_mask)
      attn_output = torch.mean(attn_output, dim=1)  # (batch_size, hidden_size)
      attn_output = self.hidden_to_embedding(attn_output)  # Linear layer to map from hidden size to embedding size (batch_size, embedding_dim)

      # There's gotta be a better way to do this, maybe we can even introduce some more linear layers in here? 
      item_embs = self.embedding(torch.arange(self.output_size).to(x.device))  
      scores = torch.matmul(attn_output, item_embs.transpose(0, 1))
      return scores