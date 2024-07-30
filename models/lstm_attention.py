import torch.nn as nn
import torch.nn.functional as F
import torch
import math
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

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

## Self Attention // Dot product?
class LSTMAttentionModel(nn.Module):
    def __init__(self, n_items, hidden_size, embedding_dim, batch_size, n_layers=1, drop_prob=0.1, max_len=5000):
      super(LSTMAttentionModel, self).__init__()
      self.batch_size = batch_size
      self.output_size = n_items
      self.input_dim = embedding_dim
      self.hidden_size = hidden_size
  
      ## Embeddings
      self.embedding = nn.Embedding(n_items, embedding_dim, padding_idx=0)
      self.dropout = nn.Dropout(drop_prob)

      # Positional Encoding
      self.positional_encoding = PositionalEncoding(embedding_dim, max_len)

      ## RNN
      self.lstm = nn.LSTM(embedding_dim, hidden_size, n_layers, batch_first = False)  ##Em algum momento testei com emb*19 e batch_first = true
      
      ## Attention
      self.query = nn.Linear(hidden_size, hidden_size)
      self.key = nn.Linear(hidden_size, hidden_size) 
      self.value = nn.Linear(hidden_size, hidden_size)
      self.softmax = nn.Softmax(dim=2) #dim=2 ##Why dimension 2?
      

    # F.scaled_dot_product_attention(query, key, value)  
    def attention_net(self, lstm_output, padding_mask): 
      queries = self.query(lstm_output)
      keys = self.key(lstm_output)
      values = self.value(lstm_output)

      # Scaled dot-product attention  // factor was self.input_dim
      score = torch.bmm(queries, keys.transpose(1, 2))/(self.hidden_size**0.5) #keys.transpose(0, 1)

      # Apply the padding mask (masking out the padding tokens by setting their scores to -inf)
      if padding_mask is not None:
        score = score.masked_fill(padding_mask.unsqueeze(1) == 0, float('-inf'))

      attention = self.softmax(score)
      weighted = torch.bmm(attention, values)
      return weighted

    def forward(self, x, lengths):
      x = x.long()
      embs = self.dropout(self.embedding(x))
      embs = self.positional_encoding(embs)  # Apply positional encoding

      #Pack pad
      embs = pack_padded_sequence(embs, lengths)
      lstm_out, _ = self.lstm(embs) # _ = (final_hidden_state, final_cell_state)
      lstm_out, lengths = pad_packed_sequence(lstm_out)
      lstm_out = lstm_out.permute(1, 0, 2) # Change dimensions to: (batch_size, sequence_length, embedding_dim)

    #  padding_mask = (lstm_out != 0)#.unsqueeze(-2)
      padding_mask = (torch.sum(lstm_out, dim=-1) != 0) 

      attn_output = self.attention_net(lstm_out, padding_mask) 
      attn_output = torch.mean(attn_output, dim=1)  # (batch_size, hidden_size)

      # There's gotta be a better way to do this, maybe we can even introduce some more linear layers in here? 
      item_embs = self.embedding(torch.arange(self.output_size).to(x.device))  # Ensure the tensor is on the same device
      scores = torch.matmul(attn_output, item_embs.transpose(0, 1))
      return scores

# First test
    # def forward(self, x, lens):
    #   x = x.long()
    #   embs = self.embedding(x) #self.drop(
    #   embs = embs.permute(1, 0, 2)  # Change to (batch_size, sequence_length, embedding_dim)
    #   embs = embs.contiguous().view(embs.size(0), -1)  # Flatten to (batch_size, sequence_length *        embedding_dim)

    #   output, (final_hidden_state, final_cell_state) = self.lstm(embs)

    #   attn_output = self.attention_net(output)

    #   item_embs = self.embedding(torch.arange(self.output_size).to(device)) #.to(self.device)
    #   # print(item_embs.size())
    #   scores = torch.matmul(attn_output, item_embs.permute(1, 0))

    #   ##Layer de probs

    #   return scores