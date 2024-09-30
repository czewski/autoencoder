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
    
def find_closest_tensor(query_embeddings, data_embeddings):
    closest_tensors = []
    
    for t in range(query_embeddings.size(0)):  # iterate over sequence length
        # query: (batch_size, dimension) at each time step t
        query_at_t = query_embeddings[t]
        data_embeddings = data_embeddings.to(device)
        
        # Calculate cosine similarity for each batch 
        dists = 1 - F.cosine_similarity(query_at_t.unsqueeze(1), data_embeddings.unsqueeze(0), dim=-1)
        min_dist, closest_index = torch.min(dists, dim=1) 
        closest_tensor = data_embeddings[closest_index] 
        
        closest_tensors.append(closest_tensor) 
    
    return torch.stack(closest_tensors)

## Self Attention // Dot product?
class LSTMAttentionModel(nn.Module): #embedding_matrix
    def __init__(self, n_items, hidden_size, embedding_dim, batch_size, alignment_func, pos_enc, data_embeddings, knn_helper, n_layers=1, drop_prob=0.25, max_len=5000):
      super(LSTMAttentionModel, self).__init__()
      self.batch_size = batch_size
      self.output_size = n_items
      self.input_dim = embedding_dim
      self.hidden_size = hidden_size
      self.alignment_func = alignment_func
      self.pos_enc = pos_enc

      # KNN
      self.use_knn = True
      self.knn_helper = knn_helper
      self.data_embeddings = data_embeddings 
  
      ## Embeddings
      #self.embedding = nn.Embedding.from_pretrained(embedding_matrix, freeze=True, padding_idx=0)
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

      # Linear layer to map from hidden size to embedding size
      self.hidden_to_embedding = nn.Linear(hidden_size, embedding_dim)
      
    # F.scaled_dot_product_attention(query, key, value)  
    def attention_net(self, lstm_output, padding_mask): 
      queries = self.query(lstm_output)
      keys = self.key(lstm_output)
      values = self.value(lstm_output)

      ## Alignment Functions (sdp, dp, additive, concat, biased_general, general, similarity)
      if self.alignment_func == 'sdp': 
        score = torch.bmm(queries, keys.transpose(1, 2))/(self.hidden_size**0.5) #keys.transpose(0, 1)

      elif self.alignment_func == 'dp':
        score = torch.bmm(queries, keys.transpose(1, 2))

      elif self.alignment_func == "general":
        # General attention: a(q, k) = q^T W k
        W_k = self.key.weight
        score = torch.bmm(queries, torch.matmul(keys, W_k).transpose(1, 2))

      elif self.alignment_func == "biased_general":
        # Biased General attention: a(q, k) = k^T W (q + b)
        bias = self.key.bias
        W_k = self.key.weight
        score = torch.bmm(keys, torch.matmul(W_k, (queries + bias).transpose(1, 2)))

      elif self.alignment_func == "concat":
        W_concat = nn.Linear(self.hidden_size * 2, self.hidden_size).to(queries.device) 
        concat_input = torch.cat((queries, keys), dim=-1)
        score = torch.tanh(W_concat(concat_input))
        score = torch.bmm(score, values.transpose(1, 2))
      
      elif self.alignment_func == "additive":
        W1 = nn.Linear(self.hidden_size, self.hidden_size).to(queries.device) 
        W2 = nn.Linear(self.hidden_size, self.hidden_size).to(keys.device) 
        score = torch.tanh(W1(queries) + W2(keys))
        score = torch.bmm(score, values.transpose(1, 2))

      # Apply the padding mask (masking out the padding tokens by setting their scores to -inf)
      if padding_mask is not None:
        score = score.masked_fill(padding_mask.unsqueeze(1) == 0, float('-inf'))

      attention = self.softmax(score)
      weighted = torch.bmm(attention, values)
      return weighted

    def forward(self, x, lengths):
      x = x.long()
      embs = self.dropout(self.embedding(x))
      initial_embs = embs.clone().detach()

      if self.pos_enc: 
        embs = self.positional_encoding(embs)  # Apply positional encoding

      #Pack pad
      embs = pack_padded_sequence(embs, lengths)
      lstm_out, _ = self.lstm(embs) # _ = (final_hidden_state, final_cell_state)
      lstm_out, lengths = pad_packed_sequence(lstm_out)
      lstm_out = lstm_out.permute(1, 0, 2) # Change dimensions to: (batch_size, sequence_length, embedding_dim)

      #padding_mask = (lstm_out != 0)#.unsqueeze(-2)
      padding_mask = (torch.sum(lstm_out, dim=-1) != 0) 

      attn_output = self.attention_net(lstm_out, padding_mask) 
      attn_output = torch.mean(attn_output, dim=1)  # (batch_size, hidden_size)

      # Linear layer to map from hidden size to embedding size
      attn_output = self.hidden_to_embedding(attn_output)  # (batch_size, embedding_dim)

      if self.use_knn: # maybe also need to add a % of the tensor...
        closest_tensor = find_closest_tensor(initial_embs, self.data_embeddings)  # Use KNN to find the closest tensor in the dataset  
        closest_tensor = torch.mean(closest_tensor, dim=0) 
        #closest_tensor = torch.mul(closest_tensor, 0.25)
        attn_output = attn_output * closest_tensor

      # There's gotta be a better way to do this, maybe we can even introduce some more linear layers in here? 
      item_embs = self.embedding(torch.arange(self.output_size).to(x.device))  
      scores = torch.matmul(attn_output, item_embs.transpose(0, 1))
      return scores