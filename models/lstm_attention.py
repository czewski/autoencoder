import torch.nn as nn
import torch.nn.functional as F
import torch
import math
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# class PositionalEncoding(nn.Module):
#     def __init__(self, embedding_dim, max_len=5000):
#         super(PositionalEncoding, self).__init__()
#         self.encoding = torch.zeros(max_len, embedding_dim)
#         position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
#         div_term = torch.exp(torch.arange(0, embedding_dim, 2).float() * (-math.log(10000.0) / embedding_dim))
#         self.encoding[:, 0::2] = torch.sin(position * div_term)
#         self.encoding[:, 1::2] = torch.cos(position * div_term)
#         self.encoding = self.encoding.unsqueeze(0)

#     def forward(self, x):
#         seq_len = x.size(1)
#         return x + self.encoding[:, :seq_len, :].to(x.device)

class Attention(nn.Module):
    def __init__(self, hidden_size):
        super(Attention, self).__init__()
        self.hidden_size = hidden_size
        self.attention = nn.Linear(hidden_size, 1, bias=False)

    def forward(self, lstm_output):
        attention_weights = torch.softmax(self.attention(lstm_output), dim=1)
        context_vector = attention_weights * lstm_output
        context_vector = torch.sum(context_vector, dim=1)
        return context_vector

class LSTM_ATTENTION(nn.Module):
    def __init__(self, n_items, hidden_size, embedding_dim, batch_size, n_layers=1):
        super(LSTM_ATTENTION, self).__init__()

        self.n_items = n_items
        self.hidden_size = hidden_size
        self.batch_size = batch_size
        self.n_layers = n_layers
        self.embedding_dim = embedding_dim

        self.emb = nn.Embedding(self.n_items, self.embedding_dim, padding_idx=0)
        self.emb_dropout = nn.Dropout(0.25)

        #self.pos_enc = PositionalEncoding(embedding_dim)

        self.lstm = nn.LSTM(embedding_dim, hidden_size, num_layers=n_layers, batch_first=True)
        self.attention = Attention(hidden_size)
        self.fc = nn.Linear(hidden_size, n_items)

    def forward(self, seq, lengths):
        embs = self.emb_dropout(self.emb(seq))
        embs = self.pos_enc(embs)

        try:
            packed_embs = nn.utils.rnn.pack_padded_sequence(embs, lengths, batch_first=True, enforce_sorted=False)
            packed_output, (h_n, c_n) = self.lstm(packed_embs)
            lstm_output, _ = nn.utils.rnn.pad_packed_sequence(packed_output, batch_first=True)

            context_vector = self.attention(lstm_output)
            output = self.fc(context_vector)

            return output
        except Exception as e:
            print(f"Error in packing padded sequence: {e}")
            return None
        

class LSTMAttentionModel(nn.Module):
    def __init__(self, n_items, hidden_size, embedding_dim, batch_size, n_layers=1, drop_prob=0.5):
        super(LSTMAttentionModel, self).__init__()
        self.batch_size = batch_size
        self.output_size = n_items
        self.hidden_size = hidden_size
        self.vocab_size = n_items
        self.embedding_length = embedding_dim
        self.n_layers = n_layers

        self.embedding = nn.Embedding(n_items, embedding_dim)
        self.lstm = nn.LSTM(embedding_dim*19, hidden_size, n_layers, 
        					dropout=drop_prob, batch_first = True,)
        self.midl = nn.Linear(hidden_size*3, 150)
        self.drop = nn.Dropout(p=0.3)
        self.label = nn.Linear(150, n_items)
		
    def attention_net(self, lstm_output, final_state):
        hidden = final_state
        hidden = hidden.squeeze(1)
        hidden = torch.t(hidden)
        # print(hidden.squeeze(0).unsqueeze(2).shape)
        # attn_weights = torch.bmm(lstm_output, hidden.unsqueeze(2)).squeeze(2)
        # soft_attn_weights = F.softmax(attn_weights, 1)
        # new_hidden_state = torch.bmm(lstm_output.transpose(1, 2), soft_attn_weights.unsqueeze(2)).squeeze     (2)
        attn_weights = torch.bmm(lstm_output, hidden.unsqueeze(0)).squeeze(2)
        soft_attn_weights = F.softmax(attn_weights, 1)
        new_hidden_state = torch.bmm(lstm_output.transpose(1, 2), soft_attn_weights).squeeze(2)
        new_hidden_state = torch.flatten(new_hidden_state)
        return new_hidden_state.unsqueeze(0)
	
    def forward(self, x, lens):
        x = x.long()
        embs = self.drop(self.embedding(x))
        embs = embs.permute(1, 0, 2)  # Change to (batch_size, sequence_length, embedding_dim)
        embs = embs.contiguous().view(embs.size(0), -1)  # Flatten to (batch_size, sequence_length *        embedding_dim)

        output, (final_hidden_state, final_cell_state) = self.lstm(embs)
        print(output.size())
        print(final_hidden_state.size())

      
        attn_output = self.attention_net(output, final_hidden_state)
        m = self.drop(self.midl(attn_output))
        logits = self.label(m)

        return logits