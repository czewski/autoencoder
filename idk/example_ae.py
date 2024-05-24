import torch.nn as nn

class Encoder(nn.Module):
    def __init__(self, seq_len, n_features, embedding_dim=64):
        super(Encoder, self).__init__()
        self.seq_len, self.n_features = seq_len, n_features
        self.embedding_dim, self.hidden_dim = embedding_dim, 2 * embedding_dim
        self.rnn1 = nn.LSTM(
            input_size=n_features,
            hidden_size=self.hidden_dim,
            num_layers=1,
            batch_first=True
        )
        self.rnn2 = nn.LSTM(
            input_size=self.hidden_dim,
            hidden_size=embedding_dim,
            num_layers=1,
            batch_first=True
        )

                #self.encoder.apply(self.weight_init)
        #self.decoder.apply(self.weight_init)

        #Method 2
        # Encoder layers
        # self.encoder = nn.Sequential(
        #     nn.Linear(19, 256),
        #     nn.ReLU(),
        #     nn.Linear(256, hidden_dim),
        #     nn.ReLU()
        # )

        # # Decoder layers
        # self.decoder = nn.Sequential(
        #     nn.Linear(hidden_dim, 256),
        #     nn.ReLU(),
        #     nn.Linear(256, 19),
        #     nn.Softmax() 
        # )

        #Method 1
        # self.encoder = nn.Linear(19, hidden_dim)  # Adjusted encoder input size
        # self.decoder = nn.Linear(hidden_dim, 19)  # Adjusted decoder output size

        #Initialize weights to ONE
        # for layer in self.encoder:
        #     if isinstance(layer, nn.Linear):
        #         weight_init.constant_(layer.weight, 1)
        #         weight_init.constant_(layer.bias, 0)  
        
        # #Initialize weights to ONE
        # for layer in self.decoder:
        #     if isinstance(layer, nn.Linear):
        #         weight_init.constant_(layer.weight, 1)
        #         weight_init.constant_(layer.bias, 0)  
    

    def forward(self, x):
        batch_size = x.shape[0]
        # print(f'ENCODER input dim: {x.shape}')
        x = x.reshape((batch_size, self.seq_len, self.n_features))
        # print(f'ENCODER reshaped dim: {x.shape}')
        x, (_, _) = self.rnn1(x)
        # print(f'ENCODER output rnn1 dim: {x.shape}')
        x, (hidden_n, _) = self.rnn2(x)
        # print(f'ENCODER output rnn2 dim: {x.shape}')
        # print(f'ENCODER hidden_n rnn2 dim: {hidden_n.shape}')
        # print(f'ENCODER hidden_n wants to be reshaped to : {(batch_size, self.embedding_dim)}')
                #Method 4
        # x = torch.sigmoid(self.encoder(x))
        # x = torch.sigmoid(self.decoder(x))

        #Method 3
        #x = F.normalize(x)
        #x = self.input_dropout(x)
        
        # for i, layer in enumerate(self.encoder):
        #     x = layer(x)
        #     if i != len(self.encoder) - 1:
        #         x = F.relu(x)
        #         #x = torch.tanh(x)

        # for i, layer in enumerate(self.decoder):
        #     x = layer(x)
        #     if i != len(self.decoder) - 1:
        #         x = F.relu(x)
        #         #x = torch.tanh(x)
        
        #Method 2
        # x = self.encoder(x)
        # x = self.decoder(x)

        #Method 1
        # encoded = F.relu(self.encoder(input))  # Pass through encoder with ReLU activation
        # #decoded = torch.sigmoid(self.decoder(encoded))  # Pass through decoder with sigmoid activation
        # decoded = torch.softmax(self.decoder(encoded), dim=1)  # Pass through decoder with sigmoid activation

                # x = x.permute(1, 0).unsqueeze(-1)

        # lstm_out, _ = self.lstm(x)
        # lstm_out = lstm_out[:, -1, :]

        # # Pass through Encoder
        # for i, layer in enumerate(self.encoder):
        #     lstm_out = layer(lstm_out)
        #     if i != len(self.encoder) - 1:
        #         lstm_out = F.relu(lstm_out)

        # # Pass through Decoder
        # for i, layer in enumerate(self.decoder):
        #     lstm_out = layer(lstm_out)
        #     if i != len(self.decoder) - 1:
        #         lstm_out = F.relu(lstm_out)

        return hidden_n.reshape((batch_size, self.embedding_dim))


class Decoder(nn.Module):
    def __init__(self, seq_len, input_dim=64, n_features=1):
        super(Decoder, self).__init__()
        self.seq_len, self.input_dim = seq_len, input_dim
        self.hidden_dim, self.n_features = 2 * input_dim, n_features
        self.rnn1 = nn.LSTM(
            input_size=input_dim,
            hidden_size=input_dim,
            num_layers=1,
            batch_first=True
        )
        self.rnn2 = nn.LSTM(
            input_size=input_dim,
            hidden_size=self.hidden_dim,
            num_layers=1,
            batch_first=True
        )
        self.output_layer = nn.Linear(self.hidden_dim, n_features)

    def forward(self, x):
        batch_size = x.shape[0]
        # print(f'DECODER input dim: {x.shape}')
        x = x.repeat(self.seq_len, self.n_features) # todo testare se funziona con più feature
        # print(f'DECODER repeat dim: {x.shape}')
        x = x.reshape((batch_size, self.seq_len, self.input_dim))
        # print(f'DECODER reshaped dim: {x.shape}')
        x, (hidden_n, cell_n) = self.rnn1(x)
        # print(f'DECODER output rnn1 dim:/ {x.shape}')
        x, (hidden_n, cell_n) = self.rnn2(x)
        x = x.reshape((batch_size, self.seq_len, self.hidden_dim))
        return self.output_layer(x)


######
# MAIN
######


class RecurrentAutoencoder(nn.Module):
    def __init__(self, seq_len, n_features, embedding_dim=64, device='cuda', batch_size=64):
        super(RecurrentAutoencoder, self).__init__()
        self.encoder = Encoder(seq_len, n_features, embedding_dim).to(device)
        self.decoder = Decoder(seq_len, embedding_dim, n_features).to(device)

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x
