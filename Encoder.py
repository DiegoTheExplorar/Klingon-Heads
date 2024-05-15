import torch.nn as nn

class Encoder(nn.Module):
    def __init__(self, input_dim, emb_dim, hid_dim, n_layers, dropout):
        super().__init__()
        
        # Embedding layer
        self.embedding = nn.Embedding(input_dim, emb_dim)
        
        # GRU layer
        self.rnn = nn.GRU(emb_dim, hid_dim, n_layers, dropout=dropout)
        
        # Dropout layer
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, src):

        embedded = self.dropout(self.embedding(src))

        outputs, hidden = self.rnn(embedded)

        return hidden
