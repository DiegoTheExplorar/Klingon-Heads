import torch.nn as nn

class Encoder(nn.Module):
    """
    Seq2Seq Encoder for GRU model

    int input_dim: Size of input
    int emb_dim: No.of dimensons of embedding vectors
    int hid_dim : No.of features in GRU's hidden state
    int n_layers : No.of GRU layers (Probably will be using 2)
    double dropout: Dropout probability of a neuron
    """
    def __init__(self, input_dim, emb_dim, hid_dim, n_layers, dropout):
        super().__init__()
        # Embedding layer
        self.embedding = nn.Embedding(input_dim, emb_dim)
        # GRU layer
        self.rnn = nn.GRU(emb_dim, hid_dim, n_layers, dropout=dropout)
        # Dropout layer
        self.dropout = nn.Dropout(dropout)
    
    """
    Forward Pro
    """
    def forward(self, src):

        embedded = self.dropout(self.embedding(src))

        outputs, hidden = self.rnn(embedded)

        return hidden
