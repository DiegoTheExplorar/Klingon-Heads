import torch.nn as nn

class Encoder(nn.Module):
    """
    Seq2Seq Encoder for GRU model. I want to store any kind
    of sequenital information to be passed on to the decoder
    
    Parameters:
    ----------
    input_dim : int
        Size of the input vocabulary
    emb_dim : int
        Dimension of the embedding vectors
    hid_dim : int
        Number of features in the GRU's hidden state
    n_layers : int
        Number of GRU layers (typically 2)
    dropout : float
        Dropout probability for the dropout layer
    """
    def __init__(self, input_dim, emb_dim, hid_dim, n_layers, dropout):
        super().__init__()
        # Embedding layer
        self.embedding = nn.Embedding(input_dim, emb_dim)
        self.hid_dim = hid_dim
        self.n_layers = n_layers
        self
        # GRU layer
        self.rnn = nn.GRU(emb_dim, hid_dim, n_layers, dropout=dropout)
        # Dropout layer
        self.dropout = nn.Dropout(dropout)
    
    """
        Forward propagation step of encoding
        
        Parameters:
        ----------
        input : Tensor
            Input tensor containing token indices (seq_len, batch_size)
        
        Returns:
        -------
        hidden : Tensor
            Hidden state tensor from the GRU (n_layers, batch_size, hid_dim)
        """
    def forward(self, input):
        #input is converted into embeddings 
        embedded = self.embedding(input)
        #forward pass into GRU and dropout probability is applied
        outputs, hidden = self.dropout(self.rnn(embedded))
        #only hidden state is required for encoding
        return hidden
