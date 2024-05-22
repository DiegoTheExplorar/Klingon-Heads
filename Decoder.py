import torch.nn as nn
class Decoder(nn.Module):
    """
    Initailising GRU Decoder. Based on the hidden state(context vector)
    my encoder has returned I want too make predictions to map 
    English to Klingon

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
    def __init__(self, output_dim, emb_dim, hid_dim, n_layers, dropout):
        super().__init__()
        self.hid_dim = hid_dim
        self.output_dim = output_dim
        self.n_layers = n_layers
        self.embedding = nn.Embedding(output_dim, emb_dim)
        self.rnn = nn.GRU(emb_dim, hid_dim, n_layers, dropout=dropout)
        self.fc_out = nn.Linear(hid_dim, output_dim)
        self.dropout = nn.Dropout(dropout)

    """
        Forward propagation step of decoding
        
        Parameters:
        ----------
        hidden : Tensor
            Hidden tensor containing token indices (seq_len, batch_size)
            This is what our encoder returns
        
        trg : Tensor
            Target tensor containing token indices (seq_len, batch_size)
            This is what our tokenized Klingon Data
        
        Returns:
        -------
        prediction : Tensor
            Predicted output tensor from the GRU (seq_len, batch_size, output_dim)
        
        hidden : Tensor
            Hidden state tensor from the GRU (n_layers, batch_size, hid_dim)
    """
    
    def forward(self, trg, hidden):
        #unsure trg is 3D
        trg = trg.unsqueeze(0)
        #input is converted into embeddings and dropout probability is applied
        embedded = self.dropout(self.embedding(trg))
        print("Embedded shape:", embedded.shape)
        #GRU layer computes new context based on previous context
        output, hidden = self.rnn(embedded, hidden)
        print("Output shape after RNN:", output.shape)
        #predicts output from GRU
        prediction = self.fc_out(output.squeeze(0))
        print("Output shape after fc_out:", output.shape)
        return prediction, hidden