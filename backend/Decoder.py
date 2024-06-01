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
    def __init__(self, output_dim, emb_dim, hid_dim, n_layers, dropout, attention):
        super().__init__()
        #self.hid_dim = hid_dim
        self.output_dim = output_dim
        #self.n_layers = n_layers
        self.embedding = nn.Embedding(output_dim, emb_dim)
        self.rnn = nn.GRU(emb_dim, hid_dim, n_layers, dropout=dropout)
        self.fc_out = nn.Linear(hid_dim, output_dim)
        self.dropout = nn.Dropout(dropout)
        self.attention = attention

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
    
    def forward(self, input, hidden, encoder_outputs):
        input = input.unsqueeze(0)
        embedded = self.dropout(self.embedding(input))
        a = self.attention(hidden[-1], encoder_outputs).unsqueeze(1)
        weighted = torch.bmm(a, encoder_outputs.permute(1, 0, 2)).permute(1, 0, 2)
        rnn_input = torch.cat((embedded, weighted), dim=2)
        output, hidden = self.rnn(rnn_input, hidden)
        output = self.fc_out(torch.cat((output, weighted, embedded), dim=2))
        return output.squeeze(0), hidden
