import torch.nn as nn
import torch

class Decoder(nn.Module):
    """
    Initializing GRU Decoder. Based on the hidden state(context vector)
    my encoder has returned I want to make predictions to map 
    English to Klingon

    Parameters:
    ----------
    output_dim : int
        Size of the output vocabulary
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
        self.output_dim = output_dim
        self.hid_dim = hid_dim
        self.n_layers = n_layers
        self.attention = attention
        self.embedding = nn.Embedding(output_dim, emb_dim)
        self.rnn = nn.GRU(hid_dim * 2 + emb_dim, hid_dim, n_layers, dropout=dropout)
        self.fc_out = nn.Linear(hid_dim * 3 + emb_dim, output_dim)
        self.dropout = nn.Dropout(dropout)

    """
    Forward propagation step of decoding

    Parameters:
    ----------
    input : Tensor
        Input tensor containing token indices (seq_len, batch_size)
        This is what our tokenized Klingon Data
    
    hidden : Tensor
        Hidden tensor containing token indices (seq_len, batch_size)
        This is what our encoder returns
    
    encoder_outputs : Tensor
        Output tensor from the encoder (seq_len, batch_size, hidden_dim)
    
    Returns:
    -------
    output : Tensor
        Predicted output tensor from the GRU (seq_len, batch_size, output_dim)
    
    hidden : Tensor
        Hidden state tensor from the GRU (n_layers, batch_size, hid_dim)
    """
    def forward(self, input, hidden, encoder_outputs):
        input = input.unsqueeze(0)
        embedded = self.dropout(self.embedding(input))
        a = self.attention(hidden[-1], encoder_outputs).unsqueeze(1)  # Compute attention weights and unsqueeze for bmm
        encoder_outputs = encoder_outputs.permute(1, 0, 2)  # Permute encoder outputs for bmm
        weighted = torch.bmm(a, encoder_outputs)  # Batch matrix multiplication
        weighted = weighted.permute(1, 0, 2)  # Permute back to (1, batch_size, hidden_dim)
        rnn_input = torch.cat((embedded, weighted), dim=2)  # Concatenate embedded input and weighted encoder outputs
        output, hidden = self.rnn(rnn_input, hidden)  # Pass through RNN
        output = self.fc_out(torch.cat((output, weighted, embedded), dim=2))  # Pass through fully connected layer
        return output.squeeze(0), hidden
