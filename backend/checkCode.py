import torch
import torch.nn as nn
import random
import numpy as np

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
        # GRU layer
        self.rnn = nn.GRU(emb_dim, hid_dim, n_layers, dropout=dropout, bidirectional=True)
        # Dropout layer
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(hid_dim * 2, hid_dim)
    
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
        embedded = self.dropout(self.embedding(input))
        outputs, hidden = self.rnn(embedded)
        #forward pass into GRU and dropout probability is applied
        hidden = torch.tanh(self.fc(torch.cat((hidden[-2,:,:], hidden[-1,:,:]), dim=1)))
        hidden = hidden.unsqueeze(0).repeat(self.n_layers, 1, 1)  # Repeat hidden state for n_layers
        return outputs, hidden


class Attention(nn.Module):
    def __init__(self, hid_dim):
        super().__init__()
        self.attn = nn.Linear(hid_dim * 3, hid_dim)
        self.v = nn.Linear(hid_dim, 1, bias=False)

    def forward(self, hidden, encoder_outputs):
        src_len = encoder_outputs.shape[0]
        hidden = hidden.repeat(src_len, 1, 1)  # Repeat hidden state src_len times
        energy = torch.tanh(self.attn(torch.cat((hidden, encoder_outputs), dim=2)))  # Concatenate and apply attention layer
        attention = self.v(energy).squeeze(2).permute(1, 0)  # Apply linear layer and permute for softmax
        return torch.softmax(attention, dim=1)
    
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

class Seq2SeqModel(nn.Module):
    def __init__(self, encoder, decoder, device):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.device = device
        # Ensure encoder and decoder have the same number of layers and hidden dimensions
        assert (encoder.hid_dim == decoder.hid_dim), "Hidden dimensions of encoder and decoder must be equal"
        assert (encoder.n_layers == decoder.n_layers), "Number of layers in encoder and decoder must be equal"

    """
    Parameters:
    ----------
    input : Tensor
        Input tensor containing token indices (seq_len, batch_size)
        Tokenized English Data
    
    trg : Tensor
        Target tensor containing token indices (seq_len, batch_size)
        This is what our tokenized Klingon Data
    
    teacher_forcing_ratio: double
        The percentage of time I use ground-truths aka during training
    
    Returns:
    -------
    outputs : Tensor
        Predicted output tensor from the GRU (seq_len, batch_size, output_dim)
    """
    def forward(self, input, trg, teacher_forcing_ratio=0.5):
        batch_size = trg.shape[1]
        trg_len = trg.shape[0]
        trg_vocab_size = self.decoder.output_dim
        outputs = torch.zeros(trg_len, batch_size, trg_vocab_size).to(self.device)
        encoder_outputs, hidden = self.encoder(input)
        input = trg[0, :]
        for t in range(1, trg_len):
            output, hidden = self.decoder(input, hidden, encoder_outputs)
            outputs[t] = output
            top1 = output.argmax(1)
            input = trg[t] if random.random() < teacher_forcing_ratio else top1
        return outputs

if __name__ == "__main__":
    INPUT_DIM = 10
    OUTPUT_DIM = 10
    ENC_EMB_DIM = 256
    DEC_EMB_DIM = 256
    HID_DIM = 512
    N_LAYERS = 2
    ENC_DROPOUT = 0.5
    DEC_DROPOUT = 0.5
    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    enc = Encoder(INPUT_DIM, ENC_EMB_DIM, HID_DIM, N_LAYERS, ENC_DROPOUT).to(DEVICE)
    attn = Attention(HID_DIM).to(DEVICE)
    dec = Decoder(OUTPUT_DIM, DEC_EMB_DIM, HID_DIM, N_LAYERS, DEC_DROPOUT, attn).to(DEVICE)
    model = Seq2SeqModel(enc, dec, DEVICE).to(DEVICE)

    print(model)

    src = torch.randint(0, INPUT_DIM, (5, 32)).to(DEVICE)  # (sequence length, batch size)
    trg = torch.randint(0, OUTPUT_DIM, (10, 32)).to(DEVICE)  # (sequence length, batch size)

    outputs = model(src, trg, 0.5)
    print(outputs.shape)  # Should print the shape of the outputs tensor
