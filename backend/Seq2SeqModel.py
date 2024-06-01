import torch.nn as nn
import torch
import random
"""
    This class puts together the decoder and encoder and 
    receives Klingon and Engish data from the tokenization process

"""

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
