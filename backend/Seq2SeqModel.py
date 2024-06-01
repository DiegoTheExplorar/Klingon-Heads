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
        # cause encoder and decoder must have same no.of layers
        assert (encoder.hid_dim == decoder.hid_dim), "Hidden dimensions of encoder and decoder not equal"
        assert (encoder.n_layers == decoder.n_layers), "Encoder and decoder layers not equal"

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
            the % of time I use ground-truths aka during training
        Returns:
        -------
        prediction : Tensor
            Predicted output tensor from the GRU (seq_len, batch_size, output_dim)
        
        hidden : Tensor
            Hidden state tensor from the GRU (n_layers, batch_size, hid_dim)
    """
    def forward(self,input, trg, teacher_forcing_ratio = 0.5):
        batch_size = trg.shape[1]
        trg_length = trg.shape[0]
        trg_size = self.decoder.output_dim
        #storing decorder outputs
        outputs = torch.zeros(trg_length,batch_size,trg_size).to(self.device)
        #output of encoder used as input for decoder
        encoder_outputs, hidden = self.encoder(input)
        #print("Encoder hidden state shape:", hidden.shape)
        # basically we want to single out the first input into the decoder as a 
        #start of sentence token. This is to let the decoder know when to start making predictions
        input = trg[0, :]
        for t in range(1, trg_length):
           #forward pass through decoder. hidden here refers to context vector from
           #encoder. hidden keeps getting updated
            output, hidden = self.decoder(input, hidden, encoder_outputs)
            #print("Decoder output shape:", output.shape)
            #Here I am just storing all the predictions made
            outputs[t] = output
            
            #leaving usage of teacher forcing to chance
            #teacher_force = random.random() < teacher_forcing_ratio
            #print("Output tensor shape in Seq to Seq:", output.shape)

            # Get the highest predicted token from our predictions
            highest = output.argmax(1)
            
            # If teacher forcing is used use next token else  use predicted token
            input = trg[t] if random.random() < teacher_forcing_ratio else highest


        return outputs
