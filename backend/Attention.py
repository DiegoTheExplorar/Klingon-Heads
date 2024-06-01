import torch
import torch.nn as nn

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

