import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from Decoder import Decoder
from Encoder import Encoder
from Seq2SeqModel import Seq2SeqModel
from DataPreprocessing import preprocess

#getting processed data
(english_tokenizer, klingon_tokenizer, max_english_length, max_klingon_length,
    english_train_padded, klingon_train_input, klingon_train_target,
    english_test_padded, klingon_test_input, klingon_test_target) = preprocess()

# parameters
input_dim = len(english_tokenizer.word_index) + 1  # Add 1 for the padding token
output_dim = len(klingon_tokenizer.word_index) + 1  # Add 1 for the padding token
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')  # Use GPU if available, otherwise use CPU
n_layers = 2
#Hyperparameters up to change
emb_dim = 256
hid_dim = 512
dropout = 0.5
teacher_forcing_ratio = 0.5
#using encoder, decoder and intialising a Seq2Seq model
encoder = Encoder(input_dim, emb_dim, hid_dim, n_layers, dropout).to(device)
decoder = Decoder(output_dim, emb_dim, hid_dim, n_layers, dropout).to(device)
model = Seq2SeqModel(encoder, decoder, device).to(device)

#initalise model's weight
def init_weights(m):
    for name, param in m.named_parameters():
        nn.init.normal_(param.data, mean=0, std=0.01)
model.apply(init_weights)
#define loss function
criterion = nn.CrossEntropyLoss()
#optimizer
optimizer = optim.AdamW(model.parameters())
