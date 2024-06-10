import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm
from Decoder import Decoder
from Encoder import Encoder
from Seq2SeqModel import Seq2SeqModel
from DataPPwithspecial import preprocess

# Getting processed data
(klingon_tokenizer, english_tokenizer, max_klingon_length,
 klingon_train_padded, english_train_input, english_train_target,
 klingon_test_padded, english_test_input, english_test_target) = preprocess()

# Parameters
input_dim = len(klingon_tokenizer.word_index) + 1  # Add 1 for the padding token
output_dim = len(english_tokenizer.word_index) + 1  # Add 1 for the padding token
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')  # Use GPU if available, otherwise use CPU
n_layers = 2

# Hyperparameters (adjustable)
emb_dim = 256
hid_dim = 512
dropout = 0.5
teacher_forcing_ratio = 0.5
epoch_count = 25
batch_size = 16

# Using encoder, decoder, and initializing a Seq2Seq model
encoder = Encoder(input_dim, emb_dim, hid_dim, n_layers, dropout).to(device)
decoder = Decoder(output_dim, emb_dim, hid_dim, n_layers, dropout).to(device)
model = Seq2SeqModel(encoder, decoder, device).to(device)
print('Assembled model')

# Initialize model's weight
def init_weights(m):
    for name, param in m.named_parameters():
        nn.init.normal_(param.data, mean=0, std=0.01)
model.apply(init_weights)

# Define loss function
criterion = nn.CrossEntropyLoss()

# Optimizer
optimizer = optim.AdamW(model.parameters())

# Initializing DataLoader to combine tokenized and padded Klingon and English sentences into a dataset
train_dataset = TensorDataset(torch.tensor(klingon_train_padded, dtype=torch.long),
                              torch.tensor(english_train_input, dtype=torch.long),
                              torch.tensor(english_train_target, dtype=torch.long))
train_loader = DataLoader(train_dataset, batch_size, shuffle=True)
print('Loaded data and starting training')

for epoch in range(epoch_count):
    epoch_loss = 0
    for src, trg_input, trg in tqdm(train_loader, desc=f'Epoch {epoch + 1}/{epoch_count}', leave=False):
        src = src.transpose(0, 1).to(device)  # (seq_len, batch_size)
        trg_input = trg_input.transpose(0, 1).to(device)  # (seq_len, batch_size)
        trg = trg.transpose(0, 1).to(device)  # (seq_len, batch_size, 1)

        optimizer.zero_grad()
        output = model(src, trg_input, teacher_forcing_ratio)

        output_dim = decoder.output_dim
        output = output[1:].reshape(-1, output_dim)
        trg = trg[1:].reshape(-1)

        loss = criterion(output, trg)
        loss.backward()
        optimizer.step()

        epoch_loss += loss.item()
    avg_epoch_loss = epoch_loss / len(train_loader)
    print(f'Epoch [{epoch + 1}/{epoch_count}], Loss: {avg_epoch_loss:.4f}')

torch.save(model.state_dict(), 'Klingon_to_English.pth')

print('Now starting eval process')
evaluation_dataset = TensorDataset(torch.tensor(klingon_test_padded, dtype=torch.long),
                                    torch.tensor(english_test_input, dtype=torch.long),
                                    torch.tensor(english_test_target, dtype=torch.long))
evaluation_loader = DataLoader(evaluation_dataset, batch_size, shuffle=False)

model.eval()  # Set the model to evaluation mode
eval_loss = 0

print('Starting eval')
with torch.no_grad():
    for src, trg_input, trg in tqdm(evaluation_loader, desc='Evaluating', leave=False):
        src = src.transpose(0, 1).to(device)
        trg_input = trg_input.transpose(0, 1).to(device)
        trg = trg.transpose(0, 1).to(device)

        output = model(src, trg_input, teacher_forcing_ratio=0)  # Turn off teacher forcing during evaluation

        output = output[1:].reshape(-1, output_dim)
        trg = trg[1:].reshape(-1)

        loss = criterion(output, trg)

        eval_loss += loss.item()

avg_eval_loss = eval_loss / len(evaluation_loader)
print(f'Evaluation Loss: {avg_eval_loss:.4f}')
