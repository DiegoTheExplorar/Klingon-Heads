import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm
from Decoder import Decoder
from Encoder import Encoder
from Seq2SeqModel import Seq2SeqModel
from DataPPwithspecial import preprocess_with_special_tokens

#getting processed data
(english_tokenizer, klingon_tokenizer, max_english_length, max_klingon_length,
    english_train_padded, klingon_train_input, klingon_train_target,
    english_test_padded, klingon_test_input, klingon_test_target) = preprocess_with_special_tokens()

# parameters
input_dim = len(english_tokenizer.word_index)  # Add 1 for the padding token
output_dim = len(klingon_tokenizer.word_index)  # Add 1 for the padding token
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')  # Use GPU if available, otherwise use CPU
n_layers = 2
#Hyperparameters up to change
emb_dim = 256
hid_dim = 512
dropout = 0.5
#determines how often the ground truth target (i.e., the correct token) is used as the next input 
#to the decoder during training, as opposed to using the decoder's own previous prediction. 
teacher_forcing_ratio = 0.5
#no.of times model is pushed through the dataset
epoch_count = 20
#batch size is just the number of subsets of the dataset to be pushed through
batch_size = 32
#using encoder, decoder and intialising a Seq2Seq model
encoder = Encoder(input_dim, emb_dim, hid_dim, n_layers, dropout).to(device)
decoder = Decoder(output_dim, emb_dim, hid_dim, n_layers, dropout).to(device)
model = Seq2SeqModel(encoder, decoder, device).to(device)
print('Assembled model')
#initalise model's weight
def init_weights(m):
    for name, param in m.named_parameters():
        nn.init.normal_(param.data, mean=0, std=0.01)
model.apply(init_weights)
#define loss function
criterion = nn.CrossEntropyLoss()
#optimizer
optimizer = optim.AdamW(model.parameters())

#initialising DataLoader to combine tokenized and padded English and Klingon sentences into a dataset
train_dataset = TensorDataset(torch.tensor(english_train_padded, dtype=torch.long),
                              torch.tensor(klingon_train_input, dtype=torch.long),
                              torch.tensor(klingon_train_target, dtype=torch.long))
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
print('loaded data and starting training')
for epoch in range(epoch_count):
    #to accumulate the current iteration's loss
    epoch_loss = 0
    for src, trg_input, trg in tqdm(train_loader, desc=f'Epoch {epoch + 1}/{epoch_count}', leave=False):
        src = src.transpose(0, 1).to(device)  # (seq_len, batch_size)
        trg_input = trg_input.transpose(0, 1).to(device)  # (seq_len, batch_size)
        trg = trg.transpose(0, 1).to(device)  # (seq_len, batch_size, 1)

        print("Source shape:", src.shape)
        print("Target input shape:", trg_input.shape)
        print("Target shape:", trg.shape)

        #prevent accumulation of gradients
        optimizer.zero_grad()
        #forwrd pass. source and target passed into mopdel to get predicted
        output = model(src, trg_input, teacher_forcing_ratio)
        print("Model output shape:", output.shape)
            
        # first token is start of sentence token so remove it
        output_dim = self.decoder.output_dim
        output = output[1:].reshape(-1, output_dim)
        trg = trg[1:].reshape(-1)

        # Calculate loss
        loss = criterion(output, trg)

        # Backpropagation
        loss.backward()

        # Update model parameters
        optimizer.step()

        epoch_loss += loss.item()
    avg_epoch_loss = epoch_loss / len(train_loader)
    print(f'Epoch [{epoch + 1}/{epoch_count}], Loss: {avg_epoch_loss:.4f}')
torch.save(model.state_dict(), 'English_to_Klingon.pth')

print('Now starting eval process')
 # Assuming you have evaluation data preprocessed and loaded into evaluation_loader
evaluation_dataset = TensorDataset(torch.tensor(english_test_padded, dtype=torch.long),
                                    torch.tensor(klingon_test_input, dtype=torch.long),
                                    torch.tensor(klingon_test_target, dtype=torch.long))
evaluation_loader = DataLoader(evaluation_dataset, batch_size, shuffle=False)

model.eval()  # Set the model to evaluation mode
eval_loss = 0

print('starting eval')
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



