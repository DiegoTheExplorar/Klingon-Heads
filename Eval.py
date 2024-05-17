import torch
from torch.utils.data import TensorDataset, DataLoader
from tqdm import tqdm
from DataPreprocessing import preprocess

#getting processed data
(english_tokenizer, klingon_tokenizer, max_english_length, max_klingon_length,
    english_train_padded, klingon_train_input, klingon_train_target,
    english_test_padded, klingon_test_input, klingon_test_target) = preprocess()
batch_size = 32
output_dim = len(klingon_tokenizer.word_index) + 1  # Add 1 for the padding token
# Define your criterion for calculating loss
criterion = torch.nn.CrossEntropyLoss()

# Define your device (e.g., CPU or GPU)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')  # Use GPU if available, otherwise use CPU

# Load the trained model
model = torch.load('seq2seq_model.pth')  # Assuming 'seq2seq_model.pth' is your saved model file

# Assuming you have evaluation data preprocessed and loaded into evaluation_loader
evaluation_dataset = TensorDataset(torch.tensor(english_test_padded, dtype=torch.long),
                                    torch.tensor(klingon_test_input, dtype=torch.long),
                                    torch.tensor(klingon_test_target, dtype=torch.long))
evaluation_loader = DataLoader(evaluation_dataset, batch_size, shuffle=False)

model.eval()  # Set the model to evaluation mode
eval_loss = 0

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
