import torch
import tensorflow as tf
import gradio as gr
import re

from Seq2SeqModel import Seq2SeqModel 
from DataPPwithspecial import preprocess 
from Decoder import Decoder
from Encoder import Encoder
# Model parameters
n_layers = 2
emb_dim = 256
hid_dim = 512
dropout = 0.5
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')  # Use GPU if available, otherwise use CPU

# Load preprocessed data and model parameters
(english_tokenizer, klingon_tokenizer, max_english_length,
    _, _, _, _, _, _) = preprocess()  # We don't need training data for inference
input_dim = len(english_tokenizer.word_index) + 1  # Add 1 for the padding token
output_dim = len(klingon_tokenizer.word_index) + 1  # Add 1 for the padding token

# Initialize encoder and decoder
encoder = Encoder(input_dim, emb_dim, hid_dim, n_layers, dropout).to(device)
decoder = Decoder(output_dim, emb_dim, hid_dim, n_layers, dropout).to(device)

# Initialize the Seq2SeqModel
model = Seq2SeqModel(encoder, decoder, device).to(device)

# Load the saved model
model.load_state_dict(torch.load('./backend/English_to_Klingon.pth'))
model.eval()  # Set the model to evaluation mode

#tokenize the English input
def preprocess_sentence(sentence, tokenizer, max_length):
    # Tokenize the sentence
    tokenized_sentence = tokenizer.texts_to_sequences([sentence])
    # Pad the sequence
    padded_sentence = tf.keras.preprocessing.sequence.pad_sequences(tokenized_sentence, maxlen=max_length, padding='post')
    return torch.tensor(padded_sentence, dtype=torch.long).to(device)

# Translation function for Gradio
def translate_english_to_klingon(english_sentence):
    # Preprocess the input English sentence
    input_sentence = preprocess_sentence(english_sentence, english_tokenizer, max_english_length)

    # Remove the extra dimension added by unsqueeze(1)
    input_sentence = input_sentence.squeeze(0)

    # Perform inference
    with torch.no_grad():
        # Pass input as both input and target with teacher forcing ratio 0
        output = model(input_sentence.unsqueeze(1), input_sentence.unsqueeze(1), 0)

    # Convert output indices to Klingon words
    output_indices = torch.argmax(output, dim=-1).squeeze().tolist()
    klingon_sentence = ' '.join([klingon_tokenizer.index_word[idx] for idx in output_indices if idx != 0])  # Remove padding token
    klingon_sentence = re.sub(r'\beos\b', '', klingon_sentence).strip()
    return klingon_sentence


# Create Gradio interface
iface = gr.Interface(fn=translate_english_to_klingon, inputs="text", outputs="text", title="English to Klingon Translation")
iface.launch()
"""
english_sentence = 'hello. nice to meet you'
print('english sentence',english_sentence)
print('translated',translate_english_to_klingon(english_sentence))
"""
