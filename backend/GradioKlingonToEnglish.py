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
(klingon_tokenizer, english_tokenizer, max_klingon_length,
    _, _, _, _, _, _) = preprocess()  # We don't need training data for inference
input_dim = len(klingon_tokenizer.word_index) + 1  # Add 1 for the padding token
output_dim = len(english_tokenizer.word_index) + 1  # Add 1 for the padding token

# Initialize encoder and decoder
encoder = Encoder(input_dim, emb_dim, hid_dim, n_layers, dropout).to(device)
decoder = Decoder(output_dim, emb_dim, hid_dim, n_layers, dropout).to(device)

# Initialize the Seq2SeqModel
model = Seq2SeqModel(encoder, decoder, device).to(device)

# Load the saved model, mapping it to the CPU if necessary
model.load_state_dict(torch.load('./backend/Klingon_to_English.pth', map_location=torch.device('cpu')))
model.eval()  # Set the model to evaluation mode

# Tokenize the Klingon input
def preprocess_sentence(sentence, tokenizer, max_length):
    # Tokenize the sentence
    tokenized_sentence = tokenizer.texts_to_sequences([sentence])
    # Pad the sequence
    padded_sentence = tf.keras.preprocessing.sequence.pad_sequences(tokenized_sentence, maxlen=max_length, padding='post')
    return torch.tensor(padded_sentence, dtype=torch.long).to(device)

# Translation function for Gradio
def translate_klingon_to_english(klingon_sentence):
    # Preprocess the input Klingon sentence
    input_sentence = preprocess_sentence(klingon_sentence, klingon_tokenizer, max_klingon_length)

    # Remove the extra dimension added by unsqueeze(1)
    input_sentence = input_sentence.squeeze(0)

    # Perform inference
    with torch.no_grad():
        # Pass input as both input and target with teacher forcing ratio 0
        output = model(input_sentence.unsqueeze(1), input_sentence.unsqueeze(1), 0)

    # Convert output indices to English words
    output_indices = torch.argmax(output, dim=-1).squeeze().tolist()
    english_sentence = ' '.join([english_tokenizer.index_word[idx] for idx in output_indices if idx != 0])  # Remove padding token
    # Regex to remove eos
    english_sentence = re.sub(r'\beos\b', '', english_sentence).strip()
    if english_sentence == "":
        english_sentence = 'sorry model sucks'
    return english_sentence

# Create Gradio interface
examples = [
    ["nuqneH"],
    ["tlhIngan Hol Dajatlh'a'?"],
    ["jIyajbe'"],
    ["Heghlu'meH QaQ jajvam"],
    ["Hoch vor Dar"]
]

iface = gr.Interface(
    fn=translate_klingon_to_english,
    inputs=gr.Textbox(label="Klingon Phrase", lines=2, placeholder="Enter Klingon text here..."),
    outputs=gr.Textbox(label="English Translation", lines=2),
    title="Klingon to English Translation",
    description="Enter text in Klingon and get its translation in English. This translator helps you convert everyday Klingon phrases into English. Try one of the example sentences to see how it works!",
    examples=examples,
    theme="default"
)

iface.launch()
