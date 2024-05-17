import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

# Load dataset
data = pd.read_csv('English_To_Klingon.csv')

# separate the sentencesS
english_sentences = data['english'].values
klingon_sentences = data['klingon'].values

# split data into training and testing tests. An 80 - 20 split is used here
english_train, english_test, klingon_train, klingon_test = train_test_split(english_sentences, klingon_sentences, test_size=0.2, random_state=42)

# Tokenize the sentences
english_tokenizer = Tokenizer()
klingon_tokenizer = Tokenizer()

english_tokenizer.fit_on_texts(english_train)
klingon_tokenizer.fit_on_texts(klingon_train)

english_train_sequences = english_tokenizer.texts_to_sequences(english_train)
klingon_train_sequences = klingon_tokenizer.texts_to_sequences(klingon_train)
english_test_sequences = english_tokenizer.texts_to_sequences(english_test)
klingon_test_sequences = klingon_tokenizer.texts_to_sequences(klingon_test)

# Padding sequences
max_english_length = max([len(seq) for seq in english_train_sequences])
max_klingon_length = max([len(seq) for seq in klingon_train_sequences])

english_train_padded = pad_sequences(english_train_sequences, maxlen=max_english_length, padding='post')
klingon_train_padded = pad_sequences(klingon_train_sequences, maxlen=max_klingon_length, padding='post')
english_test_padded = pad_sequences(english_test_sequences, maxlen=max_english_length, padding='post')
klingon_test_padded = pad_sequences(klingon_test_sequences, maxlen=max_klingon_length, padding='post')
