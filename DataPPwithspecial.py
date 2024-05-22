import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import tensorflow as tf

def preprocess_with_special_tokens():
    # Load dataset
    data = pd.read_csv('English_To_Klingon.csv')

    # Separate the sentences
    english_sentences = data['english'].values
    klingon_sentences = data['klingon'].values

    # Split data into training and testing sets
    english_train, english_test, klingon_train, klingon_test = train_test_split(
        english_sentences, klingon_sentences, test_size=0.2, random_state=42)

    # Initialize tokenizers
    english_tokenizer = tf.keras.preprocessing.text.Tokenizer()
    klingon_tokenizer = tf.keras.preprocessing.text.Tokenizer()

    # Fit tokenizers on texts
    english_tokenizer.fit_on_texts(english_train)
    klingon_tokenizer.fit_on_texts(klingon_train)

    # Manually add <sos> and <eos> tokens
    english_tokenizer.word_index['<sos>'] = len(english_tokenizer.word_index) + 1
    english_tokenizer.word_index['<eos>'] = len(english_tokenizer.word_index) + 2
    klingon_tokenizer.word_index['<sos>'] = len(klingon_tokenizer.word_index) + 1
    klingon_tokenizer.word_index['<eos>'] = len(klingon_tokenizer.word_index) + 2

    # Convert texts to sequences
    english_train_sequences = english_tokenizer.texts_to_sequences(english_train)
    klingon_train_sequences = [[klingon_tokenizer.word_index['<sos>']] + klingon_tokenizer.texts_to_sequences([sent])[0] + [klingon_tokenizer.word_index['<eos>']] for sent in klingon_train]
    english_test_sequences = english_tokenizer.texts_to_sequences(english_test)
    klingon_test_sequences = [[klingon_tokenizer.word_index['<sos>']] + klingon_tokenizer.texts_to_sequences([sent])[0] + [klingon_tokenizer.word_index['<eos>']] for sent in klingon_test]

    # Padding sequences
    max_english_length = max([len(seq) for seq in english_train_sequences])
    max_klingon_length = max([len(seq) for seq in klingon_train_sequences])

    english_train_padded = tf.keras.preprocessing.sequence.pad_sequences(english_train_sequences, maxlen=max_english_length, padding='post')
    klingon_train_padded = tf.keras.preprocessing.sequence.pad_sequences(klingon_train_sequences, maxlen=max_klingon_length, padding='post')
    english_test_padded = tf.keras.preprocessing.sequence.pad_sequences(english_test_sequences, maxlen=max_english_length, padding='post')
    klingon_test_padded = tf.keras.preprocessing.sequence.pad_sequences(klingon_test_sequences, maxlen=max_klingon_length, padding='post')

    # Prepare target data for training and testing
    klingon_train_input = klingon_train_padded[:, :-1]
    klingon_train_target = klingon_train_padded[:, 1:]
    klingon_test_input = klingon_test_padded[:, :-1]
    klingon_test_target = klingon_test_padded[:, 1:]

    # Reshape target data as needed
    klingon_train_target = np.expand_dims(klingon_train_target, -1)
    klingon_test_target = np.expand_dims(klingon_test_target, -1)

    return (english_tokenizer, klingon_tokenizer, max_english_length, max_klingon_length,
            english_train_padded, klingon_train_input, klingon_train_target,
            english_test_padded, klingon_test_input, klingon_test_target)
