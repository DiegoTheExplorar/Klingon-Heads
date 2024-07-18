import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import tensorflow as tf

def preprocess():
    # Load dataset
    data = pd.read_csv('./backend/English_To_Klingon.csv')


    # Append <BOS> and <EOS> tags to the Klingon sentences
    data['klingon'] = data['klingon'].apply(lambda x: '<BOS> ' + x + ' <EOS>')

    # Separate the sentences
    english_sentences = data['english'].values
    klingon_sentences = data['klingon'].values

    # Split data into training and testing sets. An 80 - 20 split is used here
    english_train, english_test, klingon_train, klingon_test = train_test_split(
        english_sentences, klingon_sentences, test_size=0.2, random_state=42)

    # Initialize tokenizers with specified vocabulary size
    english_tokenizer = tf.keras.preprocessing.text.Tokenizer(num_words=5000, oov_token='<UNK>')
    klingon_tokenizer = tf.keras.preprocessing.text.Tokenizer(num_words=5000, oov_token='<UNK>')

    # Fit tokenizers on training data
    english_tokenizer.fit_on_texts(english_train)
    klingon_tokenizer.fit_on_texts(klingon_train)

    # Tokenize the sentences
    english_train_sequences = english_tokenizer.texts_to_sequences(english_train)
    klingon_train_sequences = klingon_tokenizer.texts_to_sequences(klingon_train)
    english_test_sequences = english_tokenizer.texts_to_sequences(english_test)
    klingon_test_sequences = klingon_tokenizer.texts_to_sequences(klingon_test)

    # Padding sequences to a fixed length
    english_train_padded = tf.keras.preprocessing.sequence.pad_sequences(english_train_sequences, maxlen=50, padding='post')
    klingon_train_padded = tf.keras.preprocessing.sequence.pad_sequences(klingon_train_sequences, maxlen=50, padding='post')
    english_test_padded = tf.keras.preprocessing.sequence.pad_sequences(english_test_sequences, maxlen=50, padding='post')
    klingon_test_padded = tf.keras.preprocessing.sequence.pad_sequences(klingon_test_sequences, maxlen=50, padding='post')

    # Prepare target data for training
    klingon_train_input = klingon_train_padded[:, :-1] # The decoder input, which is the Klingon sentence shifted by one position to the right for training data.
    klingon_train_target = klingon_train_padded[:, 1:] # The target output, which is the same sentence shifted by one position to the left for training data.
    klingon_train_target = np.expand_dims(klingon_train_target, -1)

    # Prepare target data for testing
    klingon_test_input = klingon_test_padded[:, :-1] # The decoder input for testing data.
    klingon_test_target = klingon_test_padded[:, 1:] # The target output for testing data.
    klingon_test_target = np.expand_dims(klingon_test_target, -1)

    return (english_tokenizer, klingon_tokenizer, 50, # max_length
            english_train_padded, klingon_train_input, klingon_train_target,
            english_test_padded, klingon_test_input, klingon_test_target)

