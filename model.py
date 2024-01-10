import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, LSTM, Dense
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.layers import Embedding
import numpy as np
import pandas as pd

# Load data
data = pd.read_csv('data.csv')
input_texts = data['input'].tolist()
target_texts = ['\t' + text + '\n' for text in data['response'].tolist()]

# Tokenization
tokenizer = Tokenizer(char_level=True)
tokenizer.fit_on_texts(input_texts + target_texts)
num_tokens = len(tokenizer.word_index) + 1

# Convert texts to sequences
input_sequences = tokenizer.texts_to_sequences(input_texts)
target_sequences = tokenizer.texts_to_sequences(target_texts)

# Padding
max_sequence_length = max(max(len(seq) for seq in input_sequences), max(len(seq) for seq in target_sequences))
encoder_input_data = pad_sequences(input_sequences, maxlen=max_sequence_length, padding='post')
decoder_input_data = pad_sequences(target_sequences, maxlen=max_sequence_length, padding='post')


# Prepare decoder target data
decoder_target_data = np.zeros((len(target_sequences), max_sequence_length, num_tokens), dtype='float32')
for i, target_sequence in enumerate(target_sequences):
    for t, token_index in enumerate(target_sequence):
        if t > 0:
            decoder_target_data[i, t - 1, token_index] = 1.0

latent_dim = 256
embedding_dim = 256

# Encoder
encoder_inputs = Input(shape=(None,))
encoder_embedding = Embedding(input_dim=num_tokens, output_dim=embedding_dim)(encoder_inputs)
encoder_lstm = LSTM(latent_dim, return_state=True)
encoder_outputs, state_h, state_c = encoder_lstm(encoder_embedding)
encoder_states = [state_h, state_c]

# Decoder
decoder_inputs = Input(shape=(None,))
decoder_embedding_layer = Embedding(input_dim=num_tokens, output_dim=embedding_dim)
decoder_embedding = decoder_embedding_layer(decoder_inputs)
decoder_lstm = LSTM(latent_dim, return_sequences=True, return_state=True)
decoder_outputs, _, _ = decoder_lstm(decoder_embedding, initial_state=encoder_states)
decoder_dense = Dense(num_tokens, activation='softmax')
decoder_outputs = decoder_dense(decoder_outputs)
print(decoder_outputs)

# Define the model
model = Model([encoder_inputs, decoder_inputs], decoder_outputs)

model.compile(optimizer='rmsprop', loss='categorical_crossentropy')
model.fit([encoder_input_data, decoder_input_data], decoder_target_data,
          batch_size=64,
          epochs=100,
          validation_split=0.2)

# Encoder Inference Model
encoder_model = Model(encoder_inputs, encoder_states)

# Inputs for the decoder model at inference time
decoder_state_input_h = Input(shape=(latent_dim,))
decoder_state_input_c = Input(shape=(latent_dim,))
decoder_states_inputs = [decoder_state_input_h, decoder_state_input_c]

decoder_inference_inputs = Input(shape=(1,))  # Single timestep input
decoder_inference_embedding = decoder_embedding_layer(decoder_inference_inputs)
decoder_inference_outputs, state_h_inf, state_c_inf = decoder_lstm(
    decoder_inference_embedding, initial_state=decoder_states_inputs)
decoder_states_inf = [state_h_inf, state_c_inf]
decoder_inference_outputs = decoder_dense(decoder_inference_outputs)

decoder_model = Model(
    [decoder_inference_inputs] + decoder_states_inputs,
    [decoder_inference_outputs] + decoder_states_inf)

def decode_sequence(input_seq):
    states_value = encoder_model.predict(input_seq)

    decoded_sentence = ''
    stop_condition = False
    target_seq = np.array([[tokenizer.word_index['\t']]])  # Start token as integer ID

    while not stop_condition:
        output_tokens, h, c = decoder_model.predict([target_seq] + states_value)
        sampled_token_index = np.argmax(output_tokens[0, -1, :])
        sampled_char = tokenizer.index_word.get(sampled_token_index, '')
        decoded_sentence += sampled_char

        if sampled_char == '\n' or len(decoded_sentence) > max_sequence_length:
            stop_condition = True

        target_seq = np.array([[sampled_token_index]])  # Update with the next token ID
        states_value = [h, c]

    return decoded_sentence
    
def process_new_input(text):
    sequence = tokenizer.texts_to_sequences([text])
    padded_sequence = pad_sequences(sequence, maxlen=max_sequence_length, padding='post')
    return padded_sequence

while True:
    input_text = input('Enter your question: ')
    if input_text == 'quit':
        break
    
    input_seq = process_new_input(input_text)
    decoded_sentence = decode_sequence(input_seq)
    print('Bot reply:', decoded_sentence)
