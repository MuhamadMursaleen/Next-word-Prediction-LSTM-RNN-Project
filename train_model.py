## Data Collection
import nltk
nltk.download('gutenberg')  
from nltk.corpus import gutenberg
import pandas as pd 
import numpy as np
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential  
from tensorflow.keras.layers import Embedding, LSTM, Dense, Dropout
import tensorflow as tf
import pickle

## load the data
data = gutenberg.raw('shakespeare-hamlet.txt')

# save a file
with open('hamlet.txt', 'w') as f:
    f.write(data) 

## load the data

with open('hamlet.txt', 'r') as file:
    text = file.read().lower()

## Tokenization 
tokenizer = Tokenizer() 
tokenizer.fit_on_texts([text])      
total_words = len(tokenizer.word_index) + 1

## Create input sequences

input_sequences = []
for line in text.split('\n'):
    token_list = tokenizer.texts_to_sequences([line])[0]
    for i in range(1, len(token_list)):
        n_gram_sequence = token_list[:i+1]
        input_sequences.append(n_gram_sequence)

## pad sequences

max_sequence_length = max([len(x) for x in input_sequences])
input_sequences = np.array(pad_sequences(input_sequences, maxlen=max_sequence_length, padding='pre'))

## create predictors and label

x,y = input_sequences[:,:-1],input_sequences[:,-1]
y = tf.keras.utils.to_categorical(y, num_classes=total_words)

## Split the data
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2)

## Define the model

Model = Sequential()
Model.add(Embedding(total_words, 100, input_length=max_sequence_length-1))
Model.add(LSTM(150, return_sequences = True))
Model.add(Dropout(0.2))
Model.add(LSTM(100))
Model.add(Dense(total_words, activation='softmax'))
Model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
Model.summary()

## Train the model

history = Model.fit(x_train, y_train, epochs=100,validation_data=(x_test,y_test), verbose=1)

# Save the model
Model.save('next_word_prediction_model.h5')

# Save the tokenizer

with open('tokenizer.pickle', 'wb') as handle:
    pickle.dump(tokenizer, handle, protocol=pickle.HIGHEST_PROTOCOL)

