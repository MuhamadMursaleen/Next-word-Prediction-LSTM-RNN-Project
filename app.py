import pickle
import numpy as np
import streamlit as st 
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import load_model

model = load_model('next_word_prediction_model.h5')

# Compile the model (if necessary)
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])


# Load the tokenizer``

with open('tokenizer.pickle', 'rb') as file:
    tokenizer = pickle.load(file)

# Function to generate text
def generate_text(model, tokenizer, text, max_sequence_len):
    token_list = tokenizer.texts_to_sequences([text])[0]
    if len(token_list) >= max_sequence_len:
        token_list = token_list[-(max_sequence_len-1):]
    token_list = pad_sequences([token_list], maxlen=max_sequence_len-1, padding='pre')
    Predicted = model.predict(token_list, verbose=0)
    predected_word = np.argmax(Predicted, axis=1)
    for word, index in tokenizer.word_index.items():
        if index == predected_word:
            output_word = word
            return output_word
        
    return None

# Streamlit code
st.title('Next Word Prediction App with LSTM RNN')
text = st.text_input('Enter text:', 'The king')
if st.button('Predict next word'):
    max_sequence_len = model.input_shape[1]+1
    output = generate_text(model, tokenizer, text, max_sequence_len)
    st.write(f'Next word: {output}')