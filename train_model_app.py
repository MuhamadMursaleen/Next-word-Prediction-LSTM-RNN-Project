## Data Collection
import nltk
import pandas as pd 
import numpy as np
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential  
from tensorflow.keras.layers import Embedding, LSTM, Dense, Dropout
import tensorflow as tf
import pickle
import streamlit as st
from tensorflow.keras.models import load_model

# Load the default model
default_model = load_model('next_word_prediction_model.h5')

# Load the default tokenizer
with open('tokenizer.pickle', 'rb') as file:
    default_tokenizer = pickle.load(file)

# Streamlit app
st.title('Next Word Prediction LSTM RNN')

# Divide the view into columns
left_column, right_column = st.columns(2)

# Left column for training
with left_column:
    st.header("Training")
    
    # Text area for user input
    user_input = st.text_area("Enter the text to train the model:", height=200)

    # Text input for epochs value
    epochs_text = st.text_input('Enter the number of epochs (max 1000):', value='10')

    # Ensure the epochs value does not exceed 1000
    try:
        epochs = min(int(epochs_text), 1000)
    except ValueError:
        epochs = 10

    # Text input for model file name
    model_file_name = st.text_input('Enter the model file name (without extension):', 'trained_model')

    # Initialize session state for training and file contents
    if 'training' not in st.session_state:
        st.session_state.training = False
    if 'model_file' not in st.session_state:
        st.session_state.model_file = None
    if 'tokenizer_file' not in st.session_state:
        st.session_state.tokenizer_file = None
    if 'trained_model' not in st.session_state:
        st.session_state.trained_model = default_model
    if 'trained_tokenizer' not in st.session_state:
        st.session_state.trained_tokenizer = default_tokenizer

    def train_model(user_input, epochs, model_file_name):
        if user_input:
            text = user_input.lower()

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
            max_sequence_length = max([len(x) for x in input_sequences]) if input_sequences else 0
            input_sequences = np.array(pad_sequences(input_sequences, maxlen=max_sequence_length, padding='pre'))

            ## create predictors and label
            x, y = input_sequences[:,:-1], input_sequences[:,-1]
            y = tf.keras.utils.to_categorical(y, num_classes=total_words)

            ## Split the data
            x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2)

            ## Define the model
            Model = Sequential()
            Model.add(Embedding(total_words, 100, input_length=max_sequence_length-1))
            Model.add(LSTM(150, return_sequences=True))
            Model.add(Dropout(0.2))
            Model.add(LSTM(100))
            Model.add(Dense(total_words, activation='softmax'))
            Model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
            
            # Streamlit progress bar and status text
            with st.container():
                progress_bar = st.progress(0)
                status_text = st.empty()

            class StreamlitCallback(tf.keras.callbacks.Callback):
                def on_epoch_end(self, epoch, logs=None):
                    if not st.session_state.training:
                        self.model.stop_training = True
                    progress = (epoch + 1) / epochs
                    progress_bar.progress(progress)
                    status_text.text(
                        f"Epoch {epoch + 1}/{epochs} - {int(progress * 100)}% complete \n"
                        f"loss: {logs['loss']:.4f} - accuracy: {logs['accuracy']:.4f} \n"
                        f"val_loss: {logs['val_loss']:.4f} - val_accuracy: {logs['val_accuracy']:.4f}"
                    )

            ## Train the model
            history = Model.fit(x_train, y_train, epochs=epochs, validation_data=(x_test, y_test), verbose=0, callbacks=[StreamlitCallback()])

            st.success('Model trained successfully!')

            # Save the model
            model_path = f"{model_file_name}.h5"
            Model.save(model_path)

            # Save the tokenizer
            tokenizer_path = f"{model_file_name}_tokenizer.pickle"
            with open(tokenizer_path, 'wb') as handle:
                pickle.dump(tokenizer, handle, protocol=pickle.HIGHEST_PROTOCOL)

            # Store file contents in session state
            with open(model_path, 'rb') as f:
                st.session_state.model_file = f.read()

            with open(tokenizer_path, 'rb') as f:
                st.session_state.tokenizer_file = f.read()

            # Update session state with the new model and tokenizer
            st.session_state.trained_model = Model
            st.session_state.trained_tokenizer = tokenizer
        else:
            st.error("Please provide some text data to train the model.")

    # Train and Stop buttons 
    
    if st.button('Stop Training'):
        st.session_state.training = False

    if st.button('Train Model'):
        st.session_state.training = True
        train_model(user_input, epochs, model_file_name)

    # Provide download buttons for the model and tokenizer if they exist in session state
    if st.session_state.model_file:
        st.download_button('Download the trained model', st.session_state.model_file, file_name=f"{model_file_name}.h5")

    if st.session_state.tokenizer_file:
        st.download_button('Download the tokenizer', st.session_state.tokenizer_file, file_name=f"{model_file_name}_tokenizer.pickle")

# Right column for prediction
with right_column:
    st.header("Prediction")
    
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

    # Text input for prediction
    text = st.text_input('Enter text:', 'The king')
    if st.button('Predict next word'):
        max_sequence_len = st.session_state.trained_model.input_shape[1] + 1
        output = generate_text(st.session_state.trained_model, st.session_state.trained_tokenizer, text, max_sequence_len)
        st.write(f'Next word: {output}')
