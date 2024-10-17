# Next Word Prediction LSTM RNN Project
This project focuses on predicting the next word in a sequence using an LSTM RNN model. The model is trained on the text of Shakespeare's "Hamlet".

## Files

- **hamlet.txt**: The dataset used for training the model.
- **next_word_prediction_model.h5**: The trained LSTM RNN model.
- **requirements.txt**: A list of required libraries to run the project.
- **tokenizer.pickle**: A pickle file containing the tokenizer used in the model.
- **train_model.py**: A script to train the model.
- **train_model.ipynb**: A Jupyter notebook to train the model interactively.
- **app.py**: A Streamlit app to predict the next word.

## Setup

1. Clone the repository.
2. Install the required libraries using:
    ```bash
    pip install -r requirements.txt
    ```

## Training the Model

You can train the model using the provided script or Jupyter notebook:

- **Using the script**:
    ```bash
    python train_model.py
    ```

- **Using the Jupyter notebook**:
    Open `train_model.ipynb` in Jupyter Notebook and run the cells.

## Usage
To run the Streamlit app, use the following command:

```bash
streamlit run app.py
```

This will start a local web server, and you can interact with the app through your web browser. The app will prompt you to enter a sequence of words, and it will predict the next word using the trained model.
After training, you can use the `next_word_prediction_model.h5` and `tokenizer.pickle` to predict the next word in a sequence.

## License

This project is licensed under the MIT License.