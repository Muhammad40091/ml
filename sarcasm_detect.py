# Import some of the most important libary for using this notebook
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from sklearn.model_selection import train_test_split
import streamlit as st
import random

st.title("Sarcasm Text Detection ")

vocab_size = 10000
embedding_dim = 16
max_length = 100
trunc_type = 'post'
padding_type = 'post'
oov_tok = "<OOV>"
training_size = 20000
epochs = 5

# Read csv file for using pandas libary
sarcasm_df = pd.read_csv("Data.csv")

# Split them two column
input_seq = sarcasm_df['headlines']
target_seq = sarcasm_df['target']

tokenizer = Tokenizer(num_words=vocab_size, oov_token=oov_tok, )
tokenizer.fit_on_texts(input_seq)
word_index = tokenizer.word_index
model = tf.keras.models.load_model('sarcasm_detect.h5')

sarcastic_sentences = [
    "Wow, I love waking up early on weekends!",
    "You're doing a great job breaking things.",
    "Oh, because making things complicated is always the best approach.",
    "Sure, let's prioritize adding more bugs to the codebase.",
    "Oh, I absolutely love it when my computer crashes.",
    "Yes, please make the font size smaller. That's what I need.",
    "I'm thrilled to have another meeting scheduled for this afternoon.",
    "Oh, fantastic! The internet is down again.",
    "Brilliant idea! Let's reinvent the wheel.",
    "You know what's better than efficient code? Slow code."
]

text = st.text_input("Enter Text: ", placeholder=random.choice(sarcastic_sentences))

col2, col3 = st.columns(2)


def handle_input_text():
    if len(text) != 0:
        input_sentences = tokenizer.texts_to_sequences([text])
        input_padded_sentences = pad_sequences(input_sentences, maxlen=max_length, padding=padding_type,
                                               truncating=trunc_type)
        probs = model.predict(input_padded_sentences)
        preds = f"{int(np.round(probs))}"
        if preds == '1':
            col3.write("Sarcastic")
        else:
            col3.write("Not Sarcastic")
    else:
        col3.write("")


col2.button("Detect🔍", on_click=handle_input_text)