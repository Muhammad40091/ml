import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
import streamlit as st
import random

st.title("Sarcasm Text Detection")

vocab_size = 10000
embedding_dim = 16
max_length = 100
trunc_type = 'post'
padding_type = 'post'
oov_tok = "<OOV>"
epochs = 5

# Read csv file using pandas library
sarcasm_df = pd.read_csv("Data.csv")

# Split them into two columns
input_seq = sarcasm_df['headlines']
target_seq = sarcasm_df['target']

tokenizer = Tokenizer(num_words=vocab_size, oov_token=oov_tok)
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

text = st.text_input("Enter Text:", placeholder=random.choice(sarcastic_sentences))

color_palette = {
    'Frustrated': '#FF0000',
    'Angry': '#FFA500',
    'Sad': '#FFFF00',
    'Content': '#008000',
    'Happy': '#0000FF',
    'Excited': '#800080'
}

selected_color = 'Frustrated'

selected_color = st.selectbox("How are you feeling?", list(color_palette.keys()))

# Set the color of col3 based on the selected color
col3 = st.color_picker("Color ", color_palette[selected_color])

def handle_input_text():
    if len(text) != 0:
        input_sentences = tokenizer.texts_to_sequences([text])
        input_padded_sentences = pad_sequences(input_sentences, maxlen=max_length, padding=padding_type,
                                               truncating=trunc_type)
        probs = model.predict(input_padded_sentences)
        preds = np.round(probs).astype(int)
        if preds == 1:
            return "Sarcastic"
        else:
            return "Not Sarcastic"
    else:
        return ""

if st.button("Detect🔍"):
    result = handle_input_text()
    st.write("Text: ", text)
    st.write("Prediction: ", result)
