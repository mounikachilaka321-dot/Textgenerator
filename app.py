# app.py

import streamlit as st
from transformers import pipeline

# -----------------------------------
# Page Configuration
# -----------------------------------

st.set_page_config(
    page_title="AI NLP App",
    page_icon="🤖",
    layout="centered"
)

st.title("🤖 AI NLP Demo App")

# -----------------------------------
# Sentiment Analysis Section
# -----------------------------------

st.header("Sentiment Analysis")

samples = [
    "I really enjoyed the movie.",
    "The customer service was excellent.",
    "The app crashes frequently.",
    "I am disappointed with the product.",
    "The meeting starts at 10 AM.",
    "I bought a new notebook yesterday."
]

# Load sentiment model
emotion = pipeline(
    "sentiment-analysis",
    model="tabularisai/robust-sentiment-analysis"
)

if st.button("Analyze Sentiments"):

    for text in samples:

        result = emotion(text)

        label = result[0]['label']
        score = result[0]['score']

        st.write(f"### Sentence")
        st.write(text)

        st.write(f"**Sentiment:** {label}")
        st.write(f"**Confidence Score:** {score:.4f}")

        st.write("---")

# -----------------------------------
# Text Generation Section
# -----------------------------------

st.header("Text Generation")

prompt = st.text_input(
    "Enter your prompt",
    "Artificial Intelligence will"
)

# Load text generation model
generator = pipeline("text-generation")

if st.button("Generate Text"):

    output = generator(
        prompt,
        max_length=50,
        num_return_sequences=1
    )

    generated_text = output[0]['generated_text']

    st.write("### Generated Output")
    st.write(generated_text)