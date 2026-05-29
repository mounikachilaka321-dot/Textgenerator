import streamlit as st
from transformers import pipeline

st.title("🤖 AI NLP Demo App")

# -----------------------------------
# Sentiment Analysis
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

@st.cache_resource
def load_sentiment_model():
    return pipeline(
        "sentiment-analysis",
        model="distilbert-base-uncased-finetuned-sst-2-english"
    )

emotion = load_sentiment_model()

if st.button("Analyze Sentiment"):

    for text in samples:

        result = emotion(text)

        st.write(f"### {text}")
        st.write(result)

# -----------------------------------
# Text Generation
# -----------------------------------

st.header("Text Generation")

@st.cache_resource
def load_generator():
    return pipeline(
        "text-generation",
        model="distilgpt2"
    )

generator = load_generator()

prompt = st.text_input(
    "Enter Prompt",
    "Artificial Intelligence will"
)

if st.button("Generate Text"):

    output = generator(
        prompt,
        max_length=50,
        num_return_sequences=1
    )

    st.write(output[0]["generated_text"])
