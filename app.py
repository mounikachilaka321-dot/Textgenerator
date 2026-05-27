import streamlit as st
from transformers import pipeline

# Page title
st.title("AI NLP Demo App")

# -------------------------------
# Sentiment Analysis
# -------------------------------

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
emotion = pipeline("sentiment-analysis")

# Analyze button
if st.button("Analyze Sentiments"):

    for text in samples:
        result = emotion(text)

        st.write(f"**Sentence:** {text}")
        st.write(f"**Result:** {result}")
        st.write("---")

# -------------------------------
# Text Generation
# -------------------------------

st.header("Text Generation")

prompt = st.text_input(
    "Enter a prompt",
    "Artificial Intelligence will"
)

generator = pipeline("text-generation")

if st.button("Generate Text"):

    output = generator(
        prompt,
        max_length=50,
        num_return_sequences=1
    )

    st.write("### Generated Output")
    st.write(output[0]['generated_text'])