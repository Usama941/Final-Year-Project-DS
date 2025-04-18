import streamlit as st
import pickle
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

import base64

def add_bg_image(image_path):
    with open(image_path, "rb") as img_file:
        encoded = base64.b64encode(img_file.read()).decode()
    page_bg_img = f"""
    <style>
    .stApp {{
        background-image: url("data:image/png;base64,{encoded}");
        background-size: cover;
        background-repeat: no-repeat;
        background-attachment: fixed;
    }}
    </style>
    """
    st.markdown(page_bg_img, unsafe_allow_html=True)


# Load components
model = load_model("intent_model.keras")

with open("tokenizer.pkl", "rb") as f:
    tokenizer = pickle.load(f)

with open("label_encoder.pkl", "rb") as f:
    label_encoder = pickle.load(f)

max_len = model.input_shape[1]
sentiment_analyzer = SentimentIntensityAnalyzer()

# Define intent + sentiment prediction
def predict_intent_and_sentiment(user_input):
    # Preprocess text
    text = user_input.lower().strip()
    seq = tokenizer.texts_to_sequences([text])
    padded = pad_sequences(seq, maxlen=max_len, padding='post')
    
    # Predict intent
    intent_probs = model.predict(padded)
    pred_int = intent_probs.argmax(axis=1)[0]
    intent = label_encoder.inverse_transform([pred_int])[0]

    # Predict sentiment
    sentiment_score = sentiment_analyzer.polarity_scores(user_input)['compound']
    if sentiment_score >= 0.05:
        sentiment = "positive"
    elif sentiment_score <= -0.05:
        sentiment = "negative"
    else:
        sentiment = "neutral"

    return intent, sentiment

# Generate dynamic response
def generate_response(intent, sentiment):
    if sentiment == "negative":
        tone = "I understand, let me help with that. "
    elif sentiment == "positive":
        tone = "Great! Here's what I found. "
    else:
        tone = "Okay, working on it. "

    response = f"{tone}Detected intent: **{intent}** with a **{sentiment}** tone."
    return response

# 🌐 Streamlit UI

st.set_page_config(page_title="Intent Chatbot", page_icon="🤖")
add_bg_image("robot-blank-sign.jpg") 
# 👈 Call happens AFTER the function is defined
st.markdown("""
    <h1 style='text-align: center; color: #3498db; font-size: 42px; font-family: Arial;
  text-shadow: 2px 2px #000000;'>
        ✈️ Airline Intent Chatbot with Sentiment
    </h1>
""", unsafe_allow_html=True)


user_input = st.text_input("Ask me something about flights:")

if user_input:
    intent, sentiment = predict_intent_and_sentiment(user_input)
    bot_response = generate_response(intent, sentiment)
    st.markdown(f"### 🤖 Bot:\n{bot_response}")
