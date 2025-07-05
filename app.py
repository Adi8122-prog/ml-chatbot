import streamlit as st
import random
import json
import pickle

# Load model
with open("model.pkl", "rb") as f:
    model = pickle.load(f)

# Load intents
with open("intents.json") as f:
    intents = json.load(f)

# Chatbot response function
def get_response(user_input):
    tag = model.predict([user_input])[0]
    for intent in intents["intents"]:
        if intent["tag"] == tag:
            return random.choice(intent["responses"])
    return "Sorry, I didn't understand that."

# Streamlit UI
st.set_page_config(page_title="ChatBot", page_icon="🤖")
st.title("🤖 Simple ChatBot (ML-Based)")
st.markdown("Ask me anything like 'Hi', 'What's your name?', 'What do you do?'")

# User input
user_input = st.text_input("You:", "")

if user_input:
    response = get_response(user_input)
    st.markdown(f"**Bot:** {response}")
