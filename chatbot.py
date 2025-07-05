import random
import json
import numpy as np
import pickle
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import make_pipeline

# Load intents
with open("intents.json") as file:
    data = json.load(file)

# Prepare training data
X = []
y = []
for intent in data["intents"]:
    for pattern in intent["patterns"]:
        X.append(pattern)
        y.append(intent["tag"])

# Train the model
model = make_pipeline(TfidfVectorizer(), MultinomialNB())
model.fit(X, y)

# Save the model and vectorizer
with open("model.pkl", "wb") as f:
    pickle.dump(model, f)

# Chat function
def chat():
    print("🤖 ChatBot is ready! Type 'quit' to exit.")
    while True:
        user_input = input("You: ")
        if user_input.lower() == "quit":
            print("Bot: Bye! Have a nice day!")
            break
        pred = model.predict([user_input])[0]
        for intent in data["intents"]:
            if intent["tag"] == pred:
                print("Bot:", random.choice(intent["responses"]))
                break

if __name__ == "__main__":
    chat()
