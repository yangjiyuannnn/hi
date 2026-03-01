import streamlit as st
import pickle
import re
import nltk
import numpy as np
import matplotlib.pyplot as plt
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

# ---------------------------
# Load Model & Vectorizer
# ---------------------------
model = pickle.load(open("final_model.pkl", "rb"))
vectorizer = pickle.load(open("tfidf_vectorizer.pkl", "rb"))

# ---------------------------
# NLTK Setup
# ---------------------------
nltk.download("stopwords", quiet=True)
nltk.download("wordnet", quiet=True)

stop_words = set(stopwords.words("english"))
lemmatizer = WordNetLemmatizer()

# ---------------------------
# Initialize Session State
# ---------------------------
if "history" not in st.session_state:
    st.session_state.history = []

if "input_text" not in st.session_state:
    st.session_state.input_text = ""

# ---------------------------
# Text Cleaning Function
# ---------------------------
def clean_text(text):
    text = text.lower()
    text = re.sub(r"http\S+", "", text)
    text = re.sub(r"[^a-zA-Z\s]", "", text)
    words = text.split()
    words = [word for word in words if word not in stop_words]
    words = [lemmatizer.lemmatize(word) for word in words]
    return " ".join(words)

# ---------------------------
# Page Layout
# ---------------------------
st.title("🚗 Car Review NLP Classification System")

st.markdown("""
### 📌 About This System
This NLP system classifies car review comments into:

- 🟢 **Positive**
- 🟠 **Negative**
- 🔴 **Toxic**

Model Used: LinearSVC  
Vectorization: TF-IDF
""")

# ---------------------------
# Example Buttons
# ---------------------------
st.subheader("Try Example Comments")

col1, col2, col3 = st.columns(3)

with col1:
    if st.button("Positive Example"):
        st.session_state.input_text = "This car is amazing and very comfortable"

with col2:
    if st.button("Negative Example"):
        st.session_state.input_text = "The maintenance cost is too high"

with col3:
    if st.button("Toxic Example"):
        st.session_state.input_text = "This brand is trash and completely useless"

# ---------------------------
# User Input
# ---------------------------
st.subheader("Enter Your Car Review")

user_input = st.text_area(
    "Type your comment here:",
    value=st.session_state.input_text
)

# ---------------------------
# Prediction
# ---------------------------
if st.button("Predict Comment Type"):

    if user_input.strip() != "":

        cleaned = clean_text(user_input)
        input_vector = vectorizer.transform([cleaned])
        prediction = model.predict(input_vector)[0]

        # Confidence (approximation using decision score)
        decision_scores = model.decision_function(input_vector)
        confidence = float(np.max(np.abs(decision_scores)))

        st.subheader("Prediction Result")

        if prediction == "Positive":
            st.success("🟢 Positive")
        elif prediction == "Negative":
            st.warning("🟠 Negative")
        else:
            st.error("🔴 Toxic")

        st.write(f"Confidence Score: {round(confidence, 2)}")

        st.subheader("Cleaned Text")
        st.info(cleaned)

        # Save to history
        st.session_state.history.append((user_input, prediction))

    else:
        st.warning("Please enter a comment.")

# ---------------------------
# Prediction History
# ---------------------------
if len(st.session_state.history) > 0:

    st.subheader("Prediction History (Last 5)")

    for i, (text, pred) in enumerate(
        reversed(st.session_state.history[-5:])
    ):
        st.write(f"{i+1}. {text} → {pred}")

# ---------------------------
# Statistics Chart
# ---------------------------
if len(st.session_state.history) > 0:

    counts = {"Positive": 0, "Negative": 0, "Toxic": 0}

    for _, pred in st.session_state.history:
        counts[pred] += 1

    st.subheader("📊 Prediction Statistics")

    fig, ax = plt.subplots()
    ax.bar(
        counts.keys(),
        counts.values(),
    )
    ax.set_ylabel("Number of Predictions")
    ax.set_title("Prediction Distribution")

    st.pyplot(fig)
