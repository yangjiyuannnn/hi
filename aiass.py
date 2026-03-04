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
# Toxic Keywords
# ---------------------------
toxic_words = [
    "fuck","shit","bitch","idiot","stupid",
    "trash","garbage","asshole","bastard"
]
negative_keywords = [
"too expensive",
"too slow",
"too high",
"too low",
"too noisy",
"too heavy",
"too big",
"too small",
"maintenance cost",
"fuel consumption",
"fuel economy",
"engine problem",
"poor performance",
"bad quality",
"not reliable"
]
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
This NLP system classifies text into:

- 🟢 **Positive**
- 🟠 **Negative**
- 🔴 **Toxic**
- ⚪ **Neutral**

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
st.subheader("Enter Your Sentence")

user_input = st.text_area(
    "Type your sentence here:",
    value=st.session_state.input_text
)

# ---------------------------
# Prediction
# ---------------------------
if st.button("Predict Comment Type"):

    if user_input.strip() != "":

        text_lower = user_input.lower()

        # 1️⃣ Toxic keyword detection
        if any(word in text_lower for word in toxic_words):

            prediction = "Toxic"
            confidence = 1.0
        # Negative complaint detection
        elif any(word in text_lower for word in negative_keywords):

            prediction = "Negative"
            confidence = 0.9
        # 2️⃣ Short sentence neutral
        elif len(user_input.split()) <= 2:

            prediction = "Neutral"
            confidence = 0.5

        else:

            cleaned = clean_text(user_input)
            input_vector = vectorizer.transform([cleaned])
            prediction = model.predict(input_vector)[0]

            decision_scores = model.decision_function(input_vector)
            confidence = float(np.max(np.abs(decision_scores)))

            # 3️⃣ Toxic fallback fix
            if prediction == "Toxic":
                prediction = "Neutral"

        # ---------------------------
        # Display Result
        # ---------------------------
        st.subheader("Prediction Result")

        if prediction == "Positive":
            st.success("🟢 Positive")

        elif prediction == "Negative":
            st.warning("🟠 Negative")

        elif prediction == "Neutral":
            st.info("⚪ Neutral")

        else:
            st.error("🔴 Toxic")

        st.write(f"Confidence Score: {round(confidence,2)}")

        # Save history
        st.session_state.history.append((user_input, prediction))

    else:
        st.warning("Please enter a sentence.")

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

    counts = {
        "Positive":0,
        "Negative":0,
        "Toxic":0,
        "Neutral":0
    }

    for _, pred in st.session_state.history:
        counts[pred] += 1

    st.subheader("📊 Prediction Statistics")

    fig, ax = plt.subplots()

    ax.bar(
        counts.keys(),
        counts.values()
    )

    ax.set_ylabel("Number of Predictions")
    ax.set_title("Prediction Distribution")

    st.pyplot(fig)
