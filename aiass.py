import streamlit as st
import pickle
import re
import nltk
import numpy as np
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import matplotlib.pyplot as plt
# Load trained model and vectorizer
model = pickle.load(open("final_model.pkl", "rb"))
vectorizer = pickle.load(open("tfidf_vectorizer.pkl", "rb"))

nltk.download("stopwords", quiet=True)
nltk.download("wordnet", quiet=True)

stop_words = set(stopwords.words("english"))
lemmatizer = WordNetLemmatizer()

# Text cleaning function
def clean_text(text):
    text = text.lower()
    text = re.sub(r"http\S+", "", text)
    text = re.sub(r"[^a-zA-Z\s]", "", text)
    words = text.split()
    words = [word for word in words if word not in stop_words]
    words = [lemmatizer.lemmatize(word) for word in words]
    return " ".join(words)

# Page Title
st.title("🚗 Car Review NLP Classification System")

st.markdown("""
### 📌 System Description
This system classifies car review comments into:
- 🟢 Positive  
- 🟠 Negative  
- 🔴 Toxic  

Model: LinearSVC  
Vectorization: TF-IDF  
""")

# Example Buttons
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

# User Input
st.subheader("Enter Your Car Review")

if "input_text" not in st.session_state:
    st.session_state.input_text = ""

user_input = st.text_area("Type your comment here:", value=st.session_state.input_text)

# Prediction
if st.button("Predict Comment Type"):

    if user_input.strip() != "":

        cleaned = clean_text(user_input)
        input_vector = vectorizer.transform([cleaned])
        prediction = model.predict(input_vector)[0]

        # Confidence (approximation using decision function)
        decision_scores = model.decision_function(input_vector)
        confidence = np.max(np.abs(decision_scores))

        st.subheader("Prediction Result")

        if prediction == "Positive":
            st.success(f"🟢 Positive")
        elif prediction == "Negative":
            st.warning(f"🟠 Negative")
        else:
            st.error(f"🔴 Toxic")

        st.write(f"Confidence Score: {round(float(confidence), 2)}")

        st.subheader("Cleaned Text")
        st.info(cleaned)

        # Save history
        if "history" not in st.session_state:
            st.session_state.history = []

        st.session_state.history.append((user_input, prediction))

    else:
        st.warning("Please enter a comment.")

# Display History
if "history" in st.session_state and len(st.session_state.history) > 0:
    st.subheader("Prediction History")

    for i, (text, pred) in enumerate(reversed(st.session_state.history[-5:])):
        st.write(f"{i+1}. {text} → {pred}")

st.session_state.history.append((user_input, prediction))
# Count prediction types
counts = {"Positive": 0, "Negative": 0, "Toxic": 0}

for _, pred in st.session_state.history:
    counts[pred] += 1

st.subheader("Prediction Statistics")

# Create bar chart
fig, ax = plt.subplots()
ax.bar(counts.keys(), counts.values())
ax.set_ylabel("Number of Predictions")
ax.set_title("Prediction Distribution")

st.pyplot(fig)
