import streamlit as st
import pickle
import re
import nltk
import numpy as np
import matplotlib.pyplot as plt

from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

# ---------------------------
# Load Model
# ---------------------------
model = pickle.load(open("final_model.pkl","rb"))
vectorizer = pickle.load(open("tfidf_vectorizer.pkl","rb"))

# ---------------------------
# NLTK Setup
# ---------------------------
nltk.download("stopwords",quiet=True)
nltk.download("wordnet",quiet=True)

stop_words = set(stopwords.words("english"))
lemmatizer = WordNetLemmatizer()

# ---------------------------
# Session State
# ---------------------------
if "history" not in st.session_state:
    st.session_state.history = []

# ---------------------------
# Text Cleaning
# ---------------------------
def clean_text(text):

    text = text.lower()

    text = re.sub(r"http\S+","",text)
    text = re.sub(r"[^a-zA-Z\s]","",text)

    words = text.split()

    words = [w for w in words if w not in stop_words]

    words = [lemmatizer.lemmatize(w) for w in words]

    return " ".join(words)

# ---------------------------
# Title
# ---------------------------
st.title("NLP Review Classification System")

st.markdown("""
This NLP system classifies text into:

- 🟢 **Positive**
- 🟠 **Negative**
- ⚪ **Neutral**

Model Used: **LinearSVC**  
Feature Extraction: **TF-IDF**
""")

# ---------------------------
# NLP Pipeline
# ---------------------------
st.subheader("NLP Pipeline")

st.markdown("""
User Comment  
⬇  
Text Preprocessing  
⬇  
TF-IDF Feature Extraction  
⬇  
Machine Learning Model (LinearSVC)  
⬇  
Sentiment Prediction
""")

# ---------------------------
# Model Information
# ---------------------------
st.subheader("Model Information")

st.write("Model Type: LinearSVC")
st.write("Vectorization: TF-IDF")

vocab_size = len(vectorizer.get_feature_names_out())

st.write("Vocabulary Size:", vocab_size)

# ---------------------------
# Example Buttons
# ---------------------------
st.subheader("Example Comments")

col1,col2,col3 = st.columns(3)

with col1:
    if st.button("Positive Example"):
        st.session_state.input = "This phone works perfectly"

with col2:
    if st.button("Negative Example"):
        st.session_state.input = "The price is too expensive"

with col3:
    if st.button("Neutral Example"):
        st.session_state.input = "The meeting is tomorrow"

# ---------------------------
# User Input
# ---------------------------
st.subheader("Enter Your Comment")

user_input = st.text_area(
    "Type your sentence:",
    value=st.session_state.get("input","")
)

# ---------------------------
# Prediction
# ---------------------------
if st.button("Predict"):

    if user_input.strip()!="":

        st.subheader("NLP Processing Steps")

        # Original
        st.write("Original Text:")
        st.info(user_input)

        # Tokenization
        tokens = user_input.lower().split()
        st.write("Tokenization:")
        st.write(tokens)

        # Stopword removal
        tokens_no_stop = [w for w in tokens if w not in stop_words]
        st.write("After Stopword Removal:")
        st.write(tokens_no_stop)

        # Lemmatization
        lemmas = [lemmatizer.lemmatize(w) for w in tokens_no_stop]
        st.write("After Lemmatization:")
        st.write(lemmas)

        # Cleaned text
        cleaned = " ".join(lemmas)
        st.write("Cleaned Text:")
        st.info(cleaned)

        # TF-IDF vector
        input_vector = vectorizer.transform([cleaned])

        st.write("TF-IDF Vector Shape:")
        st.write(input_vector.shape)

        st.write("TF-IDF Vector Sample:")
        st.write(input_vector.toarray()[0][:20])

        # Top Features
        feature_names = vectorizer.get_feature_names_out()

        nonzero = input_vector.nonzero()[1]

        top_features = [feature_names[i] for i in nonzero[:10]]

        st.write("Top TF-IDF Features:")
        st.write(top_features)

        # Prediction
        prediction = model.predict(input_vector)[0]

        # Confidence
        decision_scores = model.decision_function(input_vector)

        confidence = float(np.max(np.abs(decision_scores)))

        # Display
        st.subheader("Prediction Result")

        if prediction == 2:
            st.success("🟢 Positive")

        elif prediction == 0:
            st.warning("🟠 Negative")

        else:
            st.info("⚪ Neutral")

        st.write("Confidence Score:",round(confidence,2))

        st.session_state.history.append((user_input,prediction))

    else:
        st.warning("Please enter text")

# ---------------------------
# Prediction History
# ---------------------------
if len(st.session_state.history)>0:

    st.subheader("Prediction History")

    for i,(text,pred) in enumerate(
        reversed(st.session_state.history[-5:])
    ):

        st.write(f"{i+1}. {text}")

# ---------------------------
# Statistics Chart
# ---------------------------
if len(st.session_state.history)>0:

    counts = {
        "Positive":0,
        "Negative":0,
        "Neutral":0
    }

    for _,pred in st.session_state.history:

        if pred==2:
            counts["Positive"]+=1
        elif pred==0:
            counts["Negative"]+=1
        else:
            counts["Neutral"]+=1

    st.subheader("Prediction Statistics")

    fig,ax = plt.subplots()

    colors = ["green","orange","gray"]

    ax.bar(
        counts.keys(),
        counts.values(),
        color=colors
    )

    ax.set_ylabel("Predictions")
    ax.set_title("Sentiment Distribution")

    st.pyplot(fig)
