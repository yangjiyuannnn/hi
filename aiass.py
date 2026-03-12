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

if "input_text" not in st.session_state:
    st.session_state.input_text = ""

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
Vectorization: **TF-IDF**
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
# Model Info
# ---------------------------
st.subheader("Model Information")

st.write("Model Type: LinearSVC")

vocab_size = len(vectorizer.get_feature_names_out())

st.write("Vocabulary Size:", vocab_size)

st.write("Feature Type: TF-IDF")

# ---------------------------
# Example Buttons
# ---------------------------
st.subheader("Example Comments")

col1,col2,col3 = st.columns(3)

with col1:
    if st.button("Positive Example"):
        st.session_state.input_text = "This phone works perfectly"

with col2:
    if st.button("Negative Example"):
        st.session_state.input_text = "The price is too expensive"

with col3:
    if st.button("Neutral Example"):
        st.session_state.input_text = "The meeting is tomorrow"

# ---------------------------
# User Input
# ---------------------------
st.subheader("Enter Your Comment")

user_input = st.text_area(
    "Type your sentence:",
    value=st.session_state.input_text
)

# ---------------------------
# Prediction
# ---------------------------
if st.button("Predict"):

    if user_input.strip() != "":

        st.subheader("NLP Processing Steps")

        # Original text
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

        # TF-IDF
        input_vector = vectorizer.transform([cleaned])

        st.write("TF-IDF Vector Shape:")
        st.write(input_vector.shape)

        st.write("TF-IDF Vector Sample:")
        st.write(input_vector.toarray()[0][:20])

        # Vocabulary check
        if input_vector.nnz == 0:
            st.warning("⚠ No known vocabulary detected")

        # Top features
        feature_names = vectorizer.get_feature_names_out()

        nonzero = input_vector.nonzero()[1]

        top_features = [feature_names[i] for i in nonzero[:10]]

        st.write("Top Influential Words:")

        for word in top_features:
            st.write("•",word)

        # Prediction
        prediction = str(model.predict(input_vector)[0])

        decision_scores = model.decision_function(input_vector)

        confidence = float(np.max(np.abs(decision_scores)))

        # ---------------------------
        # Display Result
        # ---------------------------
        st.subheader("Prediction Result")

        if prediction == "Positive":
            st.success("🟢 Positive Sentiment")

        elif prediction == "Negative":
            st.warning("🟠 Negative Sentiment")

        else:
            st.info("⚪ Neutral Statement")

        st.write("Confidence Score:",round(confidence,2))

        # Confidence bar
        confidence_norm = min(confidence / 2, 1)

        st.progress(confidence_norm)

        st.caption("Prediction Confidence")

        st.session_state.history.append((user_input,prediction))

    else:
        st.warning("Please enter text")

# ---------------------------
# Prediction History
# ---------------------------
if len(st.session_state.history) > 0:

    st.subheader("Prediction History")

    for i,(text,pred) in enumerate(
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
        "Neutral":0
    }

    for _,pred in st.session_state.history:

        if pred == "Positive":
            counts["Positive"] += 1

        elif pred == "Negative":
            counts["Negative"] += 1

        else:
            counts["Neutral"] += 1

    st.subheader("Prediction Statistics")

    fig,ax = plt.subplots()

    colors = ["#2ecc71","#e67e22","#95a5a6"]

    ax.bar(
        counts.keys(),
        counts.values(),
        color=colors
    )

    ax.set_ylabel("Number of Predictions")

    ax.set_title("Sentiment Distribution")

    st.pyplot(fig)
