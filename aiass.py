import streamlit as st
import pickle
import re
import nltk
import numpy as np
import matplotlib.pyplot as plt

from nltk.corpus import stopwords
from sklearn.metrics import confusion_matrix
import seaborn as sns

# ---------------------------
# Load Model
# ---------------------------
model = pickle.load(open("final_model.pkl","rb"))
vectorizer = pickle.load(open("tfidf_vectorizer.pkl","rb"))

# ---------------------------
# NLTK Setup
# ---------------------------
nltk.download("stopwords",quiet=True)

stop_words = set(stopwords.words("english"))
stop_words = stop_words - {"no","not","never"}

# ---------------------------
# Session State
# ---------------------------
if "history" not in st.session_state:
    st.session_state.history = []

if "input_text" not in st.session_state:
    st.session_state.input_text = ""

# ---------------------------
# Clean Text (SAME AS COLAB)
# ---------------------------
def clean_text(text):

    text = str(text).lower()

    text = text.replace("not bad","not_bad")
    text = text.replace("no bad","not_bad")
    text = text.replace("no problem","no_problem")

    text = re.sub(r"http\S+","",text)
    text = re.sub(r"@\w+","",text)
    text = re.sub(r"#\w+","",text)

    text = re.sub(r"[^a-z\s]","",text)

    words = text.split()

    words = [w for w in words if w not in stop_words]

    return " ".join(words)

# ---------------------------
# Title
# ---------------------------
st.title("NLP Review Classification System")

st.markdown("""
### Sentiment Categories

🟢 Positive  
🟠 Negative  
⚪ Neutral  

Model: **LinearSVC**  
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

vocab_size = len(vectorizer.get_feature_names_out())

st.write("Model Type:", "LinearSVC")
st.write("Vocabulary Size:", vocab_size)
st.write("Feature Type:", "TF-IDF (Unigram + Bigram)")

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
        st.write("Original Text")
        st.info(user_input)

        # Tokenization
        tokens = user_input.lower().split()
        st.write("Tokenization")
        st.write(tokens)

        # Cleaned text (same preprocessing as training)
        cleaned = clean_text(user_input)

        st.write("Cleaned Text")
        st.info(cleaned)

        # TF-IDF vector
        input_vector = vectorizer.transform([cleaned])

        st.write("TF-IDF Vector Shape")
        st.write(input_vector.shape)

        # vocabulary check
        if input_vector.nnz == 0:
            st.warning("⚠ Words not found in vocabulary")

        # Top features
        feature_names = vectorizer.get_feature_names_out()
        nonzero = input_vector.nonzero()[1]
        top_features = [feature_names[i] for i in nonzero[:10]]

        st.write("Top Influential Words")

        for word in top_features:
            st.write("•",word)

        # ---------------------------
        # Model Prediction
        # ---------------------------
        prediction = model.predict(input_vector)[0]

        decision_scores = model.decision_function(input_vector)
        confidence = float(np.max(np.abs(decision_scores)))

        # ---------------------------
        # Display Result
        # ---------------------------
        st.subheader("Prediction Result")

        if prediction == 2:
            label = "Positive"
            st.success("🟢 Positive Sentiment")

        elif prediction == 0:
            label = "Negative"
            st.warning("🟠 Negative Sentiment")

        else:
            label = "Neutral"
            st.info("⚪ Neutral Statement")

        st.write("Confidence Score:",round(confidence,2))

        confidence_norm = min(confidence/2,1)
        st.progress(confidence_norm)

        st.caption("Prediction Confidence")

        st.session_state.history.append((user_input,prediction,label))

    else:
        st.warning("Please enter text")

# ---------------------------
# Prediction History
# ---------------------------
if len(st.session_state.history) > 0:

    st.subheader("Prediction History")

    for i,(text,_,label) in enumerate(
        reversed(st.session_state.history[-5:])
    ):
        st.write(f"{i+1}. {text} → {label}")

# ---------------------------
# Statistics Chart
# ---------------------------
if len(st.session_state.history) > 0:

    counts = {
        "Positive":0,
        "Negative":0,
        "Neutral":0
    }

    for _,pred,_ in st.session_state.history:

        if pred == 2:
            counts["Positive"] += 1
        elif pred == 0:
            counts["Negative"] += 1
        else:
            counts["Neutral"] += 1

    st.subheader("Prediction Statistics")

    fig,ax = plt.subplots()

    ax.bar(
        counts.keys(),
        counts.values(),
        color=["#2ecc71","#e67e22","#95a5a6"]
    )

    ax.set_ylabel("Number of Predictions")
    ax.set_title("Sentiment Distribution")

    st.pyplot(fig)

# ---------------------------
# Confusion Matrix Demo
# ---------------------------
if len(st.session_state.history) > 5:

    st.subheader("Demo Confusion Matrix")

    y_true = [x[1] for x in st.session_state.history]
    y_pred = [x[1] for x in st.session_state.history]

    cm = confusion_matrix(y_true,y_pred)

    fig,ax = plt.subplots()

    sns.heatmap(cm,annot=True,fmt="d",cmap="Blues",ax=ax)

    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")

    st.pyplot(fig)
