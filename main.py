import streamlit as st
import joblib
import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer

model = joblib.load("logistic_aivshuman.pkl")


st.title("AI vs Human Classifier")
st.write("predict if the text is written by a human or a AI")

input_text = st.text_area( 
    label="input text",
    height=200
)

st.info("⚠️ The model detects writing style, not authorship. Well-written human text may be flagged as AI.")

if st.button("Predict AI or not"):
    if input_text.strip() == "":
        st.warning("Please enter a text.")
    else:
        
        prediction = model.predict([input_text])[0]
        probability = model.predict_proba([input_text])[0][1]
        st.progress(float(probability))

        if len(input_text.split()) < 20:
            st.info("⚠️ Short texts are harder to classify accurately.")

        if prediction == 1:
            st.error(f"🚨 AI (Confidence: {probability*100:.2f}%)")
        else:
            st.success(f"✅ Human (Confidence: {(1 - probability)*100:.2f}%)")






