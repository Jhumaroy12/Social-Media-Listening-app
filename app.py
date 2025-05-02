import streamlit as st
import joblib
import os

def load_model():
    model_path = "logistic_regression_model.pkl"  
    vectorizer_path = "tfidf_vectorizer.pkl" 
    
    if not os.path.exists(model_path) or not os.path.exists(vectorizer_path):
        st.error("Model or vectorizer file not found. Please check the file path.")
        return None, None
    
    return joblib.load(model_path), joblib.load(vectorizer_path)

def predict_emotion(text):
    model, vectorizer = load_model()
    if model is None or vectorizer is None:
        return "", 0.0
    text_vector = vectorizer.transform([text])
    prediction = model.predict(text_vector)[0]
    confidence = model.predict_proba(text_vector).max() if hasattr(model, 'predict_proba') else 1.0
    return prediction, confidence

st.title("Social Media Listening")
st.write("Enter text to predict the emotion.")

user_input = st.text_area("Enter your text:")
if st.button("Predict Emotion"):
    if user_input.strip():
        prediction, confidence = predict_emotion(user_input)
        st.write(f"### Predicted Emotion: {prediction}")
        st.write(f"**Confidence:** {confidence:.2f}")
    else:
        st.warning("Please enter some text to analyze.")