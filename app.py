import streamlit as st
from src.predictor import predict_job

st.title("Fake Job Posting Detection System")

job_text = st.text_area("Paste Job Description")

if st.button("Analyze"):
    result = predict_job(job_text)
    st.success(f"Prediction: {result}")