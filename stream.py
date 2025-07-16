import streamlit as st
import pandas as pd
import torch
import torch.nn as nn
import numpy as np
import cloudpickle
from phase2_model import DepressionModel


# ------------------------
# Page Config
# ------------------------
st.set_page_config(page_title="Depression Predictor", layout="wide")
st.title(" Real-Time Depression Predictor")


# ------------------------
# Load Preprocessor and Model
# ------------------------
@st.cache_resource
def load_model_preprocessor():
    with open("preprocessor.pkl", "rb") as f:
        preprocessor = cloudpickle.load(f)

    input_dim = len(preprocessor.feature_names)
    model = DepressionModel(input_dim)
    model.load_state_dict(torch.load("depression_model.pth", map_location=torch.device("cpu")))
    model.eval()
    return preprocessor, model

preprocessor, model = load_model_preprocessor()


# ------------------------
# Prediction Function
# ------------------------
def predict_depression(data_df):
    X = preprocessor.transform(data_df)
    X_tensor = torch.tensor(X.values, dtype=torch.float32)

    with torch.no_grad():
        outputs = model(X_tensor)
        probs = torch.sigmoid(outputs).squeeze()
        if probs.ndim == 0:
            probs = torch.tensor([probs.item()])
        predictions = (probs >= 0.5).int().numpy()
        labels = ['High' if p == 1 else 'Low' for p in predictions]
        return probs.numpy(), labels


# ------------------------
# Live Input Form First
# ------------------------
st.header(" Enter Details for Real-Time Prediction")

with st.form("depression_form"):
    gender = st.selectbox("Gender", ["Male", "Female"])
    age = st.slider("Age", 18, 60, 25)
    role = st.selectbox("Working Professional or Student", ["Working Professional", "Student"])
    profession = st.text_input("Profession", "")
    
    # These fields allow None (missing) via an extra "None" option
    academic_pressure = st.selectbox("Academic Pressure (1–5)", ["None", 1, 2, 3, 4, 5])
    work_pressure = st.selectbox("Work Pressure (1–5)", ["None", 1, 2, 3, 4, 5])
    cgpa_option = st.selectbox("CGPA", ["None"] + [round(x * 0.1, 2) for x in range(0, 101)])
    cgpa = None if cgpa_option == "None" else float(cgpa_option)

    study_satisfaction = st.selectbox("Study Satisfaction (1–5)", ["None", 1, 2, 3, 4, 5])
    job_satisfaction = st.selectbox("Job Satisfaction (1–5)", ["None", 1, 2, 3, 4, 5])
    work_study_hours = st.slider("Work/Study Hours", 0, 12, 6)
    financial_stress = st.selectbox("Financial Stress (1–5)", ["None", 1, 2, 3, 4, 5])
    sleep_duration = st.selectbox("Sleep Duration", ["Less than 5 hours", "5-6 hours", "7-8 hours", "More than 8 hours"])
    dietary_habits = st.selectbox("Dietary Habits", ["Healthy", "Moderate", "Unhealthy"])
    degree = st.selectbox(
    "Degree",
    [
        "None", "Class 12", "Diploma", "B.A", "B.Sc", "B.Com", "BBA", "BCA", "B.E", "B.Tech",
        "B.Arch", "B.Ed", "LLB", "MBBS", "M.A", "M.Sc", "M.Com", "MBA", "MCA", "M.Tech", "PhD", "MD"
    ]
    )

    suicidal_thoughts = st.selectbox("Have you ever had suicidal thoughts ?", ["Yes", "No"])
    family_history = st.selectbox("Family History of Mental Illness", ["Yes", "No"])

    submit = st.form_submit_button("Predict")

if submit:
    def parse(val):
        return None if val == "None" else val

    input_df = pd.DataFrame([{
        'Gender': gender,
        'Age': age,
        'Working Professional or Student': role,
        'Profession': profession or None,
        'Academic Pressure': parse(academic_pressure),
        'Work Pressure': parse(work_pressure),
        'CGPA': cgpa,
        'Study Satisfaction': parse(study_satisfaction),
        'Job Satisfaction': parse(job_satisfaction),
        'Work/Study Hours': work_study_hours,
        'Financial Stress': parse(financial_stress),
        'Sleep Duration': sleep_duration,
        'Dietary Habits': dietary_habits,
        'Degree': degree or None,
        'Have you ever had suicidal thoughts ?': suicidal_thoughts,
        'Family History of Mental Illness': family_history
    }])

    probs, labels = predict_depression(input_df)
    st.success(f"Predicted Depression Risk: **{labels[0]}** ({probs[0]:.2f} probability)")


# ------------------------
# CSV Prediction Output Below
# ------------------------
st.markdown("---")
st.header(" CSV Predictions Output")

try:
    df = pd.read_csv("test_predictions.csv")
    st.success("Loaded 'test_predictions.csv' successfully.")

    # Predict
    probs, labels = predict_depression(df)
    df["Predicted_Probability"] = probs
    df["Predicted_Depression"] = labels

    display_df = df.drop(columns=['id', 'Name', 'City'], errors='ignore')
    if "CGPA" in display_df.columns:
        display_df["CGPA"] = pd.to_numeric(display_df["CGPA"], errors="coerce")

    st.subheader(" Prediction Table")
    st.dataframe(display_df, use_container_width=True)

    csv_download = display_df.to_csv(index=False).encode("utf-8")
    st.download_button("⬇ Download Prediction CSV", csv_download, "depression_predictions.csv", "text/csv")

except FileNotFoundError:
    st.error("File 'test_predictions.csv' not found.")
except Exception as e:
    st.error(f"Error: {e}")
