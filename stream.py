# ==============================================
# Depression Prediction: Real-time + Batch App
# ==============================================

import streamlit as st
import pandas as pd
import cloudpickle
import torch


import numpy as np
from torch import nn
import os
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer

#  Page configuration
st.set_page_config(page_title="Depression Predictor", layout="centered")

# ------------------------------------------------------
# Define the DepressionPreprocessor class (REQUIRED)
# ------------------------------------------------------
class DepressionPreprocessor(BaseEstimator, TransformerMixin):
    def __init__(self):
        self.column_transformer = None
        self.feature_names = None
        self.columns_to_drop = ['id', 'Name', 'City']

    def fit(self, X, y=None):
        X = X.drop(columns=self.columns_to_drop, errors='ignore')
        self.num_cols = X.select_dtypes(include=['float64', 'int64']).columns.tolist()
        self.cat_cols = X.select_dtypes(include='object').columns.tolist()

        num_pipeline = Pipeline([
            ('imputer', SimpleImputer(strategy='mean')),
            ('scaler', StandardScaler())
        ])

        cat_pipeline = Pipeline([
            ('imputer', SimpleImputer(strategy='constant', fill_value='Unknown')),
            ('encoder', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
        ])

        self.column_transformer = ColumnTransformer([
            ('num', num_pipeline, self.num_cols),
            ('cat', cat_pipeline, self.cat_cols)
        ])

        self.column_transformer.fit(X)
        num_features = self.num_cols
        cat_features = list(self.column_transformer.named_transformers_['cat']['encoder'].get_feature_names_out(self.cat_cols))
        self.feature_names = num_features + cat_features
        return self

    def transform(self, X):
        X = X.drop(columns=self.columns_to_drop, errors='ignore')
        return pd.DataFrame(self.column_transformer.transform(X), columns=self.feature_names)

    def fit_transform(self, X, y=None):
        return self.fit(X).transform(X)

# ------------------------------------------------------
# Define PyTorch Model (Same as during training)
# ------------------------------------------------------
class DepressionModel(nn.Module):
    def __init__(self, input_dim):
        super(DepressionModel, self).__init__()
        self.network = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.network(x)

# ------------------------------------------------------
# Load Preprocessor & Model
# ------------------------------------------------------

@st.cache_resource
def load_model_preprocessor():
    with open("preprocessor.pkl", "rb") as f:
        preprocessor = cloudpickle.load(f)
    
    input_dim = len(preprocessor.feature_names)
    
    model = DepressionModel(input_dim)
    model.load_state_dict(torch.load("depression_model.pth", map_location=torch.device("cpu")))
    model.eval()
    
    return preprocessor, model

# ------------------------------------------------------
# Predict function for a single input
# ------------------------------------------------------
def predict_single(input_dict):
    df = pd.DataFrame([input_dict])
    processed = preprocessor.transform(df)
    tensor = torch.tensor(processed.values, dtype=torch.float32)
    with torch.no_grad():
        prob = model(tensor).item()
    label = "Depressed" if prob > 0.5 else "Not Depressed"
    return prob, label

# ------------------------------------------------------
# UI: Real-time Prediction
# ------------------------------------------------------
st.title(" Real-Time Depression Predictor")



# ------------------------------------------------------
# Batch Prediction for First 1000 Rows from Test Data
# ------------------------------------------------------
st.header(" Batch Prediction: First 1000 Users")

# Check if test file exists
test_file_path = "test.csv"
if os.path.exists(test_file_path):
    # Load test data and select first 1000 rows
    test_df = pd.read_csv(test_file_path).head(1000)

    # Load preprocessor and model BEFORE using
    preprocessor, model = load_model_preprocessor()

    # Store user info (Name, City, etc.) before dropping
    meta_info = test_df[["Name", "Gender", "Age", "City", "Working Professional or Student"]].copy()

    # Preprocess and predict
    processed_data = preprocessor.transform(test_df)

    tensor = torch.tensor(processed_data.values, dtype=torch.float32)
    with torch.no_grad():
        probs = model(tensor).squeeze().numpy()
        labels = ["Depressed" if p > 0.5 else "Not Depressed" for p in probs]

    # Combine original + prediction
    prediction_df = meta_info.copy()
    prediction_df["Predicted_Probability"] = np.round(probs, 2)
    prediction_df["Predicted_Depression"] = labels

    st.subheader(" First 1000 Predictions")
    st.dataframe(prediction_df)

    batch_output_file = "first_1000_predictions.csv"
    prediction_df.to_csv(batch_output_file, index=False)

    with open(batch_output_file, "rb") as f:
        st.download_button("⬇ Download First 1000 Predictions as CSV", f, file_name=batch_output_file, mime="text/csv")
else:
    st.warning(" test.csv file not found. Please ensure it's placed in the app directory.")

    
    
    
    
# --------------------------------
# Real-time form inputs (in form)
# --------------------------------
with st.form("real_time_form"):
    st.subheader(" Enter Your Information")
    name = st.text_input("Name", "Anonymous")
    gender = st.selectbox("Gender", ["Male", "Female", "Other"])
    age = st.slider("Age", 18, 60, 30)
    city = st.text_input("City", "Chennai")
    status = st.selectbox("Status", ["Working Professional", "Student"])
    profession = st.text_input("Profession", "Engineer")
    academic_pressure = st.slider("Academic Pressure", 1, 5, 3)
    work_pressure = st.slider("Work Pressure", 1, 5, 3)
    cgpa = st.slider("CGPA", 5.0, 10.0, 7.5)
    study_satisfaction = st.slider("Study Satisfaction", 1, 5, 3)
    job_satisfaction = st.slider("Job Satisfaction", 1, 5, 3)
    sleep = st.selectbox("Sleep Duration", ["< 4 hours", "4-6 hours", "6-8 hours", "> 8 hours"])
    diet = st.selectbox("Dietary Habits", ["Vegetarian", "Non-Vegetarian", "Vegan", "Mixed"])
    degree = st.text_input("Degree", "B.Tech")
    suicidal = st.selectbox("Suicidal Thoughts", ["Yes", "No"])
    hours = st.slider("Work/Study Hours", 0.0, 12.0, 6.0)
    financial = st.slider("Financial Stress", 1, 5, 3)
    family_history = st.selectbox("Family History of Mental Illness", ["Yes", "No"])

    submitted = st.form_submit_button(" Predict")

# -------------------------------
# Prediction + CSV Output Block
# -------------------------------
if submitted:
    input_data = {
        "Name": name,
        "Gender": gender,
        "Age": age,
        "City": city,
        "Working Professional or Student": status,
        "Profession": profession,
        "Academic Pressure": academic_pressure,
        "Work Pressure": work_pressure,
        "CGPA": cgpa,
        "Study Satisfaction": study_satisfaction,
        "Job Satisfaction": job_satisfaction,
        "Sleep Duration": sleep,
        "Dietary Habits": diet,
        "Degree": degree,
        "Have you ever had suicidal thoughts ?": suicidal,
        "Work/Study Hours": hours,
        "Financial Stress": financial,
        "Family History of Mental Illness": family_history
    }

    prob, result = predict_single(input_data)
    input_data["Predicted_Probability"] = round(prob, 2)
    input_data["Predicted_Depression"] = result

    result_df = pd.DataFrame([input_data])
    st.subheader(" Prediction Result")
    st.dataframe(result_df)

    output_file = "realtime_prediction.csv"
    result_df.to_csv(output_file, index=False)

    with open(output_file, "rb") as f:
        st.download_button("⬇ Download Prediction as CSV", f, file_name=output_file, mime="text/csv")