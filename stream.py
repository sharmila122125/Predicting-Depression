import streamlit as st
import pandas as pd
import torch
import torch.nn as nn
import cloudpickle
import numpy as np 

# ------------------------
#  Set Page Config (First Streamlit Command)
# ------------------------
st.set_page_config(page_title=" Depression Predictor", layout="wide")
st.title(" Real-Time Depression Predictor")


# ------------------------
#  Define Model Architecture Matching Saved State
# ------------------------
class DepressionModel(nn.Module):
    def __init__(self, input_dim):
        super(DepressionModel, self).__init__()
        self.network = nn.Sequential(
            nn.Linear(input_dim, 64),  # layer 0
            nn.ReLU(),                 # layer 1
            nn.Dropout(0.3),           # layer 2
            nn.Linear(64, 32),         # layer 3
            nn.ReLU(),                 # layer 4
            nn.Linear(32, 1)           # layer 5 — FINAL layer
        )

    def forward(self, x):
        return self.network(x)

# ------------------------
#  Load Preprocessor and Model
# ------------------------
@st.cache_resource
def load_model_preprocessor():
    with open("preprocessor.pkl", "rb") as f:
        preprocessor = cloudpickle.load(f)

    input_dim = len(preprocessor.feature_names)
    model = DepressionModel(input_dim)

    state_dict = torch.load("depression_model.pth", map_location=torch.device("cpu"))
    model.load_state_dict(state_dict)
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

        # Ensure always a 1D array (even if 1 record)
        if probs.ndim == 0:
            probabilities = np.array([probs.item()])
        else:
            probabilities = probs.numpy()

        predicted_labels = (probabilities >= 0.5).astype(int)
        depression_status = ['High' if label == 1 else 'Low' for label in predicted_labels]

        return probabilities, depression_status


# ------------------------
#  Load CSV Automatically (No Upload)
# ------------------------
try:
    df = pd.read_csv("test_predictions.csv")
    st.success("Loaded 'test_predictions.csv' successfully.")

 

    # Run prediction
    probabilities, predictions = predict_depression(df)
# Append predictions to DataFrame
    df["Predicted_Probability"] = probabilities
    df["Predicted_Depression"] = predictions

#  Show full prediction table
    st.markdown(" **Full Prediction Table**")
    st.dataframe(df, use_container_width=True)

#  Download button
    csv_download = df.to_csv(index=False).encode("utf-8")
    st.download_button(
    label="📥 Download Prediction CSV",
    data=csv_download,
    file_name="depression_predictions.csv",
    mime="text/csv"
)



except FileNotFoundError:
    st.error(" 'test_predictions.csv' file not found. Please make sure it's in the same folder as the app.")
except Exception as e:
    st.error(f" Error during prediction: {e}")

