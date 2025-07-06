# Predicting-Depression
# 🧠 Predicting Depression from Mental Health Survey Data using Deep Learning

This project uses a deep learning model built with PyTorch to predict the likelihood of depression based on responses to a mental health survey. It includes a real-time prediction interface built using Streamlit.

## 🚀 Project Overview

- **Domain:** Healthcare / Mental Health
- **Goal:** Predict whether an individual is likely to experience depression
- **Tech Stack:** Python, PyTorch, Streamlit, Pandas, Scikit-learn
- **Deployment:** Streamlit Web App

---

## 📁 Files Included

| File / Folder              | Description                                      |
|---------------------------|--------------------------------------------------|
| `stream.py`               | Main Streamlit app to run predictions            |
| `preprocessor.pkl`        | Preprocessing pipeline (encoders, scalers, etc.) |
| `depression_model.pth`    | Trained PyTorch model                            |
| `.gitignore`              | Files/folders to ignore in Git tracking          |
| `user_predictions.xlsx`   | Sample output file with predictions              |

---

## 📊 Input Format

| Column                            | Example              |
|----------------------------------|----------------------|
| Name                             | John Doe             |
| Gender                           | Male                 |
| Age                              | 25                   |
| City                             | Chennai              |
| Working Professional or Student  | Working Professional |
| Academic Pressure (1–5)          | 3                    |
| Work Pressure (1–5)              | 4                    |

---

## ✅ Output Format

| Name     | Gender | Age | City    | ... | Predicted_Probability | Predicted_Depression |
|----------|--------|-----|---------|-----|------------------------|----------------------|
| John Doe | Male   | 25  | Chennai | ... | 0.76                   | High                 |
| Alice    | Female | 22  | Mumbai  | ... | 0.34                   | Low                  |

---

## ▶️ How to Run Locally

1. **Clone this repository**:
    ```bash
    git clone https://github.com/sharmila122125/Predicting-Depression.git
    cd Predicting-Depression
    ```

2. **Install dependencies**:
    ```bash
    pip install -r requirements.txt
    ```

3. **Run Streamlit app**:
    ```bash
    streamlit run stream.py
    ```

---

## 🧠 Model Architecture

- Input: Encoded numerical + categorical survey responses
- Layers: Fully connected (MLP)
- Output: Binary classification (High / Low Depression)
- Loss: Binary Cross Entropy
- Optimizer: Adam

---

## 🌐 Deployment

You can deploy the app on:
- [Streamlit Cloud](https://share.streamlit.io/)
- AWS Elastic Beanstalk / EC2

---

## 📬 Contact

If you have any feedback or questions, feel free to reach out!

---


