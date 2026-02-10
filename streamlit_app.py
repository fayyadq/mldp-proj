import streamlit as st
import pandas as pd
import numpy as np
import joblib

# ==========================================
# 1️⃣ Feature engineering
# ==========================================
def engineer_features(X):
    X = X.copy()
    if 'study_hours' in X.columns and 'class_attendance' in X.columns:
        X['study_attendance_interaction'] = X['study_hours'] * X['class_attendance']
    if 'study_hours' in X.columns and 'sleep_hours' in X.columns:
        X['study_sleep_ratio'] = X['study_hours'] / (X['sleep_hours'] + 1)
    return X

# ==========================================
# 2️⃣ Load model
# ==========================================
@st.cache_data
def load_model():
    return joblib.load("exam_model_champion.pkl")

model = load_model()

# ==========================================
# 3️⃣ Streamlit UI
# ==========================================
st.title("📊 Exam Score Predictor")
st.header("📥 Enter Student Details")

# Numeric Inputs
user_numeric = {}
user_numeric['study_hours'] = st.number_input("Study Hours per Day", 0.0, 24.0, 5.0)
user_numeric['class_attendance'] = st.slider("Class Attendance Rate", 0.0, 1.0, 0.8)
user_numeric['sleep_hours'] = st.number_input("Sleep Hours per Day", 0.0, 12.0, 7.0)

# Categorical Inputs
user_categorical = {}
user_categorical['gender'] = st.selectbox("Gender", ["male", "female", "other"])
user_categorical['internet_access'] = st.selectbox("Internet Access", ["yes", "no"])
user_categorical['study_method'] = st.selectbox("Study Method", ["self-study", "group-study", "tutor"])

# Build input DataFrame
input_data = pd.DataFrame([{**user_numeric, **user_categorical}])

# Engineer features
input_data = engineer_features(input_data)

# ==========================================
# 4️⃣ Handle missing columns (dynamic)
# ==========================================
# Get columns that the model expects (from training)
try:
    expected_columns = model.named_steps['pre'].get_feature_names_out()
except AttributeError:
    # fallback if preprocessor does not support get_feature_names_out
    expected_columns = input_data.columns.tolist()

# Add missing columns with defaults
for col in expected_columns:
    if col not in input_data.columns:
        if col in ['study_hours', 'class_attendance', 'sleep_hours', 'study_attendance_interaction', 'study_sleep_ratio']:
            input_data[col] = 0.0
        else:
            input_data[col] = "unknown"

# Reorder columns
input_data = input_data[expected_columns]

# ==========================================
# 5️⃣ Prediction
# ==========================================
if st.button("🎯 Predict Exam Score"):
    try:
        prediction = model.predict(input_data)[0]
        prediction = max(0, min(100, prediction))  # clip 0-100
        st.success(f"📘 Predicted Exam Score: {prediction:.2f}")

        st.write("### 📌 Interpretation")
        if prediction >= 75:
            st.write("Excellent performance expected.")
        elif prediction >= 50:
            st.write("Average performance expected.")
        else:
            st.write("Student may require additional academic support.")
    except Exception as e:
        st.error(f"Prediction failed: {e}")

st.markdown("---")
st.caption("Model: Feature-Engineered Ridge Regression | MLDP Project | Streamlit")
