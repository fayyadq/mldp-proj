import streamlit as st
import pandas as pd
import numpy as np
import joblib
from sklearn.compose import ColumnTransformer

# ==========================================
# Feature engineering
# ==========================================
def engineer_features(X):
    X = X.copy()
    # Only engineer interaction features that actually matter
    if 'study_hours' in X.columns and 'class_attendance' in X.columns:
        X['study_attendance_interaction'] = X['study_hours'] * X['class_attendance']
    if 'study_hours' in X.columns and 'sleep_hours' in X.columns:
        X['study_sleep_ratio'] = X['study_hours'] / (X['sleep_hours'] + 1)
    return X

# ==========================================
# Load model
# ==========================================
@st.cache_resource
def load_model():
    m = joblib.load("exam_model_champion.pkl")
    # Patch ColumnTransformer passthrough bug
    if hasattr(m, 'named_steps'):
        for step in m.named_steps.values():
            if isinstance(step, ColumnTransformer):
                if not hasattr(step, '_name_to_fitted_passthrough'):
                    step._name_to_fitted_passthrough = {}
    return m

model = load_model()

# Extract preprocessor info
preprocessor = model.named_steps['pre']
expected_columns = preprocessor.feature_names_in_
numeric_cols = preprocessor.transformers_[0][2]  # numeric
categorical_cols = preprocessor.transformers_[1][2]  # categorical

# ==========================================
# Streamlit UI
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

# Build DataFrame
input_data = pd.DataFrame([{**user_numeric, **user_categorical}])

# Ensure proper types
for col in numeric_cols:
    if col in input_data.columns:
        input_data[col] = pd.to_numeric(input_data[col], errors='coerce').fillna(0.0)
for col in categorical_cols:
    if col in input_data.columns:
        input_data[col] = input_data[col].astype(str)

# Engineer features
input_data = engineer_features(input_data)

# Reorder columns as expected by model
missing_cols = [col for col in expected_columns if col not in input_data.columns]
for col in missing_cols:
    if col in numeric_cols:
        input_data[col] = 0.0
    else:
        input_data[col] = "unknown"

# Ensure proper column order
input_data = input_data[expected_columns]

# Prediction button
if st.button("🎯 Predict Exam Score"):
    # Safe prediction
    try:
        prediction = model.predict(input_data)[0]
        prediction = max(0, min(100, prediction))  # clip to 0-100

        st.success(f"📘 Predicted Exam Score: {prediction:.2f}")

        st.write("### 📌 Interpretation")
        if prediction >= 75:
            st.write("Excellent performance expected.")
        elif prediction >= 50:
            st.write("Average performance expected.")
        else:
            st.write("Student may require additional academic support.")
    except Exception as e:
        st.error(f"❌ Prediction failed: {e}")

st.markdown("---")
st.caption("Model: Feature-Engineered Ridge Regression | MLDP Project | Streamlit")
