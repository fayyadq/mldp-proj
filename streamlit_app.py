import streamlit as st
import pandas as pd
import joblib

# Load your trained model (Pipeline with FunctionTransformer, preprocessor, and Ridge/LinearRegression)
model = joblib.load("exam_model_champion.pkl")

# --- USER INPUT SECTION ---
st.header("Predict Exam Score")

# Example: numerical inputs
hours_studied = st.number_input("Hours Studied", min_value=0.0, max_value=24.0, value=5.0)
attendance_rate = st.number_input("Attendance Rate (%)", min_value=0.0, max_value=100.0, value=80.0)
sleep_hours = st.number_input("Sleep Hours", min_value=0.0, max_value=24.0, value=7.0)

# Example: categorical inputs (replace with your real categories)
# For instance, "study_program" could be "Science", "Arts", "Commerce"
study_program = st.selectbox("Study Program", ["Science", "Arts", "Commerce"])

# Step 1: Build input DataFrame with all columns your model expects
input_dict = {
    "hours_studied": [hours_studied],
    "attendance_rate": [attendance_rate],
    "sleep_hours": [sleep_hours],
    "study_program": [study_program]
    # Add other features here if your model needs them
}

input_data = pd.DataFrame(input_dict)

# Step 2: Apply the same feature engineering function your model uses
def engineer_features(X_in):
    X_out = X_in.copy()
    if "hours_studied" in X_out.columns and "attendance_rate" in X_out.columns:
        X_out["study_attendance_interaction"] = X_out["hours_studied"] * X_out["attendance_rate"]
    if "hours_studied" in X_out.columns and "sleep_hours" in X_out.columns:
        X_out["study_sleep_ratio"] = X_out["hours_studied"] / (X_out["sleep_hours"] + 1)
    return X_out

input_data = engineer_features(input_data)

# --- PREDICTION ---
if st.button("Predict Exam Score"):
    prediction = model.predict(input_data)[0]
    st.success(f"Predicted Exam Score: {prediction:.2f}")
