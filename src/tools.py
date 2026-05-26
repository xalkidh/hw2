import pickle
import numpy as np
import pandas as pd
from langchain.tools import tool

# Φόρτωση μοντέλου και scaler
with open("models/best_model.pkl", "rb") as f:
    model = pickle.load(f)

with open("models/scaler.pkl", "rb") as f:
    scaler = pickle.load(f)

@tool
def predict_dropout(input_data: str) -> str:
    """
    Προβλέπει αν ένας φοιτητής θα εγκαταλείψει τις σπουδές του.
    Το input πρέπει να είναι JSON string με τα εξής πεδία:
    Age, Gender, Year_of_Study, GPA, Attendance_Rate, 
    Study_Hours_per_Day, Stress_Index, Financial_Support,
    Part_Time_Job, Family_Income_Level, Internet_Access
    """
    import json

    try:
        data = json.loads(input_data)
        df = pd.DataFrame([data])

        # Feature engineering (ιδιο με το HW1)
        df["study_attendance_ratio"] = df["Study_Hours_per_Day"] * df["Attendance_Rate"] / 100
        df["stress_gpa_ratio"] = df["Stress_Index"] / (df["GPA"] + 0.1)

        # Categorical και numerical features
        cat_features = df.select_dtypes(exclude=["number"]).columns.tolist()
        num_features = df.select_dtypes(include=["number"]).columns.tolist()

        # One-hot encoding
        df = pd.get_dummies(df, columns=cat_features, drop_first=True)

        # Align columns με το scaler
        expected_features = scaler.feature_names_in_
        for col in expected_features:
            if col not in df.columns:
                df[col] = 0
        df = df[expected_features]

        # Scaling
        X_scaled = scaler.transform(df)

        # Prediction
        prediction = model.predict(X_scaled)[0]
        probability = model.predict_proba(X_scaled)[0][1]

        label = "Dropout" if prediction == 1 else "No Dropout"
        return f"Prediction: {label} (probability of dropout: {probability:.1%})"

    except Exception as e:
        return f"Error: {str(e)}"

@tool
def retrieve_information(query: str) -> str:
    """
    Ανακτά πληροφορίες από τη βάση γνώσης για ερωτήσεις σχετικές 
    με το student dropout.
    """
    from src.rag import retrieve
    return retrieve(query)