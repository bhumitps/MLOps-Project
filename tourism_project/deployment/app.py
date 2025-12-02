
"""Streamlit app for Tourism Package Purchase Prediction."""

import os
import streamlit as st
import pandas as pd
import joblib
from huggingface_hub import hf_hub_download

st.set_page_config(page_title="Tourism Package Purchase Predictor", layout="centered")

st.title("Tourism Package Purchase Prediction App")
st.write(
    """
This app predicts whether a customer is likely to **purchase the tourism package** (`ProdTaken`)
based on their profile and interaction details.
    """
)

@st.cache_resource
def load_model():
    """Download and load the trained tourism model from Hugging Face Hub."""
    repo_id = "bhumitps/MLops"
    filename = "best_tourism_model_v1.joblib"
    model_path = hf_hub_download(
        repo_id=repo_id,
        filename=filename,
        repo_type="model",
        token=os.getenv("HF_TOKEN"),
    )
    model = joblib.load(model_path)
    return model

model = load_model()

st.sidebar.header("Customer Information")

# Collect user inputs
age = st.sidebar.number_input("Age", min_value=18, max_value=80, value=35)

typeofcontact = st.sidebar.selectbox(
    "Type of Contact",
    options=["Self Enquiry", "Company Invited"],
)

citytier = st.sidebar.selectbox("City Tier", options=[1, 2, 3], index=1)

duration_of_pitch = st.sidebar.number_input(
    "Duration of Pitch (minutes)", min_value=0, max_value=60, value=10
)

occupation = st.sidebar.selectbox(
    "Occupation",
    options=["Salaried", "Free Lancer", "Small Business", "Large Business"],
)

gender = st.sidebar.selectbox("Gender", options=["Male", "Female", "Fe Male"])

num_person_visiting = st.sidebar.selectbox(
    "Number of Persons Visiting", options=[1, 2, 3, 4, 5], index=2
)

num_followups = st.sidebar.number_input(
    "Number of Follow-ups", min_value=0, max_value=10, value=3
)

product_pitched = st.sidebar.selectbox(
    "Product Pitched",
    options=["Basic", "Standard", "Deluxe", "Super Deluxe", "King"],
)

preferred_property_star = st.sidebar.selectbox(
    "Preferred Property Star", options=[3.0, 4.0, 5.0], index=0
)

marital_status = st.sidebar.selectbox(
    "Marital Status", options=["Single", "Married", "Divorced", "Unmarried"]
)

number_of_trips = st.sidebar.number_input(
    "Number of Trips (per year)", min_value=0, max_value=30, value=2
)

passport = st.sidebar.selectbox("Passport", options=[0, 1], index=1)

pitch_satisfaction_score = st.sidebar.selectbox(
    "Pitch Satisfaction Score", options=[1, 2, 3, 4, 5], index=2
)

own_car = st.sidebar.selectbox("Own Car", options=[0, 1], index=1)

number_of_children_visiting = st.sidebar.selectbox(
    "Number of Children Visiting", options=[0, 1, 2, 3], index=0
)

designation = st.sidebar.selectbox(
    "Designation",
    options=["Executive", "Manager", "Senior Manager", "AVP", "VP"],
)

monthly_income = st.sidebar.number_input(
    "Monthly Income", min_value=1000.0, max_value=1000000.0, value=20000.0, step=500.0
)

# Create input DataFrame matching training features
input_data = pd.DataFrame(
    [
        {
            "Age": age,
            "TypeofContact": typeofcontact,
            "CityTier": citytier,
            "DurationOfPitch": duration_of_pitch,
            "Occupation": occupation,
            "Gender": gender,
            "NumberOfPersonVisiting": num_person_visiting,
            "NumberOfFollowups": num_followups,
            "ProductPitched": product_pitched,
            "PreferredPropertyStar": preferred_property_star,
            "MaritalStatus": marital_status,
            "NumberOfTrips": number_of_trips,
            "Passport": passport,
            "PitchSatisfactionScore": pitch_satisfaction_score,
            "OwnCar": own_car,
            "NumberOfChildrenVisiting": number_of_children_visiting,
            "Designation": designation,
            "MonthlyIncome": monthly_income,
        }
    ]
)

st.subheader("Input Summary")
st.write(input_data)

if st.button("Predict Package Purchase"):
    prediction = model.predict(input_data)[0]
    prob = model.predict_proba(input_data)[0, 1]

    if prediction == 1:
        st.success(f"The model predicts that the customer is **likely to purchase** the package. (Probability: {{prob:.2%}})")
    else:
        st.info(f"The model predicts that the customer is **unlikely to purchase** the package. (Probability: {{prob:.2%}})")

st.caption("Model source: Hugging Face Hub repo `bhumitps/MLops`.")
