import streamlit as st
import pandas as pd
import mlflow.pyfunc
from sklearn.preprocessing import LabelEncoder
import numpy as np

# Load the model
MODEL_NAME = "Tourism_Purchase_Predictor"
# Set to 'Production' stage to fetch the latest production version
model_uri = f"models:/{MODEL_NAME}/Production"
# The app will fail if the model is not in the Production stage
try:
    model = mlflow.pyfunc.load_model(model_uri)
except Exception as e:
    st.error(f"Could not load MLflow model '{MODEL_NAME}' in Production stage. Error: {e}")
    model = None

st.set_page_config(layout="wide")
st.title("Travel Package Purchase Predictor")
st.markdown("Enter customer details to predict the likelihood of purchasing the Wellness Tourism Package.")

# --- Define Input Fields ---
col1, col2 = st.columns(2)

with col1:
    age = st.number_input("Age", min_value=18, max_value=80, value=35)
    city_tier = st.selectbox("City Tier (1, 2, or 3)", [1, 2, 3])
    duration_of_pitch = st.number_input("Duration of Pitch (min)", min_value=1.0, max_value=60.0, value=10.0, step=0.1)
    monthly_income = st.number_input("Monthly Income", min_value=5000.0, max_value=100000.0, value=25000.0, step=100.0)
    num_persons = st.number_input("Number of Persons Visiting", min_value=1, max_value=8, value=3)
    num_followups = st.number_input("Number of Follow-ups", min_value=1, max_value=6, value=3)
    num_trips = st.number_input("Number of Trips per Year", min_value=1.0, max_value=25.0, value=3.0, step=0.1)
    num_children = st.number_input("Number of Children Visiting", min_value=0.0, max_value=5.0, value=0.0, step=1.0)
    pitch_satisfaction = st.slider("Pitch Satisfaction Score (1-5)", 1, 5, 3)

with col2:
    typeof_contact = st.selectbox("Type of Contact", ['Company Invited', 'Self Inquiry'])
    occupation = st.selectbox("Occupation", ['Salaried', 'Small Business', 'Free Lancer', 'Large Business', 'Government'])
    gender_raw = st.selectbox("Gender", ['Male', 'Female', 'Other'])
    property_star = st.slider("Preferred Property Star (1-5)", 1.0, 5.0, 3.0, step=1.0)
    marital_status = st.selectbox("Marital Status", ['Single', 'Married', 'Divorced', 'Unmarried'])
    designation = st.selectbox("Designation", ['Manager', 'Executive', 'Senior Manager', 'AVP', 'VP', 'Director'])
    passport = st.selectbox("Passport", [0, 1], format_func=lambda x: 'Yes' if x == 1 else 'No')
    own_car = st.selectbox("Own Car", [0, 1], format_func=lambda x: 'Yes' if x == 1 else 'No')
    product_pitched = st.selectbox("Product Pitched", ['Deluxe', 'Basic', 'Standard', 'Super Deluxe', 'King'])


# --- Data Transformation (Matches data_prep.py logic) ---
if st.button("Predict Purchase") and model is not None:

    # 1. Create initial DataFrame
    input_data = pd.DataFrame([{
        'Age': age, 'CityTier': city_tier, 'DurationOfPitch': duration_of_pitch,
        'MonthlyIncome': monthly_income, 'NumberOfPersonVisiting': num_persons,
        'NumberOfFollowups': num_followups, 'NumberOfTrips': num_trips,
        'NumberOfChildrenVisiting': num_children, 'PitchSatisfactionScore': pitch_satisfaction,
        'TypeofContact': typeof_contact, 'Occupation': occupation, 'Gender': gender_raw,
        'PreferredPropertyStar': property_star, 'MaritalStatus': marital_status,
        'Designation': designation, 'Passport': passport, 'OwnCar': own_car,
        'ProductPitched': product_pitched
    }])

    # 2. Label Encoding for Gender: Fit on all possibilities including 'Other'
    gender_encoder = LabelEncoder()
    gender_encoder.fit(['Male', 'Female', 'Other'])
    input_data['Gender'] = gender_encoder.transform(input_data['Gender'])

    # 3. One-Hot Encoding
    categorical_cols_ohe = [
        'Occupation', 'Designation', 'MaritalStatus', 'ProductPitched', 'TypeofContact'
    ]
    input_data_ohe = pd.get_dummies(input_data, columns=categorical_cols_ohe, drop_first=True)

    # 4. Column Alignment (CRITICAL: Must match the features of Xtrain used during model training)
    try:
        # Get the feature names from the model's signature (best practice)
        model_feature_names = [col['name'] for col in model.metadata.get_model_input_schema().to_dict()['inputs']]

        # Filter and reorder the input data to match the model's feature list
        final_input_df = input_data_ohe.reindex(columns=model_feature_names, fill_value=0)

        # Prediction
        prediction_proba = model.predict(final_input_df)[0]
        # Use a simple 0.5 threshold for final prediction
        prediction = 1 if prediction_proba >= 0.5 else 0

        result = "Will likely purchase the package" if prediction == 1 else "Will likely NOT purchase the package"

        st.subheader("Prediction Result:")
        if prediction == 1:
            st.success(f"The model predicts: **{result}** (Likelihood: {prediction_proba:.2f})")
        else:
            st.info(f"The model predicts: **{result}** (Likelihood: {prediction_proba:.2f})")

    except Exception as e:
        st.error(f"An error occurred during prediction: {e}")
        st.warning("Prediction failed. This is often due to a feature mismatch between the application and the trained model.")
