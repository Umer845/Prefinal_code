import streamlit as st
import pandas as pd
import joblib
import psycopg2
from datetime import datetime

# Premium rate multiplier
PREMIUM_RATE_MULTIPLIER = 1.25  # 25% increase

# Database connection settings
DB_CONFIG = {
    "dbname": "AutoMotor_Insurance",
    "user": "postgres",
    "password": "United2025",
    "host": "localhost",
    "port": "5432"
}

def insert_into_db(vehicle_make, vehicle_model, vehicle_make_year, sum_insured,
                   annual_premium, min_rate, actual_rate, max_rate):
    try:
        conn = psycopg2.connect(**DB_CONFIG)
        cur = conn.cursor()

        # Convert all NumPy types to Python native types
        vehicle_make = str(vehicle_make)
        vehicle_model = str(vehicle_model)
        vehicle_make_year = int(vehicle_make_year)
        sum_insured = float(sum_insured)
        annual_premium = float(annual_premium)
        min_rate = float(min_rate)
        actual_rate = float(actual_rate)
        max_rate = float(max_rate)

        cur.execute("""
            INSERT INTO premium_results 
            (vehicle_make, vehicle_model, vehicle_make_year, sum_insured, predicted_premium, min_premium_rate, actual_premium_rate, max_premium_rate)
            VALUES (%s,%s,%s,%s,%s,%s,%s,%s)
        """, (vehicle_make, vehicle_model, vehicle_make_year, sum_insured,
              annual_premium, min_rate, actual_rate, max_rate))

        conn.commit()
        cur.close()
        conn.close()
        return True
    except Exception as e:
        st.error(f"❌ Database insert error: {e}")
        return False

def show():
    st.title("🚗 Motor Insurance Annual Premium Prediction")

    # Load trained model & features
    try:
        model = joblib.load("models/catboost_premium_model_4.pkl")
        feature_cols = joblib.load("models/model_features_4.pkl")
        categorical_cols = joblib.load("models/model_cat_features_4.pkl")
    except FileNotFoundError as e:
        model = None
        st.error(f"❌ Model not found: {e}")

    if model:
        st.subheader("Enter Vehicle Details")

        with st.form(key="predict_form"):
            vehicle_make = st.text_input("Vehicle Make", value="Toyota").upper()
            vehicle_sub_make = st.text_input("Vehicle Model", value="Corolla").upper()
            vehicle_make_year = st.number_input(
                "Vehicle Make Year", min_value=1980, max_value=datetime.now().year, value=2020
            )
            sum_insured = st.number_input(
                "Sum Insured", min_value=10000, value=500000
            )

            submit = st.form_submit_button("Predict Premium")

        if submit:
            vehicle_age = datetime.now().year - vehicle_make_year

            # Prepare input dict
            input_dict = {
                "VEHICLE MAKE": vehicle_make,
                "VEHICLE MODEL": vehicle_sub_make,
                "VEHICLE MAKE YEAR": vehicle_make_year,
                "SUM INSURED": sum_insured,
                "vehicle_age": vehicle_age
            }

            # Fill missing columns
            for col in feature_cols:
                if col not in input_dict:
                    input_dict[col] = 0 if col not in categorical_cols else ""

            input_df = pd.DataFrame([input_dict])[feature_cols]

            # Predict monthly premium
            monthly_premium = model.predict(input_df)[0]

            # Calculate annual premium
            annual_premium = monthly_premium * 11

            # Calculate rates
            actual_rate = (annual_premium / sum_insured * 100) if sum_insured != 0 else 0
            min_rate = actual_rate * 0.90
            max_rate = actual_rate * PREMIUM_RATE_MULTIPLIER

            # Display results
            st.markdown(f"<div style='background-color:#2196F3; padding:15px; border-radius:8px; margin-bottom:10px;'><h4 style='color:white; margin:0;'>Monthly Premium: {monthly_premium:,.2f}</h4></div>", unsafe_allow_html=True)
            st.markdown(f"<div style='background-color:#4CAF50; padding:15px; border-radius:8px; margin-bottom:10px;'><h4 style='color:white; margin:0;'>Annual Premium: {annual_premium:,.2f}</h4></div>", unsafe_allow_html=True)
            st.markdown(f"<div style='background-color:#ae25c9; padding:15px; border-radius:8px; margin-bottom:10px;'><h4 style='color:white; margin:0;'>Actual Premium Rate: {actual_rate:.2f}%</h4></div>", unsafe_allow_html=True)
            st.markdown(f"<div style='background-color:#FF9800; padding:15px; border-radius:8px; margin-bottom:10px;'><h4 style='color:white; margin:0;'>Minimum Rate: {min_rate:.2f}%</h4></div>", unsafe_allow_html=True)
            st.markdown(f"<div style='background-color:#F44336; padding:15px; border-radius:8px; margin-bottom:10px;'><h4 style='color:white; margin:0;'>Maximum Rate: {max_rate:.2f}%</h4></div>", unsafe_allow_html=True)

            # Store in DB
            success = insert_into_db(vehicle_make, vehicle_sub_make, vehicle_make_year, sum_insured,
                                     annual_premium, min_rate, actual_rate, max_rate)
            if success:
                st.success("✅ Data stored in database successfully.")

    else:
        st.info("⚠️ Train the model first by uploading dataset and clicking **Train Model**.")

if __name__ == "__main__":
    show()
