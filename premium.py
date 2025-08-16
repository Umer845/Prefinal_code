import streamlit as st
import pandas as pd
import joblib
import psycopg2
from datetime import datetime

# Database connection
DB_CONFIG = {
    "dbname": "AutoMotor_Insurance",
    "user": "postgres",
    "password": "United2025",
    "host": "localhost",
    "port": "5432"
}

# Premium rate multiplier
PREMIUM_RATE_MULTIPLIER = 1.25  # Maximum allowed increase

def fetch_min_max_rate(vehicle_make, vehicle_model):
    """Fetch minimum and maximum premium rates from DB for given make & sub-make"""
    try:
        conn = psycopg2.connect(**DB_CONFIG)
        cur = conn.cursor()
        cur.execute("""
            SELECT MIN(actual_premium_rate), MAX(actual_premium_rate)
            FROM premium_results
            WHERE vehicle_make = %s AND vehicle_model = %s
        """, (vehicle_make, vehicle_model))
        result = cur.fetchone()
        cur.close()
        conn.close()
        if result and result[0] is not None:
            return float(result[0]), float(result[1])
        else:
            return None, None
    except Exception as e:
        st.error(f"❌ Error fetching min/max rates: {e}")
        return None, None

def insert_into_db(vehicle_make, vehicle_model, vehicle_make_year, sum_insured,
                   annual_premium, min_rate, actual_rate, max_rate):
    """Insert predicted premium data into DB"""
    try:
        conn = psycopg2.connect(**DB_CONFIG)
        cur = conn.cursor()
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
    st.title("🚗 Motor Insurance Premium Prediction")

    # Load model & features
    try:
        model = joblib.load("models/catboost_premium_model_4.pkl")
        feature_cols = joblib.load("models/model_features_4.pkl")
        categorical_cols = joblib.load("models/model_cat_features_4.pkl")
    except FileNotFoundError as e:
        st.error(f"❌ Model not found: {e}")
        return

    st.subheader("Enter Vehicle Details")
    with st.form(key="premium_form"):
        vehicle_make = st.text_input("Vehicle Make", value="TOYOTA").upper()
        vehicle_sub_make = st.text_input("Vehicle Sub Make", value="COROLLA").upper()
        vehicle_make_year = st.number_input("Vehicle Make Year", min_value=1980, max_value=datetime.now().year, value=2020)
        sum_insured = st.number_input("Sum Insured", min_value=10000, value=500000)
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

        # Fill missing feature columns
        for col in feature_cols:
            if col not in input_dict:
                input_dict[col] = 0 if col not in categorical_cols else ""

        input_df = pd.DataFrame([input_dict])[feature_cols]

        # Predict premium
        monthly_premium = model.predict(input_df)[0]
        annual_premium = monthly_premium * 12
        actual_rate = (annual_premium / sum_insured * 100) if sum_insured != 0 else 0

        # Fetch min/max rate from DB
        min_rate_db, max_rate_db = fetch_min_max_rate(vehicle_make, vehicle_sub_make)
        min_rate = min_rate_db if min_rate_db is not None else actual_rate * 0.90
        max_rate = max_rate_db if max_rate_db is not None else actual_rate * PREMIUM_RATE_MULTIPLIER

        # Display results
        st.markdown(f"<div style='background-color:#2196F3; padding:10px; border-radius:8px;'><b>Monthly Premium:</b> {monthly_premium:,.2f}</div>", unsafe_allow_html=True)
        st.markdown(f"<div style='background-color:#4CAF50; padding:10px; border-radius:8px;'><b>Annual Premium:</b> {annual_premium:,.2f}</div>", unsafe_allow_html=True)
        st.markdown(f"<div style='background-color:#ae25c9; padding:10px; border-radius:8px;'><b>Actual Premium Rate:</b> {actual_rate:.2f}%</div>", unsafe_allow_html=True)
        st.markdown(f"<div style='background-color:#FF9800; padding:10px; border-radius:8px;'><b>Minimum Rate:</b> {min_rate:.2f}%</div>", unsafe_allow_html=True)
        st.markdown(f"<div style='background-color:#F44336; padding:10px; border-radius:8px;'><b>Maximum Rate:</b> {max_rate:.2f}%</div>", unsafe_allow_html=True)

        # Store in DB
        insert_into_db(vehicle_make, vehicle_sub_make, vehicle_make_year, sum_insured,
                       annual_premium, min_rate, actual_rate, max_rate)

if __name__ == "__main__":
    show()
