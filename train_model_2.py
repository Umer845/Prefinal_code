import streamlit as st
import pandas as pd
import numpy as np
from catboost import CatBoostRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
import joblib
import os

st.title("🚗 Motor Insurance Premium Prediction - Model Training")

# --- Risk Scoring Function (optional: used for dataset display, not for prediction here) ---
def calculate_risk_score(vehicle_use, vehicle_age, sum_insured, driver_age):
    # Vehicle use scoring
    if vehicle_use.lower() == 'personal':
        vehicleuse_score = 0.2
    elif vehicle_use.lower() == 'commercial':
        vehicleuse_score = 1.0
    else:
        vehicleuse_score = 0.6

    # Vehicle age scoring
    if vehicle_age <= 2:
        vehicleage_score = 0.4
    elif 2 <= vehicle_age <= 5:
        vehicleage_score = 0.6
    elif 6 <= vehicle_age <= 8:
        vehicleage_score = 0.8
    else:
        vehicleage_score = 1.0

    # Sum insured scoring
    if sum_insured <= 300000:
        suminsured_score = 0.2
    elif 300001 <= sum_insured <= 750000:
        suminsured_score = 0.4
    elif 750001 <= sum_insured <= 1500000:
        suminsured_score = 0.6
    elif 1500001 <= sum_insured <= 3000000:
        suminsured_score = 0.8
    else:
        suminsured_score = 1.0

    # Driver age scoring
    if driver_age < 25:
        driverage_score = 1.0
    elif 25 <= driver_age <= 35:
        driverage_score = 0.6
    elif 36 <= driver_age <= 55:
        driverage_score = 0.4
    else:
        driverage_score = 1.0

    # Total score
    raw_score = vehicleuse_score + vehicleage_score + suminsured_score + driverage_score

    # Risk label thresholds
    if 1.2 <= raw_score < 1.8:
        label = "Low"
    elif 1.8 <= raw_score < 2.4:
        label = "Low to Moderate"
    elif 2.4 <= raw_score < 3.0:
        label = "Medium to High"
    else:
        label = "High"

    return raw_score, label



# --- File Upload and Training ---
uploaded_file = st.file_uploader("Upload your Motor Insurance dataset (CSV)", type=["csv"])

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
    df.columns = df.columns.str.strip()

    st.write("### Dataset Preview")
    st.dataframe(df.head())

    target_col = st.selectbox("Select Target Column", df.columns)

    current_year = 2025
    if "VEHICLE MAKE YEAR" in df.columns:
        df["vehicle_age"] = current_year - df["VEHICLE MAKE YEAR"]
    else:
        st.error("❌ 'VEHICLE MAKE YEAR' column not found.")
        st.stop()

    if "DRIVER AGE" not in df.columns:
        st.warning("⚠ 'DRIVER AGE' column missing, filling with default value 30.")
        df["DRIVER AGE"] = 30

    df["risk_percentage"], df["risk_label"] = zip(*df.apply(lambda r:
        calculate_risk_score(
            r.get("VEHICLE USE", "personal"),
            r["vehicle_age"],
            r["SUM INSURED"],
            r["DRIVER AGE"]
        ), axis=1))

    feature_cols = [c for c in df.columns if c != target_col]
    X = df[feature_cols]
    y = df[target_col]

    categorical_cols = X.select_dtypes(include=["object"]).columns.tolist()

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    if st.button("Train Model"):
        model = CatBoostRegressor(
            iterations=5000,
            learning_rate=0.07,
            depth=10,
            loss_function="RMSE",
            eval_metric="RMSE",
            cat_features=categorical_cols,
            random_seed=42,
            verbose=False
        )
        model.fit(X_train, y_train, eval_set=(X_test, y_test), plot=False)

        y_pred = model.predict(X_test)

        r2 = r2_score(y_test, y_pred)
        mae = mean_absolute_error(y_test, y_pred)
        rmse = np.sqrt(mean_squared_error(y_test, y_pred))
        accuracy = 100 * (1 - (abs(y_test - y_pred) / y_test).mean())

        st.subheader("📊 Model Performance")
        st.write(f"**R² Score:** {r2:.4f}")
        st.write(f"**MAE:** {mae:,.2f}")
        st.write(f"**RMSE:** {rmse:,.2f}")
        st.write(f"🎯 **Accuracy:** {accuracy:.2f}%")

        os.makedirs("models", exist_ok=True)
        model.save_model("models/catboost_premium_model_3.pkl")      
        joblib.dump(feature_cols, "models/model_features_3.pkl")
        joblib.dump(categorical_cols, "models/model_cat_features_3.pkl")

        st.success("✅ Model trained and saved in `models/` folder")
