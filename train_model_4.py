import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
import joblib
import os
from catboost import CatBoostRegressor

st.title("🚗 Motor Insurance Premium Prediction - Model Training (CatBoost)")

# --- Risk Scoring Function ---
def calculate_risk_score(vehicle_use, vehicle_age, sum_insured, driver_age):
    if str(vehicle_use).lower() == 'personal':
        vehicleuse_score = 0.2
    elif str(vehicle_use).lower() == 'commercial':
        vehicleuse_score = 1.0
    else:
        vehicleuse_score = 0.6

    if vehicle_age <= 2:
        vehicleage_score = 0.4
    elif 2 <= vehicle_age <= 5:
        vehicleage_score = 0.6
    elif 6 <= vehicle_age <= 8:
        vehicleage_score = 0.8
    else:
        vehicleage_score = 1.0

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

    if driver_age < 25:
        driverage_score = 1.0
    elif 25 <= driver_age <= 35:
        driverage_score = 0.6
    elif 36 <= driver_age <= 55:
        driverage_score = 0.4
    else:
        driverage_score = 1.0

    raw_score = vehicleuse_score + vehicleage_score + suminsured_score + driverage_score

    if 1.2 <= raw_score < 1.8:
        label = "Low"
    elif 1.8 <= raw_score < 2.4:
        label = "Low to Moderate"
    elif 2.4 <= raw_score < 3.0:
        label = "Medium to High"
    else:
        label = "High"

    return raw_score, label


# --- File Upload ---
uploaded_file = st.file_uploader(
    "Upload your Motor Insurance dataset (CSV, XLS, XLSX)",
    type=["csv", "xls", "xlsx"]
)

if uploaded_file is not None:
    file_ext = uploaded_file.name.split(".")[-1].lower()
    if file_ext == "csv":
        df = pd.read_csv(uploaded_file)
    elif file_ext in ["xls", "xlsx"]:
        df = pd.read_excel(uploaded_file)
    else:
        st.error("❌ Unsupported file format.")
        st.stop()

    df.columns = df.columns.str.strip()
    st.write("### Dataset Preview")
    st.dataframe(df.head())

    target_col = st.selectbox("Select Target Column", df.columns)

    # --- Target Cleaning ---
    df[target_col] = pd.to_numeric(df[target_col], errors="coerce")
    df = df.dropna(subset=[target_col])
    if df.empty:
        st.error(f"❌ No rows left after cleaning target column '{target_col}'.")
        st.stop()

    # --- Vehicle Age ---
    current_year = 2025
    if "VEHICLE MAKE YEAR" in df.columns:
        df["VEHICLE MAKE YEAR"] = pd.to_numeric(df["VEHICLE MAKE YEAR"], errors="coerce")
        df["VEHICLE MAKE YEAR"].fillna(current_year, inplace=True)
        df["vehicle_age"] = current_year - df["VEHICLE MAKE YEAR"]
    else:
        st.error("❌ 'VEHICLE MAKE YEAR' column not found.")
        st.stop()

    # --- Driver Age ---
    if "DRIVER AGE" not in df.columns:
        st.warning("⚠ 'DRIVER AGE' missing, filling with 30.")
        df["DRIVER AGE"] = 30
    else:
        df["DRIVER AGE"] = pd.to_numeric(df["DRIVER AGE"], errors="coerce").fillna(30)

    # --- Sum Insured ---
    if "SUM INSURED" not in df.columns:
        st.error("❌ 'SUM INSURED' column not found.")
        st.stop()
    else:
        df["SUM INSURED"] = pd.to_numeric(df["SUM INSURED"], errors="coerce").fillna(0)

    # --- Risk Scores ---
    df["risk_percentage"], df["risk_label"] = zip(*df.apply(lambda r:
        calculate_risk_score(
            r.get("VEHICLE USE", "personal"),
            r["vehicle_age"],
            r["SUM INSURED"],
            r["DRIVER AGE"]
        ), axis=1))

    # --- Features & Target ---
    feature_cols = [c for c in df.columns if c != target_col]
    X = df[feature_cols]
    y = df[target_col]

    # --- Handle NaNs ---
    categorical_cols = X.select_dtypes(include=["object"]).columns.tolist()
    for col in categorical_cols:
        X[col] = X[col].astype(str).fillna("Unknown")

    numeric_cols = X.select_dtypes(include=[np.number]).columns.tolist()
    for col in numeric_cols:
        X[col] = X[col].fillna(X[col].median())

    # --- Train-test Split ---
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # --- Train Model ---
    if st.button("Train Model"):
        if y_train.var() == 0 or y_test.var() == 0:
            st.error("⚠ Target has zero variance — R² score is undefined. Please use a dataset with varying target values.")
            st.stop()

        model = CatBoostRegressor(
            iterations=1000,
            learning_rate=0.05,
            depth=8,
            cat_features=categorical_cols,
            verbose=0,
            random_state=42
        )

        model.fit(X_train, y_train)

        y_pred = model.predict(X_test)

        # --- Metrics ---
        r2 = r2_score(y_test, y_pred)
        mae = mean_absolute_error(y_test, y_pred)
        rmse = np.sqrt(mean_squared_error(y_test, y_pred))

        # Mean Absolute Percentage Error (MAPE)
        mape = np.mean(np.abs((y_test - y_pred) / y_test.replace(0, np.nan))) * 100
        accuracy = max(0, 100 - mape)  # Ensure accuracy is not negative

        st.subheader("📊 Model Performance")
        st.write(f"**R² Score:** {r2:.4f}")
        st.write(f"**MAE:** {mae:,.2f}")
        st.write(f"**RMSE:** {rmse:,.2f}")
        st.write(f"🎯 **Accuracy:** {accuracy:.2f}%")


        # --- Save Model & Metadata ---
        os.makedirs("models", exist_ok=True)
        joblib.dump(model, "models/catboost_premium_model_4.pkl")
        joblib.dump(X.columns.tolist(), "models/model_features_4.pkl")
        joblib.dump(categorical_cols, "models/model_cat_features_4.pkl")

        st.success("✅ CatBoost model trained and saved in `models/` folder")
