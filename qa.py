# import pandas as pd
# import joblib

# # Load your trained model
# model = joblib.load("model.pkl")  # change path if needed

# # Required feature columns
# feature_cols = [
#     'INSURANCE TYPE', 'VEHICLE TYPE', 'VEHICLE USE', 'VEHICLE MAKE',
#     'VEHICLE MODEL', 'VEHICLE MAKE YEAR', 'SUM INSURED', 'Rate',
#     'vehicle_age', 'DRIVER AGE', 'risk_percentage', 'risk_label'
# ]

# def show():
#     # Example: replace with actual user input collection
#     input_dict = {
#         'INSURANCE TYPE': 'Comprehensive',
#         'VEHICLE TYPE': 'Car',
#         'VEHICLE USE': 'Private',
#         'VEHICLE MAKE': 'Toyota',
#         'VEHICLE MODEL': 'Corolla',
#         'VEHICLE MAKE YEAR': 2018,
#         'SUM INSURED': 15000,
#         'Rate': 3.5,
#         'vehicle_age': 5,
#         'DRIVER AGE': 35,
#         'risk_percentage': 0.15,
#         'risk_label': 'Low'
#     }

#     # Create DataFrame
#     df_temp = pd.DataFrame([input_dict])

#     # Clean column names
#     df_temp.columns = df_temp.columns.str.strip()

#     # Fill missing columns with default values
#     for col in feature_cols:
#         if col not in df_temp.columns:
#             df_temp[col] = None  # or 0 for numeric features

#     # Reorder columns to match model
#     input_df = df_temp[feature_cols]

#     # Make prediction
#     prediction = model.predict(input_df)[0]

#     print("Prediction:", prediction)
#     return prediction

# if __name__ == "__main__":
#     show()


import streamlit as st

def show():
    st.title("QA")
    st.warning("QA logic applied later on")