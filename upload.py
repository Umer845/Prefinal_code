import streamlit as st
import pandas as pd
import psycopg2
from psycopg2 import sql
from datetime import datetime

# Database connection
DB_HOST = "localhost"
DB_NAME = "AutoMotor_Insurance"
DB_USER = "postgres"
DB_PASS = "United2025"
DB_PORT = 5432

def connect_db():
    try:
        conn = psycopg2.connect(
            host=DB_HOST,
            database=DB_NAME,
            user=DB_USER,
            password=DB_PASS,
            port=DB_PORT
        )
        return conn
    except Exception as e:
        st.error(f"Database connection failed: {e}")
        return None

def upload_to_db(df, table_name="motor_data"):
    conn = connect_db()
    if conn is None:
        return
    try:
        cur = conn.cursor()

        # Lowercase column names
        df.columns = [col.lower().replace(" ", "_") for col in df.columns]

        # Get table info
        cur.execute(f"SELECT column_name, data_type FROM information_schema.columns WHERE table_name = '{table_name}'")
        table_info = cur.fetchall()
        table_columns = [col[0] for col in table_info]
        table_types = {col[0]: col[1] for col in table_info}

        # Remove 'id' column (auto-increment)
        table_columns = [col for col in table_columns if col != "id"]

        # Keep only columns that exist in table
        df = df[[col for col in df.columns if col in table_columns]]

        # Fill missing columns with None
        for col in table_columns:
            if col not in df.columns:
                df[col] = None

        # Clean numeric/integer columns
        numeric_cols = ["vehicle_make_year", "sum_insured", "policy_premium", "no_of_claims", "rate"]
        for col in numeric_cols:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce')

        # Insert into DB
        cols = df.columns
        for i, row in df.iterrows():
            values = [None if pd.isna(x) else x for x in row]
            insert_query = sql.SQL("INSERT INTO {} ({}) VALUES ({})").format(
                sql.Identifier(table_name),
                sql.SQL(', ').join(map(sql.Identifier, cols)),
                sql.SQL(', ').join(sql.Placeholder() * len(cols))
            )
            cur.execute(insert_query, values)

        conn.commit()
        cur.close()
        conn.close()
        st.success("Data uploaded successfully!")

    except Exception as e:
        st.error(f"Error uploading data: {e}")

def show():
    st.title("Upload Motor Insurance Data")
    st.write("Upload CSV, XLS, or XLSX files to store in the database.")

    uploaded_file = st.file_uploader("Choose a file", type=["csv", "xls", "xlsx"])

    if uploaded_file is not None:
        try:
            if uploaded_file.name.endswith(".csv"):
                df = pd.read_csv(uploaded_file)
            else:
                df = pd.read_excel(uploaded_file)

            # Ensure vehicle_make and vehicle_model (sub-make) are not null
            if "vehicle_make" in df.columns:
                df["vehicle_make"] = df["vehicle_make"].fillna("UNKNOWN")
            if "vehicle_model" in df.columns:
                df["vehicle_model"] = df["vehicle_model"].fillna("UNKNOWN")

            st.dataframe(df.head())
            if st.button("Upload to Database"):
                upload_to_db(df)
        except Exception as e:
            st.error(f"Error reading file: {e}")

