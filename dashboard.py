import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import psycopg2

# PostgreSQL connection config
DB_CONFIG = {
    "dbname": "AutoMotor_Insurance",
    "user": "postgres",
    "password": "United2025",
    "host": "localhost",
    "port": "5432"
}

def fetch_dashboard_data():
    """Fetch premium and risk data from database"""
    try:
        conn = psycopg2.connect(**DB_CONFIG)
        
        # --- Monthly Premium ---
        query_premium = """
            SELECT 
                EXTRACT(MONTH FROM created_at) AS month,
                SUM(predicted_premium) AS total_premium,
                SUM(sum_insured) AS total_sum_insured
            FROM premium_results
            GROUP BY month
            ORDER BY month
        """
        premium_df = pd.read_sql(query_premium, conn)

        # --- Risk profile counts ---
        query_risk = """
            SELECT 
                risk_label,
                COUNT(*) AS count
            FROM risk_profile_results
            GROUP BY risk_label
        """
        risk_df = pd.read_sql(query_risk, conn)

        # --- Premium type distribution (using vehicle_use from risk_profile_results) ---
        query_type = """
            SELECT 
                vehicle_use AS premium_type,
                COUNT(*) AS count
            FROM risk_profile_results
            GROUP BY vehicle_use
        """
        type_df = pd.read_sql(query_type, conn)

        # --- Average premium rate by age group ---
        query_age = """
            SELECT 
                CASE 
                    WHEN driver_age BETWEEN 18 AND 25 THEN '18-25'
                    WHEN driver_age BETWEEN 26 AND 35 THEN '26-35'
                    WHEN driver_age BETWEEN 36 AND 50 THEN '36-50'
                    ELSE '51+' 
                END AS age_group,
                AVG(p.actual_premium_rate) AS avg_rate
            FROM premium_results p
            JOIN risk_profile_results r
              ON EXTRACT(YEAR FROM p.created_at) = EXTRACT(YEAR FROM r.created_at)
            GROUP BY age_group
            ORDER BY age_group
        """
        age_df = pd.read_sql(query_age, conn)

        conn.close()
        return premium_df, risk_df, type_df, age_df
    except Exception as e:
        st.error(f"❌ Database fetch error: {e}")
        return None, None, None, None


def show():
    st.title("🚗 Motor Insurance Dashboard")
    
    # Fetch data
    premium_df, risk_df, type_df, age_df = fetch_dashboard_data()
    if premium_df is None:
        return

    # Top KPIs
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Avg Premium", f"{premium_df['total_premium'].mean():,.0f}")
    col2.metric("Total Policies", f"{premium_df['total_sum_insured'].count()}")
    col3.metric("Claims Ratio", "28%", delta="-2%")  # placeholder
    col4.metric("Loss Ratio", "35%", delta="+1%")    # placeholder

    # Monthly Premium trend
    fig1 = go.Figure()
    fig1.add_trace(go.Scatter(
        x=premium_df["month"], y=premium_df["total_premium"],
        mode='lines+markers', name='Total Premium'
    ))
    fig1.update_layout(title="Monthly Premium Trend", template="plotly_dark")
    st.plotly_chart(fig1, use_container_width=True)

    # Pie charts
    col1, col2 = st.columns(2)

    if not risk_df.empty:
        fig2 = px.pie(
            risk_df, names='risk_label', values='count',
            title="Risk Profile Distribution",
            template="plotly_dark",
            color_discrete_sequence=px.colors.sequential.Blues
        )
        col1.plotly_chart(fig2, use_container_width=True)

    if not type_df.empty:
        fig3 = px.pie(
            type_df, names='premium_type', values='count',
            title="Premium Type Distribution",
            template="plotly_dark",
            color_discrete_sequence=px.colors.sequential.RdBu
        )
        col2.plotly_chart(fig3, use_container_width=True)

    if not age_df.empty:
        fig4 = px.bar(
            age_df, x='age_group', y='avg_rate',
            title="Average Premium Rate by Age Group",
            template="plotly_dark",
            labels={"age_group": "Age Group", "avg_rate": "Premium Rate"}
        )
        st.plotly_chart(fig4, use_container_width=True)


if __name__ == "__main__":
    st.set_page_config(layout="wide")
    show()
