import streamlit as st
import pandas as pd
import numpy as np

st.set_page_config(page_title="BI Dashboard - IntelliStream", layout="wide")

# CSS Styling
st.markdown("""
<style>
body {
    background-color: #12121f;
    color: white;
    font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
}
.header {
    text-align: center;
    margin-bottom: 1rem;
}
.banner-image {
    width: 100%;
    max-height: 250px;
    object-fit: cover;
    border-radius: 15px;
    margin-bottom: 1.5rem;
    box-shadow: 0 8px 20px rgba(0,0,0,0.4);
}
.kpi-card {
    background-color: #1f1f2e;
    padding: 1.5rem;
    border-radius: 15px;
    box-shadow: 0 6px 18px rgba(0,0,0,0.3);
    text-align: center;
    color: #fff;
    transition: transform 0.3s ease-in-out;
    cursor: default;
}
.kpi-card:hover {
    transform: scale(1.05);
    box-shadow: 0 10px 28px rgba(108, 99, 255, 0.6);
}
.kpi-icon {
    font-size: 40px;
    margin-bottom: 10px;
    color: #6c63ff;
}
.kpi-value {
    font-size: 36px;
    font-weight: 700;
    margin-bottom: 5px;
}
.kpi-label {
    font-size: 18px;
    opacity: 0.8;
}
.data-table {
    background-color: #1f1f2e;
    border-radius: 15px;
    padding: 1rem;
    color: white !important;
}
</style>
""", unsafe_allow_html=True)

# Header with Banner Image
st.markdown('<div class="header">', unsafe_allow_html=True)
st.title("📊 IntelliStream BI Dashboard")
st.image("https://images.unsplash.com/photo-1504384308090-c894fdcc538d?auto=format&fit=crop&w=1350&q=80", caption="Data Insights & Analytics", use_column_width=True, output_format="auto", clamp=True)
st.markdown('</div>', unsafe_allow_html=True)

# Sample KPIs
kpi1, kpi2, kpi3 = st.columns(3)

with kpi1:
    st.markdown("""
    <div class="kpi-card">
        <div class="kpi-icon">📈</div>
        <div class="kpi-value">1,245</div>
        <div class="kpi-label">Monthly Active Users</div>
    </div>
    """, unsafe_allow_html=True)

with kpi2:
    st.markdown("""
    <div class="kpi-card">
        <div class="kpi-icon">💰</div>
        <div class="kpi-value">$75,320</div>
        <div class="kpi-label">Monthly Revenue</div>
    </div>
    """, unsafe_allow_html=True)

with kpi3:
    st.markdown("""
    <div class="kpi-card">
        <div class="kpi-icon">🎯</div>
        <div class="kpi-value">87%</div>
        <div class="kpi-label">Goal Completion Rate</div>
    </div>
    """, unsafe_allow_html=True)

st.markdown("---")

# Generate sample data for charts
np.random.seed(42)
dates = pd.date_range(start='2025-01-01', periods=30)
user_growth = np.cumsum(np.random.randint(20, 100, size=30))
revenue = np.cumsum(np.random.randint(1000, 5000, size=30))

df_line = pd.DataFrame({
    'Date': dates,
    'Active Users': user_growth,
    'Revenue ($)': revenue
})

df_bar = pd.DataFrame({
    'Category': ['OTT Subscriptions', 'Finance Integrations', 'Script Predictions', 'Churn Predictions', 'Recommender Usage'],
    'Count': [1245, 860, 430, 580, 970]
})

# Charts layout
chart1, chart2 = st.columns(2)

with chart1:
    st.subheader("Active Users & Revenue Over Time")
    st.line_chart(df_line.set_index('Date'))

with chart2:
    st.subheader("Feature Usage Counts")
    st.bar_chart(df_bar.set_index('Category'))

st.markdown("---")

# Data Table
st.subheader("Detailed Monthly Data")
data_table = pd.DataFrame({
    "Metric": ["Active Users", "Revenue ($)", "Goal Completion %", "Churn Rate %", "New Subscriptions"],
    "January": [1245, 75320, 87, 5.2, 320],
    "February": [1340, 81200, 89, 4.8, 350],
    "March": [1400, 86000, 90, 4.5, 370],
})
st.dataframe(data_table.style.set_table_styles([
    {'selector': 'th', 'props': [('background-color', '#6c63ff'), ('color', 'white')]},
    {'selector': 'td', 'props': [('background-color', '#2b2b42'), ('color', 'white')]}
]))

