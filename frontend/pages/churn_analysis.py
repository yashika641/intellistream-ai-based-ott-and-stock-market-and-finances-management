import streamlit as st
import pandas as pd
import numpy as np
import joblib  # or pickle
from catboost import CatBoostClassifier

# Load your pre-trained model
@st.cache_data
def load_model():
    model = joblib.load(r'C:\Users\Dell\OneDrive\Desktop\intellistream\intellistream-ai-based-ott-and-stock-market-and-finances-management\model_codes\best_model.joblib')  # or your saved CatBoost model path
    return model

model = load_model()

st.title("Churn Prediction UI")

st.markdown("""
Enter user details below to predict if they are likely to churn.
""")

# Create input widgets for each feature your model expects.
# Make sure the feature names and types match your training data preprocessing.

def user_input_features():
    age = st.number_input("Age", min_value=10, max_value=100, value=30)
    gender = st.selectbox("Gender", options=[0, 1])  # encoded value
    region = st.selectbox("Region", options=[0, 1, 2])  # example
    plan = st.selectbox("Plan", options=[0, 1, 2])
    device_type = st.selectbox("Device Type", options=[0, 1, 2])
    preferred_genre = st.selectbox("Preferred Genre", options=[0, 1, 2])
    payment_method = st.selectbox("Payment Method", options=[0, 1, 2])
    ad_type = st.selectbox("Ad Type", options=[0, 1, 2])
    auto_renew = st.selectbox("Auto Renew", options=[0, 1])
    discount_used = st.number_input("Discount Used", min_value=0, max_value=100, value=0)
    complaints = st.number_input("Complaints", min_value=0, max_value=10, value=0)
    support_calls = st.number_input("Support Calls", min_value=0, max_value=10, value=0)
    rating = st.slider("Rating", 0.0, 5.0, 3.0)
    reviews_written = st.number_input("Reviews Written", min_value=0, max_value=100, value=0)
    peak_hours = st.selectbox("Peak Hours", options=[0, 1])
    total_revenue = st.number_input("Total Revenue", min_value=0.0, value=100.0)
    watch_time_ratio = st.slider("Watch Time Ratio", 0.0, 1.0, 0.5)
    avg_watch_hours_per_month = st.number_input("Avg Watch Hours per Month", min_value=0.0, value=10.0)
    binge_indicator = st.selectbox("Binge Indicator", options=[0, 1])
    is_heavy_user = st.selectbox("Is Heavy User", options=[0, 1])
    tenure = st.number_input("Tenure (months)", min_value=0.0, value=12.0)

    data = {
        'age': age,
        'gender': gender,
        'region': region,
        'plan': plan,
        'device_type': device_type,
        'preferred_genre': preferred_genre,
        'payment_method': payment_method,
        'ad_type': ad_type,
        'auto_renew': auto_renew,
        'discount_used': discount_used,
        'complaints': complaints,
        'support_calls': support_calls,
        'rating': rating,
        'reviews_written': reviews_written,
        'peak_hours': peak_hours,
        'total_revenue': total_revenue,
        'watch_time_ratio': watch_time_ratio,
        'avg_watch_hours_per_month': avg_watch_hours_per_month,
        'binge_indicator': binge_indicator,
        'is_heavy_user': is_heavy_user,
        'tenure': tenure
    }
    features = pd.DataFrame(data, index=[0])
    return features

input_df = user_input_features()

if st.button("Predict Churn"):
    # Run your model prediction here
    # Make sure you apply same scaling or feature transformation as training
    # If model expects scaled input, load scaler or apply scaling here
    # For simplicity, assuming model takes raw input matching your features
    prediction_proba = model.predict_proba(input_df)[:, 1][0]
    prediction = model.predict(input_df)[0]
    
    st.write(f"Prediction: {'Will Churn' if prediction == 1 else 'Will NOT Churn'}")
    st.write(f"Churn Probability: {prediction_proba:.2f}")

    # Optional: Show SHAP explanation if you want
    import shap
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(input_df)
    st.pyplot(shap.summary_plot(shap_values, input_df))
