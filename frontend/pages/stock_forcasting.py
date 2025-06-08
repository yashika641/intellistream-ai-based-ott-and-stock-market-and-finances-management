import streamlit as st
import pandas as pd
import numpy as np

# Dummy prediction function — replace with your model
def predict_stock(news_df):
    # Let's simulate some stock price predictions as example
    # You will replace this with your actual model inference code
    np.random.seed(42)
    predictions = np.random.randn(len(news_df)) * 2 + 100  # random stock prices around 100
    return predictions

st.title("OTT News Based Stock Price Prediction")

st.markdown("""
Upload OTT news data (CSV) containing news headlines or descriptions to predict stock price movements.
""")

uploaded_file = st.file_uploader("Upload OTT News CSV", type=["csv"])

if uploaded_file is not None:
    try:
        news_df = pd.read_csv(uploaded_file)
        st.subheader("OTT News Data Preview")
        st.dataframe(news_df.head(10))
        
        if st.button("Predict Stock Prices"):
            with st.spinner("Predicting..."):
                predictions = predict_stock(news_df)
                news_df['Predicted_Stock_Price'] = predictions
                
                st.subheader("Predicted Stock Prices")
                st.dataframe(news_df[['Predicted_Stock_Price']])
                
                # Optional: Show histogram of predicted prices
                st.bar_chart(news_df['Predicted_Stock_Price'])
    except Exception as e:
        st.error(f"Error loading or processing the file: {e}")

else:
    st.info("Please upload a CSV file to get started.")
