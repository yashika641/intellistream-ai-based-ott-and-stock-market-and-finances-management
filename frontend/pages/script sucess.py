import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from io import StringIO
import joblib
import os

# Assuming your functions (load_data, preprocess_data, split_data, lightgbm_training, save_model) are in model_pipeline.py
from model_codes.script_nlp_model import load_data, split_data, lightgbm_training, save_model, final_preprocess

st.title("Script Success Prediction Model")

uploaded_file = st.file_uploader("Upload a CSV with scripts data", type=["csv"])

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
    st.write("Data preview:")
    st.dataframe(df.head())
else:
    st.warning("Please upload a CSV file.")
    st.stop()

if st.button("Run preprocessing and train model"):
    with st.spinner("Preprocessing and training... This may take a while."):
        try:
            # Optionally preprocess or use final preprocess if df already has script_text
            x_train, x_test, y_train, y_test = split_data(df)
            results, model = lightgbm_training(x_train, y_train, x_test, y_test)
            st.success("Model trained successfully!")

            st.subheader("Results:")
            st.write(results)

            # Plot feature importance
            from sklearn.inspection import permutation_importance
            importances = permutation_importance(model, x_test, y_test, n_repeats=10, random_state=42, n_jobs=-1)
            importances_series = pd.Series(importances.importances_mean, index=x_test.columns)
            importances_series = importances_series.sort_values()

            st.subheader("Feature Importance")
            fig, ax = plt.subplots()
            importances_series.plot(kind='barh', ax=ax)
            st.pyplot(fig)

            # Save the model
            save_model(model, 'lightgbm_best')
            st.info("Model saved as 'models/lightgbm_best.pkl'")

        except Exception as e:
            st.error(f"Error during training: {e}")

st.markdown("---")

st.subheader("Predict Success for New Script Text")

input_script = st.text_area("Paste your script text here:")

if st.button("Predict Success"):
    if not input_script.strip():
        st.warning("Please enter some script text.")
    else:
        try:
            # Load the trained model and preprocessing here
            model_path = 'models/lightgbm_best.pkl'
            if not os.path.exists(model_path):
                st.error("Model file not found. Please train the model first.")
            else:
                model = joblib.load(model_path)
                # You must run the same preprocessing pipeline here on input_script as training
                
                # For demonstration: dummy feature vector creation (replace with your real preprocess)
                # WARNING: Replace with your actual feature extraction for new input scripts
                input_features = pd.DataFrame({
                    'avg_sentence_length': [len(input_script.split()) / max(1, len(input_script.split('.')))],
                    'tfidf_mean': [0],  # placeholder
                    'subjectivity_score': [0],  # placeholder
                    'positive_word_count': [0],  # placeholder
                    'negative_word_count': [0],  # placeholder
                    'named_entity_count': [0]  # placeholder
                })
                
                prediction = model.predict(input_features)[0]
                label = "Success" if prediction == 1 else "Not Success"
                st.write(f"Predicted script success: **{label}**")

        except Exception as e:
            st.error(f"Prediction error: {e}")
