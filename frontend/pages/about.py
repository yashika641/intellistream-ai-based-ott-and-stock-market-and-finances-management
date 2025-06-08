import streamlit as st
from PIL import Image
import base64

# Set background image
def set_bg(image_file):
    with open(image_file, "rb") as img:
        encoded = base64.b64encode(img.read()).decode()
    st.markdown(
        f"""
        <style>
        .stApp {{
            background-image: url("data:image/png;base64,{encoded}");
            background-size: cover;
            background-attachment: fixed;
        }}
        </style>
        """,
        unsafe_allow_html=True
    )

# Set your custom background
set_bg("background.jpg")  # replace with your image file

# Page Title
st.title("📊 About IntelliStream")

st.markdown("""
Welcome to **IntelliStream**, your AI-powered OTT and stock market analytics suite. This platform combines the power of machine learning with real-time data to provide insights for both entertainment and finance domains.
""")

st.markdown("### 🔍 Modules Overview")

modules = [
    {
        "name": "Churn Prediction Engine",
        "desc": "Detects potential OTT user dropout based on usage patterns.",
        "stack": "RNN, XGBoost, Autoencoders",
        "icon": "📉"
    },
    {
        "name": "Hybrid Recommender System",
        "desc": "Recommends shows/movies using history + mood + collaborative filtering.",
        "stack": "Neural CF, NLP, Content-Based (Emotion/Mood Extraction)",
        "icon": "🎬"
    },
    {
        "name": "Script Success Predictor",
        "desc": "Evaluates a script's potential for OTT success.",
        "stack": "BERT, Topic Modeling, Regression, Sentiment Analysis",
        "icon": "📝"
    },
    {
        "name": "Finance Signal Integrator",
        "desc": "Maps OTT buzz (social/news) to stock price shifts.",
        "stack": "LSTM, ARIMA, BERT, Prophet",
        "icon": "📈"
    },
    {
        "name": "BI Dashboard Suite",
        "desc": "Visualizes engagement heatmaps, financial correlations, and predictions.",
        "stack": "Streamlit / Dash + Plotly",
        "icon": "📊"
    },
    {
        "name": "Explainability Engine",
        "desc": "Explains model decisions using SHAP, LIME, and ELI5.",
        "stack": "SHAP, LIME, ELI5",
        "icon": "🧠"
    },
    {
        "name": "Admin & Upload Panel",
        "desc": "Secure portal for uploading scripts, datasets, and viewing reports.",
        "stack": "Firebase Auth, MongoDB, FastAPI",
        "icon": "🛠️"
    }
]

for module in modules:
    st.markdown(f"""
    #### {module['icon']} {module['name']}
    **Description:** {module['desc']}  
    **AI/ML Stack:** `{module['stack']}`
    """)

st.markdown("---")
st.markdown("🔐 _Built with ❤️ by Kaushal | Powered by Streamlit & Python_")
