import streamlit as st
from streamlit_extras.stylable_container import stylable_container

# ---- Page Config ---- #
st.set_page_config(
    page_title="AI OTT Suite",
    page_icon="🎬",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ---- Sidebar (Right) ---- #
with st.sidebar:
    st.image("https://cdn-icons-png.flaticon.com/512/3135/3135715.png", width=80)
    st.title("🎥 AI OTT Platform")
    st.text_input("🔍 Search Modules")
    st.button("🔑 Login")
    st.button("📝 Signup")
    st.markdown("---")
    st.markdown("### 🤖 Need Help?")
    st.markdown("Chat with our AI Assistant below 👇")
    chatbot = st.chat_input("Ask me anything...")
    if chatbot:
        st.write(f"🔍 You asked: {chatbot}")
        st.write("🤖 (Pretend AI): I'm still learning to answer better!")
    st.markdown("---")
    st.markdown("#### 📞 Contact Us")
    st.markdown("📧 support@aiottsuite.com")
    st.markdown("📱 +91-99999-88888")
    st.markdown("#### 💬 Feedback Form")
    st.markdown("[Submit here](https://forms.gle/your-feedback-form)")

# ---- Header ---- #
st.markdown("""
    <div style='text-align: center; padding: 20px 0;'>
        <h1 style='font-size: 3em;'>🎬 AI-Powered OTT Analytics Suite</h1>
        <p style='font-size: 1.2em;'>A powerful platform to analyze, predict, and enhance content success on OTT platforms using AI</p>
    </div>
""", unsafe_allow_html=True)

# ---- Modules Overview ---- #
CARD_CSS = """
    background-color: #f8f9fa;
    border-radius: 12px;
    padding: 20px;
    margin: 10px 0;
    transition: all 0.3s ease-in-out;
    box-shadow: 2px 2px 10px rgba(0,0,0,0.1);
"""

cols = st.columns(3)
modules = [
    ("Churn Prediction Engine", "Detects user dropout using RNN, XGBoost", "📉"),
    ("Hybrid Recommender System", "Suggests content using mood + history", "🎯"),
    ("Script Success Predictor", "Predicts script success via NLP", "📝"),
    ("Finance Signal Integrator", "Maps OTT buzz to stock prices", "💹"),
    ("BI Dashboard Suite", "Visual dashboards with Plotly, Dash", "📊"),
    ("Explainability Engine", "Model interpretation using SHAP/LIME", "🔍"),
]

for i, (title, desc, icon) in enumerate(modules):
    with cols[i % 3]:
        unique_key = f"card_{i}"
        with stylable_container(key=unique_key, css_styles=CARD_CSS):
            st.markdown(f"## {icon} {title}")
            st.write(desc)

# ---- Footer ---- #
st.markdown("""
    <hr>
    <div style='text-align: center;'>
        <p>© 2025 AI OTT Suite | Made with ❤️ for smarter OTT experiences</p>
        <p><a href='#'>About</a> | <a href='mailto:support@aiottsuite.com'>Contact</a> | <a href='https://forms.gle/your-feedback-form'>Feedback</a></p>
    </div>
""", unsafe_allow_html=True)

# Optional: Background animation (can be replaced with custom image/video if hosted)
st.markdown("""
    <style>
    body {
        background: linear-gradient(135deg, #e0eafc, #cfdef3);
        background-attachment: fixed;
    }
    </style>
""", unsafe_allow_html=True)
