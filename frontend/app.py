import streamlit as st

st.set_page_config(page_title="IntelliStream Dashboard", layout="wide")

# --- CSS for top nav bar ---
st.markdown("""
<style>
/* Top navbar container */
.navbar {
    background-color: #1f1f2e;
    overflow: hidden;
    display: flex;
    justify-content: center;
    align-items: center;
    padding: 0.5rem 0;
    border-radius: 0 0 10px 10px;
    box-shadow: 0 4px 10px rgba(0,0,0,0.3);
    font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
    z-index: 100;
}

/* Navbar links */
.navbar a {
    color: white;
    padding: 14px 20px;
    text-decoration: none;
    font-weight: 600;
    font-size: 18px;
    transition: background-color 0.3s ease;
    border-radius: 5px;
    margin: 0 5px;
}

/* Hover effect */
.navbar a:hover {
    background-color: #3e3e55;
    cursor: pointer;
}

/* Active/current link */
.navbar a.active {
    background-color: #6c63ff;
}

/* Fix main content padding from navbar */
.main-content {
    padding-top: 1rem;
}
</style>
""", unsafe_allow_html=True)

# --- Top nav bar links ---
# Use query params or session_state to track which page is selected

# Pages for nav bar
pages = [
    "Home",
    "Churn Prediction",
    "Recommender System",
    "Script Predictor",
    "Finance Integrator",
    "BI Dashboard",
    "Explainability",
    "Admin Panel",
    "About"
]

# Get current page from query param or default to Home
query_params = st.experimental_get_query_params()
current_page = query_params.get("page", ["Home"])[0]

# Show nav bar
nav_html = '<div class="navbar">'
for page in pages:
    active_class = "active" if page == current_page else ""
    nav_html += f'<a href="?page={page}" class="{active_class}">{page}</a>'
nav_html += '</div>'

st.markdown(nav_html, unsafe_allow_html=True)

# --- Main content padding from navbar ---
st.markdown('<div class="main-content">', unsafe_allow_html=True)

# Sidebar still optional, if you want to keep it
st.sidebar.image("logo.png", width=150)
st.sidebar.title("📂 IntelliStream Menu")
selected_sidebar = st.sidebar.radio("Navigate to:", pages, index=pages.index(current_page) if current_page in pages else 0)

# Sync sidebar selection with top nav
if selected_sidebar != current_page:
    st.experimental_set_query_params(page=selected_sidebar)
    st.experimental_rerun()

selected = current_page

# Styling cards as before
st.markdown("""
    <style>
    .card {
        background-color: #1f1f2e;
        padding: 1.5rem;
        border-radius: 15px;
        box-shadow: 0 4px 15px rgba(0,0,0,0.2);
        color: white;
        text-align: center;
        transition: all 0.3s ease-in-out;
        margin-bottom: 20px;
    }
    .card:hover {
        transform: scale(1.05);
        box-shadow: 0 6px 20px rgba(0,0,0,0.3);
    }
    .card-icon {
        font-size: 36px;
        margin-bottom: 10px;
    }
    </style>
""", unsafe_allow_html=True)

# Main content: Cards
if selected == "Home":
    st.title("🎯 IntelliStream AI Dashboard")

    col1, col2, col3 = st.columns(3)

    with col1:
        st.markdown("""
        <div class="card">
            <div class="card-icon">📉</div>
            <h4>Churn Prediction</h4>
            <p>Detects potential OTT user dropout using ML models.</p>
        </div>
        """, unsafe_allow_html=True)

    with col2:
        st.markdown("""
        <div class="card">
            <div class="card-icon">🎬</div>
            <h4>Recommender System</h4>
            <p>Suggests content using mood + history + NLP.</p>
        </div>
        """, unsafe_allow_html=True)

    with col3:
        st.markdown("""
        <div class="card">
            <div class="card-icon">📝</div>
            <h4>Script Predictor</h4>
            <p>Evaluates OTT script success potential.</p>
        </div>
        """, unsafe_allow_html=True)

    col4, col5, col6 = st.columns(3)

    with col4:
        st.markdown("""
        <div class="card">
            <div class="card-icon">📈</div>
            <h4>Finance Signal</h4>
            <p>Correlates OTT buzz with stock price shifts.</p>
        </div>
        """, unsafe_allow_html=True)

    with col5:
        st.markdown("""
        <div class="card">
            <div class="card-icon">📊</div>
            <h4>BI Dashboard</h4>
            <p>Visualizes financial & user engagement data.</p>
        </div>
        """, unsafe_allow_html=True)

    with col6:
        st.markdown("""
        <div class="card">
            <div class="card-icon">🧠</div>
            <h4>Explainability Engine</h4>
            <p>Interpret predictions with SHAP, LIME, and ELI5.</p>
        </div>
        """, unsafe_allow_html=True)

st.markdown('</div>', unsafe_allow_html=True)
