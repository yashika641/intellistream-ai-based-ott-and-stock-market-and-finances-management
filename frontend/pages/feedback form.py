import streamlit as st
import re

st.set_page_config(page_title="Feedback - IntelliStream", layout="centered")

st.markdown("""
<style>
.feedback-card {
    background-color: #1f1f2e;
    padding: 2rem;
    border-radius: 15px;
    box-shadow: 0 8px 20px rgba(0,0,0,0.3);
    max-width: 500px;
    margin: 5rem auto;
    color: white;
    font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
}
.feedback-card h2 {
    text-align: center;
    margin-bottom: 1.5rem;
}
.feedback-card input[type="text"],
.feedback-card input[type="email"],
.feedback-card textarea,
.feedback-card select {
    width: 100%;
    padding: 12px 15px;
    margin: 8px 0 20px 0;
    border-radius: 10px;
    border: none;
    outline: none;
    font-size: 16px;
    box-sizing: border-box;
    background-color: #2b2b42;
    color: white;
}
.feedback-card input[type="text"]:focus,
.feedback-card input[type="email"]:focus,
.feedback-card textarea:focus,
.feedback-card select:focus {
    box-shadow: 0 0 6px 2px #6c63ff;
    transition: box-shadow 0.3s ease-in-out;
}
.feedback-card button {
    width: 100%;
    padding: 12px;
    border-radius: 10px;
    background-color: #6c63ff;
    color: white;
    font-weight: 600;
    font-size: 18px;
    border: none;
    cursor: pointer;
    transition: background-color 0.3s ease-in-out;
}
.feedback-card button:hover {
    background-color: #554bcc;
}
.feedback-error {
    color: #ff6b6b;
    margin-bottom: 1rem;
    text-align: center;
}
.feedback-success {
    color: #7bed9f;
    margin-bottom: 1rem;
    text-align: center;
}
</style>
""", unsafe_allow_html=True)

def validate_email(email):
    pattern = r'^[\w\.-]+@[\w\.-]+\.\w+$'
    return re.match(pattern, email)

def feedback_form():
    st.markdown('<div class="feedback-card">', unsafe_allow_html=True)
    st.markdown("<h2>💬 Send Us Your Feedback</h2>", unsafe_allow_html=True)

    if "feedback_error" not in st.session_state:
        st.session_state.feedback_error = ""
    if "feedback_success" not in st.session_state:
        st.session_state.feedback_success = ""

    name = st.text_input("Name")
    email = st.text_input("Email")
    rating = st.selectbox("Rating", options=["Select a rating", "⭐️", "⭐⭐️", "⭐⭐⭐️", "⭐⭐⭐⭐️", "⭐⭐⭐⭐⭐️"])
    comments = st.text_area("Comments / Suggestions")

    if st.button("Submit Feedback"):
        if not name.strip() or not email.strip() or rating == "Select a rating":
            st.session_state.feedback_error = "Please fill in your name, email, and select a rating."
            st.session_state.feedback_success = ""
        elif not validate_email(email):
            st.session_state.feedback_error = "Please enter a valid email address."
            st.session_state.feedback_success = ""
        else:
            # TODO: Handle/store feedback (database, email, etc)
            st.session_state.feedback_error = ""
            st.session_state.feedback_success = "Thank you for your valuable feedback!"

            # Optionally clear inputs or disable form here if needed

    if st.session_state.feedback_error:
        st.markdown(f'<div class="feedback-error">{st.session_state.feedback_error}</div>', unsafe_allow_html=True)
    if st.session_state.feedback_success:
        st.markdown(f'<div class="feedback-success">{st.session_state.feedback_success}</div>', unsafe_allow_html=True)

    st.markdown('</div>', unsafe_allow_html=True)

feedback_form()
