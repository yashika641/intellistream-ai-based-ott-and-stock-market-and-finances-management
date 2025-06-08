import streamlit as st
import re

st.set_page_config(page_title="Sign Up - IntelliStream", layout="centered")

st.markdown("""
<style>
.signup-card {
    background-color: #1f1f2e;
    padding: 2rem;
    border-radius: 15px;
    box-shadow: 0 8px 20px rgba(0,0,0,0.3);
    max-width: 400px;
    margin: 5rem auto;
    color: white;
    font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
}
.signup-card h2 {
    text-align: center;
    margin-bottom: 1.5rem;
}
.signup-card input[type="text"],
.signup-card input[type="email"],
.signup-card input[type="password"] {
    width: 100%;
    padding: 12px 15px;
    margin: 8px 0 20px 0;
    border-radius: 10px;
    border: none;
    outline: none;
    font-size: 16px;
    box-sizing: border-box;
}
.signup-card input[type="text"]:focus,
.signup-card input[type="email"]:focus,
.signup-card input[type="password"]:focus {
    box-shadow: 0 0 6px 2px #6c63ff;
    transition: box-shadow 0.3s ease-in-out;
}
.signup-card button {
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
.signup-card button:hover {
    background-color: #554bcc;
}
.signup-error {
    color: #ff6b6b;
    margin-bottom: 1rem;
    text-align: center;
}
.back-link {
    margin-top: 15px;
    display: block;
    text-align: center;
}
a.link {
    color: #6c63ff;
    text-decoration: none;
    font-weight: 600;
    cursor: pointer;
    font-size: 14px;
}
a.link:hover {
    text-decoration: underline;
}
</style>
""", unsafe_allow_html=True)

def validate_email(email):
    # Simple regex for email validation
    pattern = r'^[\w\.-]+@[\w\.-]+\.\w+$'
    return re.match(pattern, email)

def signup_page():
    st.markdown('<div class="signup-card">', unsafe_allow_html=True)
    st.markdown("<h2>📝 Create Your IntelliStream Account</h2>", unsafe_allow_html=True)

    if "signup_error" not in st.session_state:
        st.session_state.signup_error = ""
    if "signup_success" not in st.session_state:
        st.session_state.signup_success = ""

    username = st.text_input("Username")
    email = st.text_input("Email")
    password = st.text_input("Password", type="password")
    confirm_password = st.text_input("Confirm Password", type="password")

    if st.button("Sign Up"):
        if not username.strip() or not email.strip() or not password.strip() or not confirm_password.strip():
            st.session_state.signup_error = "Please fill in all fields."
            st.session_state.signup_success = ""
        elif not validate_email(email):
            st.session_state.signup_error = "Please enter a valid email address."
            st.session_state.signup_success = ""
        elif password != confirm_password:
            st.session_state.signup_error = "Passwords do not match."
            st.session_state.signup_success = ""
        else:
            # TODO: Replace with actual sign up logic (database/API)
            st.session_state.signup_error = ""
            st.session_state.signup_success = f"Account created successfully for {username}! You can now log in."

    if st.session_state.signup_error:
        st.markdown(f'<div class="signup-error">{st.session_state.signup_error}</div>', unsafe_allow_html=True)

    if st.session_state.signup_success:
        st.success(st.session_state.signup_success)

    st.markdown("""
    <div class="back-link">
        <a href="#" class="link" onclick="alert('Go back to login flow not implemented yet!')">← Back to Login</a>
    </div>
    """, unsafe_allow_html=True)

    st.markdown('</div>', unsafe_allow_html=True)

signup_page()
