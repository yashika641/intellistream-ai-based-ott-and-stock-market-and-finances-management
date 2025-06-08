import streamlit as st

st.set_page_config(page_title="Login - IntelliStream", layout="centered")

st.markdown("""
<style>
.login-card {
    background-color: #1f1f2e;
    padding: 2rem;
    border-radius: 15px;
    box-shadow: 0 8px 20px rgba(0,0,0,0.3);
    max-width: 400px;
    margin: 5rem auto;
    color: white;
    font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
}
.login-card h2 {
    text-align: center;
    margin-bottom: 1.5rem;
}
.login-card input[type="text"],
.login-card input[type="password"] {
    width: 100%;
    padding: 12px 15px;
    margin: 8px 0 20px 0;
    border-radius: 10px;
    border: none;
    outline: none;
    font-size: 16px;
    box-sizing: border-box;
}
.login-card input[type="text"]:focus,
.login-card input[type="password"]:focus {
    box-shadow: 0 0 6px 2px #6c63ff;
    transition: box-shadow 0.3s ease-in-out;
}
.login-card button {
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
.login-card button:hover {
    background-color: #554bcc;
}
.login-error {
    color: #ff6b6b;
    margin-bottom: 1rem;
    text-align: center;
}
.links {
    margin-top: 15px;
    display: flex;
    justify-content: space-between;
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

def login_page():
    st.markdown('<div class="login-card">', unsafe_allow_html=True)
    st.markdown("<h2>🔐 Login to IntelliStream</h2>", unsafe_allow_html=True)

    if "login_error" not in st.session_state:
        st.session_state.login_error = ""

    username = st.text_input("Username")
    password = st.text_input("Password", type="password")

    if st.button("Login"):
        if username.strip() == "" or password.strip() == "":
            st.session_state.login_error = "Please enter both username and password."
        else:
            # Dummy check - replace with real authentication
            if username == "admin" and password == "intellistream":
                st.session_state.login_error = ""
                st.success(f"Welcome back, {username}!")
                # TODO: redirect or set logged-in state
            else:
                st.session_state.login_error = "Invalid username or password."

    if st.session_state.login_error:
        st.markdown(f'<div class="login-error">{st.session_state.login_error}</div>', unsafe_allow_html=True)

    # Links for forgot password and signup
    st.markdown("""
    <div class="links">
        <a href="#" class="link" onclick="alert('Forgot Password flow not implemented yet!')">Forgot Password?</a>
        <a href="#" class="link" onclick="alert('Sign Up flow not implemented yet!')">Sign Up</a>
    </div>
    """, unsafe_allow_html=True)

    st.markdown('</div>', unsafe_allow_html=True)

login_page()
