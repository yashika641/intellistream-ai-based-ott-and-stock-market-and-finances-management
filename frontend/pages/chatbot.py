import streamlit as st

st.set_page_config(page_title="IntelliStream Chatbot", layout="wide")

# CSS for chat bubbles, avatars, and auto-scroll
st.markdown("""
<style>
.chat-container {
    max-height: 500px;
    overflow-y: auto;
    padding: 10px;
    background: #2c2f48;
    border-radius: 15px;
    color: white;
    font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
    scroll-behavior: smooth;
}

/* Chat message container: flexbox for left/right alignment with avatar */
.chat-message {
    display: flex;
    margin: 8px 0;
    max-width: 70%;
}

/* Bot messages aligned left */
.chat-message.bot {
    flex-direction: row;
    align-items: flex-start;
}

/* User messages aligned right */
.chat-message.user {
    flex-direction: row-reverse;
    align-items: flex-start;
    margin-left: auto;
}

/* Avatar style */
.avatar {
    width: 36px;
    height: 36px;
    border-radius: 50%;
    margin: 0 10px;
    flex-shrink: 0;
}

/* User avatar */
.avatar.user {
    background: #6c63ff;
    background-image: url('https://i.imgur.com/OV1M8Pu.png'); /* cute user icon */
    background-size: cover;
}

/* Bot avatar */
.avatar.bot {
    background: #444a85;
    background-image: url('https://i.imgur.com/7jM9mDD.png'); /* cute robot icon */
    background-size: cover;
}

/* Message bubble */
.message {
    padding: 10px 15px;
    border-radius: 20px;
    color: white;
    word-wrap: break-word;
}

/* Bot message bubble shape */
.message.bot {
    background-color: #444a85;
    border-radius: 20px 20px 20px 0;
}

/* User message bubble shape */
.message.user {
    background-color: #6c63ff;
    border-radius: 20px 20px 0 20px;
}

/* Input text field styling */
input[type="text"] {
    padding: 12px 20px;
    width: 100%;
    border-radius: 25px;
    border: none;
    outline: none;
    font-size: 16px;
    box-sizing: border-box;
}

input[type="text"]:focus {
    box-shadow: 0 0 5px 2px #6c63ff;
    transition: box-shadow 0.3s ease-in-out;
}

</style>
""", unsafe_allow_html=True)

st.title("🤖 IntelliStream Chatbot")

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

def get_bot_response(user_msg):
    # Dummy response: replace with your chatbot API call
    return f"Echo: {user_msg}"

# Chat container
chat_placeholder = st.empty()

def render_chat():
    with chat_placeholder.container():
        st.markdown('<div class="chat-container" id="chat-container">', unsafe_allow_html=True)
        for chat in st.session_state.chat_history:
            role = "user" if chat["user"] else "bot"
            st.markdown(f'''
            <div class="chat-message {role}">
                <div class="avatar {role}"></div>
                <div class="message {role}">{chat["message"]}</div>
            </div>
            ''', unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)

render_chat()

# User input
user_input = st.text_input("Type your message and press Enter:", key="input_text")

if user_input:
    # Append user message
    st.session_state.chat_history.append({"user": True, "message": user_input})

    # Bot response
    bot_reply = get_bot_response(user_input)
    st.session_state.chat_history.append({"user": False, "message": bot_reply})

    # Rerun app to refresh chat
    st.experimental_rerun()

# JavaScript to auto scroll chat container to bottom
st.markdown("""
<script>
const chatContainer = window.parent.document.querySelector('.chat-container');
if(chatContainer){
    chatContainer.scrollTop = chatContainer.scrollHeight;
}
</script>
""", unsafe_allow_html=True)
