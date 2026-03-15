import streamlit as st
from langchain_huggingface import HuggingFaceEndpoint
from langchain_core.messages import HumanMessage, AIMessage
from dotenv import load_dotenv
import os

# ---------------- PAGE CONFIG ----------------
st.set_page_config(page_title="MY GPT", page_icon="🤖", layout="wide")

# ---------------- STYLE ----------------
st.markdown("""
<style>

body {
    background-color: #f7f7f8;
}

/* user bubble */
.user-msg {
    background-color: #DCF8C6;
    padding: 12px 16px;
    border-radius: 12px;
    max-width: 70%;
    margin-bottom: 10px;
    color: black;
}

/* ai bubble */
.ai-msg {
    background-color: white;
    padding: 12px 16px;
    border-radius: 12px;
    max-width: 70%;
    margin-bottom: 10px;
    border: 1px solid #e5e5e5;
    color: black;
}

/* center container */
.chat-container {
    max-width: 750px;
    margin: auto;
    padding-top: 20px;
}

</style>
""", unsafe_allow_html=True)

# ---------------- LOAD API KEY ----------------
load_dotenv()

api_key = os.getenv("HUGGINGFACEHUB_API_TOKEN")

if api_key is None:
    st.error("HuggingFace API key not found. Add it to your .env file.")
    st.stop()

# ---------------- LOAD MODEL ----------------
llm = HuggingFaceEndpoint(
    repo_id="HuggingFaceH4/zephyr-7b-beta",
    task="text-generation",
    huggingfacehub_api_token=api_key,
    max_new_tokens=512,
    temperature=0.7
)

# ---------------- CHAT HISTORY ----------------
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

# ---------------- HEADER ----------------
st.markdown("<h2 style='text-align:center;'>🤖 MY CHATBOT</h2>", unsafe_allow_html=True)
st.markdown("<p style='text-align:center;color:gray;'>Chat with HuggingFace Model</p>", unsafe_allow_html=True)

st.markdown("<div class='chat-container'>", unsafe_allow_html=True)

# ---------------- SHOW CHAT ----------------
for msg in st.session_state.chat_history:

    if isinstance(msg, HumanMessage):
        st.markdown(
            f"<div class='user-msg'><b>You:</b> {msg.content}</div>",
            unsafe_allow_html=True
        )

    elif isinstance(msg, AIMessage):
        st.markdown(
            f"<div class='ai-msg'><b>AI:</b> {msg.content}</div>",
            unsafe_allow_html=True
        )

# ---------------- USER INPUT ----------------
user_input = st.chat_input("Send a message...")

if user_input:

    # store user message
    st.session_state.chat_history.append(HumanMessage(content=user_input))

    with st.spinner("AI is thinking..."):

        response = llm.invoke(user_input)

    # store ai response
    st.session_state.chat_history.append(AIMessage(content=response))

    st.rerun()

st.markdown("</div>", unsafe_allow_html=True)
