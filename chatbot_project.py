import streamlit as st
from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage
from dotenv import load_dotenv
import os

# ------------------------- PAGE CONFIG -------------------------
st.set_page_config(page_title="MY GPT", page_icon="🤖", layout="wide")

# ------------------------- CUSTOM CSS (ChatGPT Style) -------------------------
st.markdown("""
<style>

body {
    background-color: #f7f7f8;
}

/* Chat bubble styling */
.user-msg {
    background-color: #DCF8C6;
    padding: 12px 16px;
    border-radius: 12px;
    max-width: 70%;
    margin-bottom: 10px;
    color: black;
}

.ai-msg {
    background-color: white;
    padding: 12px 16px;
    border-radius: 12px;
    max-width: 70%;
    margin-bottom: 10px;
    border: 1px solid #e5e5e5;
    color: black;
}

/* Centering chat container */
.chat-container {
    max-width: 750px;
    margin: auto;
    padding-top: 20px;
}

</style>
""", unsafe_allow_html=True)


# ------------------------- LOAD API KEY -------------------------
load_dotenv()
api_key = os.getenv("HUGGINGFACEHUB_API_TOKEN")

# ------------------------- LOAD MODEL -------------------------
llm = HuggingFaceEndpoint(
    repo_id="google/gemma-2-2b-it",
    task="conversational",
    huggingfacehub_api_token=api_key,
    max_new_tokens=150
)

model = ChatHuggingFace(llm=llm)

# ------------------------- INITIAL CHAT HISTORY -------------------------
if "chat_history" not in st.session_state:
    st.session_state.chat_history = [
        SystemMessage(content="You are a helpful AI assistant. Respond clearly.")
    ]


# ------------------------- HEADER -------------------------
st.markdown("<h2 style='text-align:center;'>🤖 MY CHATBOT</h2>", unsafe_allow_html=True)
st.markdown("<p style='text-align:center; color:gray;'>Chat with your HuggingFace model</p>", unsafe_allow_html=True)

st.markdown("<div class='chat-container'>", unsafe_allow_html=True)

# ------------------------- DISPLAY CHAT HISTORY -------------------------
for msg in st.session_state.chat_history:
    if isinstance(msg, HumanMessage):
        st.markdown(f"<div class='user-msg'><b>You:</b> {msg.content}</div>", unsafe_allow_html=True)
    elif isinstance(msg, AIMessage):
        st.markdown(f"<div class='ai-msg'><b>AI:</b> {msg.content}</div>", unsafe_allow_html=True)

# ------------------------- USER INPUT -------------------------
user_input = st.chat_input("Send a message...")

if user_input:
    # Add user message
    st.session_state.chat_history.append(HumanMessage(content=user_input))

    # Get model response
    response = model.invoke(user_input)
    st.session_state.chat_history.append(AIMessage(content=response.content))

    # Rerun to update UI instantly
    st.rerun()

st.markdown("</div>", unsafe_allow_html=True)
