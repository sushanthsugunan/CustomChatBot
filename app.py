import streamlit as st
from chatbot_neo import chat_with_bot

st.title("Custom Chatbot")
st.write("Type a message and get a response from the chatbot.")

user_input = st.text_input("You: ", "")
if st.button("Send"):
    if user_input:
        response = chat_with_bot(user_input)
        st.write(f"Bot: {response}")
