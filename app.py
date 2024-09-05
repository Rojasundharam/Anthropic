import streamlit as st
from chatbot import ChatBot
from config import TASK_SPECIFIC_INSTRUCTIONS
from google_drive_utils import get_drive_service

def main():
    st.title("JKKN Assistant🤖")
    
    drive_service = get_drive_service()
    
    if drive_service is None:
        st.write("Please authenticate with Google Drive to continue.")
        return

    if "messages" not in st.session_state:
        st.session_state.messages = [
            {'role': "user", "content": TASK_SPECIFIC_INSTRUCTIONS},
            {'role': "assistant", "content": "Understood"},
        ]
    
    if "chatbot" not in st.session_state:
        st.session_state.chatbot = ChatBot(st.session_state)
    
    # Display user and assistant messages skipping the first two
    for message in st.session_state.messages[2:]:
        # ignore tool use blocks
        if isinstance(message["content"], str):
            with st.chat_message(message["role"]):
                st.markdown(message["content"])
    
    if user_msg := st.chat_input("Type your message here..."):
        st.chat_message("user").markdown(user_msg)
        
        with st.chat_message("assistant"):
            with st.spinner("JKKN Assist is thinking..."):
                response_placeholder = st.empty()
                full_response = st.session_state.chatbot.process_user_input(user_msg)
                response_placeholder.markdown(full_response)

if __name__ == "__main__":
    main()