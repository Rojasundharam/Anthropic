import streamlit as st
from chatbot import ChatBot
from config import TASK_SPECIFIC_INSTRUCTIONS
from google_drive_utils import get_drive_service

def main():
    st.title("Chat with JKKN Assist, JKKN Educational Institutions' Assistant🤖")
    
    drive_service = get_drive_service()
    
    if drive_service is None:
        st.write("Please authenticate with Google Drive to continue.")
        return

    if "messages" not in st.session_state:
        st.session_state.messages = [
            {'role': "user", "content": TASK_SPECIFIC_INSTRUCTIONS},
            {'role': "assistant", "content": "Understood, I'm ready to assist with inquiries about JKKN Educational Institutions."},
        ]
    
    if "chatbot" not in st.session_state:
        try:
            st.session_state.chatbot = ChatBot(st.session_state)
        except ValueError as e:
            st.error(f"Error initializing chatbot: {str(e)}")
            st.info("Please ensure the ANTHROPIC_API_KEY is correctly set in your .env file.")
            return

    # Display user and assistant messages skipping the first two
    for message in st.session_state.messages[2:]:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
    
    if user_msg := st.chat_input("Type your question about JKKN institutions here..."):
        st.chat_message("user").markdown(user_msg)
        
        with st.chat_message("assistant"):
            with st.spinner("JKKN Assist is thinking..."):
                response_placeholder = st.empty()
                full_response = st.session_state.chatbot.process_user_input(user_msg)
                response_placeholder.markdown(full_response)

    # Add feedback buttons
    if len(st.session_state.messages) > 2:
        col1, col2, col3 = st.columns([1,1,5])
        with col1:
            if st.button("👍"):
                st.success("Thank you for your positive feedback!")
        with col2:
            if st.button("👎"):
                st.error("We're sorry the response wasn't helpful. We'll work on improving!")

if __name__ == "__main__":
    main()