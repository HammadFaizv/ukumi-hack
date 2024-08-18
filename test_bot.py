import streamlit as st
from chatbot import chatbot_answer
from test_bot2 import generate_response

def chat_bot():
    transcript = ""
    with open("temp/trans.txt", "r") as file:
                transcript = file.read()
            
    st.title("Youtube Chatbot")

            # Chat history
    if 'chat_history' not in st.session_state:
        st.session_state['chat_history'] = []

            # User input
    user_input = st.text_input("You: ", "")

    if user_input:
        
        st.session_state['chat_history'].append(f"You: {user_input}")

        # Generate response
        response = generate_response(transcript, user_input)
        st.session_state['chat_history'].append(f"Bot: {response}")

    # Display chat history
    for message in st.session_state['chat_history']:
        st.write(message)