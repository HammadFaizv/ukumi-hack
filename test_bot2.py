# import streamlit as st
import openai

client = openai.OpenAI(api_key='api_key')
# transcript = ""

# with open('temp/trans.txt', 'r') as file:
#     transcript = file.read()

# Function to generate chatbot response
def generate_response(transcript, prompt):
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role":"system", "content": "You are an assistant who will help answer the questions based on a transcript"},
            {"role": "system", "content": "Here is a transcript of the video: " + transcript},
            {"role": "user", "content": "Answer this question: " + prompt}]
    )

    answer = response.choices[0].message.content
    print(answer)
    return answer

# Streamlit app layout
# st.title("Streamlit Chatbot")

# # Chat history
# if 'chat_history' not in st.session_state:
#     st.session_state['chat_history'] = []

# # User input
# user_input = st.text_input("You: ", "")

# if user_input:
#     # Add user input to chat history
#     st.session_state['chat_history'].append(f"You: {user_input}")

#     # Generate response
#     response = generate_response(transcript, user_input)
#     st.session_state['chat_history'].append(f"Bot: {response}")

# # Display chat history
# for message in st.session_state['chat_history']:
#     st.write(message)