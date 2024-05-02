import streamlit as st

import requests

# App title
st.set_page_config(page_title="LangChain Chatbot")


# Store LLM generated responses
if "messages" not in st.session_state.keys():
    st.session_state.messages = [
        {"role": "assistant", "content": "How may I help you?"}
    ]

# Display chat messages
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.write(message["content"])


# Function for generating LLM response
def generate_response(prompt_input):
    try:
        inputs = {"input": prompt_input}
        response = requests.post("http://localhost:8001/query", json=inputs, timeout=30)
        response.raise_for_status()  # Check for HTTP errors
        response_data = response.json()
        return response_data.get("answer", "Sorry, I couldn't process that request.")
    except requests.exceptions.RequestException as e:
        print(f"Request failed: {e}")
        return "I'm sorry, but I'm having trouble understanding you right now."


# User-provided prompt
if prompt := st.chat_input():
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.write(prompt)

# Generate a new response if last message is not from assistant
if st.session_state.messages[-1]["role"] != "assistant":
    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            response = generate_response(prompt)
            st.write(response)
    message = {"role": "assistant", "content": response}
    st.session_state.messages.append(message)
