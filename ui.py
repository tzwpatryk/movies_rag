import streamlit as st
from film_agent import app

st.set_page_config(page_title="Film Agent", page_icon="🎬")

st.title("Film Agent 🎬")
st.markdown("Your AI movie expert. Ask me anything about movies!")

if "messages" not in st.session_state:
    st.session_state.messages = []

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

if prompt := st.chat_input("What movie are you looking for? (e.g., 'Horror with a hockey mask killer')"):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        with st.spinner("Searching for movies..."):
            try:
                inputs = {
                    "question": prompt,
                    "original_question": prompt,
                    "retry_count": 0,
                    "context": "",
                    "is_relevant": "no",
                }
                result = app.invoke(inputs)
                response = result.get("generation", "Sorry, I couldn't generate an answer.")
                
                st.markdown(response)
                st.session_state.messages.append({"role": "assistant", "content": response})
            except Exception as e:
                st.error(f"An error occurred: {e}")
