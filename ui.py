import streamlit as st
import uuid
from film_agent import app
from langchain_core.messages import HumanMessage

st.set_page_config(page_title="Film Agent", page_icon="🎬")

st.title("Film Agent 🎬")
st.markdown("Your AI movie expert. Ask me anything about movies!")

if "messages" not in st.session_state:
    st.session_state.messages = []

if "thread_id" not in st.session_state:
    st.session_state.thread_id = str(uuid.uuid4())

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

if prompt := st.chat_input(
    "What movie are you looking for? (e.g., 'Horror with a hockey mask killer')"
):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        message_placeholder = st.empty()
        status_placeholder = st.empty()
        full_response = ""

        with st.spinner("Thinking..."):
            try:
                inputs = {
                    "question": prompt,
                    "synthesized_query": prompt,
                    "retry_count": 0,
                    "context": "",
                    "is_relevant": "no",
                    "chat_history": [HumanMessage(content=prompt)],
                }

                config = {"configurable": {"thread_id": st.session_state.thread_id}}

                for event in app.stream(inputs, config=config):
                    for node, values in event.items():
                        if node == "retrieve":
                            status_placeholder.text("🔍 Searching movies...")
                        elif node == "grade_documents":
                            status_placeholder.text("⚖️ Grading relevance...")
                        elif node == "rewrite_query":
                            status_placeholder.text("🔄 Refining search query...")
                        elif node == "generate":
                            status_placeholder.empty()
                            full_response = values["generation"]
                            message_placeholder.markdown(full_response)

                if full_response:
                    st.session_state.messages.append(
                        {"role": "assistant", "content": full_response}
                    )
                else:
                    status_placeholder.error("Failed to generate a response.")

            except Exception as e:
                status_placeholder.empty()
                st.error(f"An error occurred: {e}")
