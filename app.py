import streamlit as st

import os
from dotenv import load_dotenv
load_dotenv()


OPENAI_API_KEY = os.environ.get("OPEN_API_KEY")
GENAI_BASE_URL = os.environ.get("OPENAI_BASE_URL")


st.title("AI Customer Service Center")

st.markdown(
    """
    <style>
    [data-testid="stChatMessageAvatarUser"] {
        background-color: #28cdc8 !important;
    }
    [data-testid="stChatMessageAvatarAssistant"] {
        background-color: #ffe65a !important;
    }
    </style>
    """,
    unsafe_allow_html=True
)


st.session_state["genai_model_name"] = "Meta-Llama-3.1-8B-Instruct"

# Initialize chat history
default_message = { "role": "assistant", "content": "안녕하슈. 메아리봇이올시다."}
if "messages" not in st.session_state:
    st.session_state.messages = [default_message]

from langchain_openai.chat_models import ChatOpenAI

llm = ChatOpenAI(model=st.session_state.genai_model_name,
                    temperature=0,
                    base_url=GENAI_BASE_URL)
# Pages, Sidebar
def chat(): # st.Page의 첫번재 parameter는 "~.py" 파이썬 파일 명 또는 함수명
    # Display chat messages from history on app rerun
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    # Accept user input
    if prompt := st.chat_input("What is up?"):
        # Display user message in chat message container
        with st.chat_message("user"):
            st.markdown(prompt)
        # Add user message to chat history
        st.session_state.messages.append({"role": "user", "content": prompt})

        # Display assistant response in chat message container
        full_response = llm.invoke(prompt)
        with st.chat_message("assistant"):
            st.write(full_response)

        # def openai_stream():
        #     response = llm.stream([
        #             {"role": m["role"], "content": m["content"]}
        #             for m in st.session_state.messages
        #         ])
        #     for chunk in response:
        #         yield chunk.content

        # with st.chat_message("assistant"):
        #     full_response = st.write_stream(openai_stream())

        # Add assistant response to chat history
        st.session_state.messages.append({"role": "assistant", "content": full_response})

def settings():
    st.write(st.session_state.bar)

# Widgets shared by all the pages
st.sidebar.checkbox("Bar", [True, False], key="bar")

pages = [st.Page(chat, title="chat", icon="💬"), st.Page(settings, title="settings", icon="⚙️")]
pg = st.navigation(pages)
pg.run()






