import os
from dotenv import load_dotenv
load_dotenv()

OPENAI_API_KEY = os.environ.get("OPEN_API_KEY")
GENAI_BASE_URL = os.environ.get("OPENAI_BASE_URL")

import streamlit as st

from utils.database import *
from utils.etc import format_docs, stream_data

from langchain_openai import ChatOpenAI
from langchain.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser


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
st.session_state["embed_model_name"] = "BAAI/bge-multilingual-gemma2"


# Set LLM
llm = ChatOpenAI(
    model=st.session_state.genai_model_name,
    temperature=0,
    max_tokens=None,
    base_url=GENAI_BASE_URL,
    api_key=OPENAI_API_KEY
)

# Set Prompt
template = """
다음 문맥을 기반으로 질문에 답변하세요:
{context}

질문: {question}

답변을 단계적으로 사고 과정을 통해 생성하세요:
1. 질문을 분석합니다.
2. 문맥에서 관련 정보를 식별합니다.
3. 질문에 답하기 위한 논리적 단계를 나열합니다.
4. 최종적으로 질문에 대한 명확하고 간결한 답을 제공합니다.

답변:
"""

prompt_template = ChatPromptTemplate.from_template(template)

retriever = st.session_state.retriver

chain = (
    {"context": retriever | format_docs, "question": RunnablePassthrough()}
    | prompt_template
    | llm
    | StrOutputParser()
)


# 1. Chat Page
def chat(): # st.Page의 첫번재 parameter는 "~.py" 파이썬 파일 명 또는 함수명
    # st.title("AI 고객 지원 센터")
    st.info("""안녕하세요. AI 고객 지원 센터입니다. 무엇을 도와드릴까요? :sunglasses:""", icon="ℹ️")
    st.divider()

    # Accept user input
    if prompt := st.chat_input("Type a message..."):
        # Display user message in chat message container
        with st.chat_message("user"):
            st.markdown(prompt)
        
        try:
            # Invoke RAG chain
            with st.chat_message("assistant"):
                with st.spinner('Waiting...'):
                    full_response = chain.invoke(prompt)
                
                st.write_stream(stream_data(full_response))

        except Exception as e:
            # Handle potential errors
            st.write(f"Error processing the request: {e}")


# 2. Settings Page
def settings():
    # st.title("AI 고객 지원 센터: 설정")
    st.warning("안녕하세요. 이곳은 설정페이지입니다.", icon="⚙️")
    st.divider()
    
    st.subheader(":mag: RAG 구성")
    st.write(f"- Generator: {st.session_state.genai_model_name}\n- Embedding: {st.session_state.embed_model_name}\n- Retriver: FAISS")
    
    st.subheader(f":globe_with_meridians: Web 검색")
    st.write("- ON" if st.session_state.bar else "- OFF")


# Widgets shared by all the pages
st.sidebar.checkbox(":globe_with_meridians: web 검색", [True, False], key="bar")


pages = [st.Page(chat, title="chat", icon="💬"), st.Page(settings, title="settings", icon="⚙️")]
pg = st.navigation(pages)
pg.run()