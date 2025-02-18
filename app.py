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


st.sidebar.checkbox(":globe_with_meridians: web 검색", value=False, key="bar")

st.session_state["genai_model_name"] = "Meta-Llama-3.1-8B-Instruct"
st.session_state["embed_model_name"] = "BAAI/bge-m3"


# Set LLM
llm = ChatOpenAI(
    model=st.session_state.genai_model_name,
    temperature=0,
    max_tokens=None,
    base_url=GENAI_BASE_URL,
    api_key=OPENAI_API_KEY
)


# Set Vectorstore and Retriever
if "vectorstore" not in st.session_state:
    st.session_state.vectorstore = load_vectorstore(load_path="./db/vdb_0218-0936", isWeb=st.session_state.bar, embedding_model=st.session_state.embed_model_name)
    st.session_state.retriver = st.session_state.vectorstore.as_retriever(search_kwargs={"k": 5}) # search_type="mmr": 관련성/다양성 고려 비율 설정 가능
    
retriever = st.session_state.retriver


# Set Prompt
template = """
다음 문맥을 기반으로 질문에 정확하고 신뢰할 수 있는 답변을 제공하세요:
{context}

질문: {question}

답변을 생성하기 전에 다음을 내부적으로 수행하세요:
1. 질문을 분석하여 핵심 요구 사항을 파악합니다.
2. 문맥에서 관련 정보를 식별합니다.
3. 논리적 사고를 통해 가장 적합한 답변을 도출합니다.

최종적으로 사용자에게 명확하고 간결한 답변만 제공하세요.

답변:
"""
prompt_template = ChatPromptTemplate.from_template(template)


# Configure Chain
chain = (
    {"context": retriever | format_docs, "question": RunnablePassthrough()}
    | prompt_template
    | llm
    | StrOutputParser()
)


# 1. Chat Page
def chat(): # st.Page의 첫번재 parameter는 "~.py" 파이썬 파일 명 또는 함수명
    st.info("""안녕하세요. AI 고객 지원 센터입니다.\n\n무엇을 도와드릴까요? :sunglasses:""", icon="ℹ️")
    st.divider()

    # Accept user input
    if prompt := st.chat_input("Type a message..."):
        # Display user message in chat message container
        with st.chat_message("user"):
            st.markdown(prompt)
        
               
        # Invoke RAG chain
        with st.chat_message("assistant"):
            with st.spinner('Waiting...'):
                try:   
#                     st.write(retriever.invoke(prompt)) # DEBUG: retriever works
#                     st.write(f"LLM reponse: {llm.invoke(prompt)}") # DEBUG: Internal Server Error
                    full_response = chain.invoke(prompt)

                    st.write_stream(stream_data(full_response))

                except Exception as e:
                    # Handle potential errors
                    st.write(f"Error processing the request: {e}")


# 2. Settings Page
def settings():
    st.warning("안녕하세요. 이곳은 설정페이지입니다.", icon="⚙️")
    st.divider()
    
    st.subheader(":mag: RAG 구성")
    st.write(f"- Generator: {st.session_state.genai_model_name}\n- Embedding: {st.session_state.embed_model_name}\n- Retriver: FAISS")
    
    st.subheader(f":globe_with_meridians: Web 검색")
    st.write("- ON" if st.session_state.bar else "- OFF")



pages = [st.Page(chat, title="chat", icon="💬"), st.Page(settings, title="settings", icon="⚙️")]
pg = st.navigation(pages)
pg.run()