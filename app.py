import streamlit as st
import os
from dotenv import load_dotenv
import torch
load_dotenv()

OPENAI_API_KEY = os.environ.get("OPEN_API_KEY")
GENAI_BASE_URL = os.environ.get("OPENAI_BASE_URL")

from langchain_community.vectorstores import FAISS
from langchain_community.vectorstores.utils import DistanceStrategy
from langchain.document_loaders.pdf import PyPDFDirectoryLoader
from langchain_huggingface.embeddings import HuggingFaceEmbeddings
from langchain_openai.chat_models import ChatOpenAI
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.prompts import PromptTemplate
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain


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
default_message = { "role": "assistant", "content": "안녕하세요, KT AI 고객센터입니다. 무엇을 도와드릴까요?" }
if "messages" not in st.session_state:
    st.session_state.messages = [default_message]


def load_split_docs(directory_path='./docs'):
    """Loads and splits documents into smaller chunks."""
    try:
        documents = PyPDFDirectoryLoader(directory_path).load()
        print(f"Number of source documents: {len(documents)}")

        # Split documents into chunks
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=512, chunk_overlap=32)
        chunks = text_splitter.split_documents(documents)
        print(f"Number of text chunks: {len(chunks)}")

        return chunks
    except Exception as e:
        print(f"Error loading or splitting documents: {e}")
        return []
    

def load_embeddings(model_name="sentence-transformers/paraphrase-MiniLM-L6-v2"):
    """Loads HuggingFace embeddings model."""
    try:
        # Set embeddings model args
        model_kwars = {"device": "cuda" if torch.cuda.is_available() else "cpu", 'trust_remote_code': True}
        encode_kwargs = {"normalize_embeddings": True} # "batch_size": 16

        # Define embeddings
        embeddings = HuggingFaceEmbeddings(model_name=model_name,
                                        model_kwargs=model_kwars,
                                        encode_kwargs=encode_kwargs)
        print(f"Embeddings loaded: {embeddings != None}")

        return embeddings
    except Exception as e:
        print(f"Error loading embeddings: {e}")
        return None


def create_vectorstore(chunks, save_path="./db/faiss"):
    """Creates a vectorstore using FAISS and HuggingFace embeddings."""
    try:
        # Load embeddings
        embeddings = load_embeddings()

        # Index embeddings
        vectorstore = FAISS.from_documents(chunks, embeddings, distance_strategy=DistanceStrategy.COSINE)
        print(f"Vectorstore created.")

        if save_path != "": 
            vectorstore.save_local(save_path)  # save db in local
            print(f"Vectorstore saved locally at {save_path}")

        return vectorstore
    except Exception as e:
        print(f"Error creating vectorstore: {e}")
        raise


def load_vectorstore(load_path="./db/faiss"):
    embeddings = load_embeddings()

    try:
        if os.path.exists(load_path):

            vectorstore = FAISS.load_local(load_path,
                                        embeddings,
                                        allow_dangerous_deserialization=True )
            print(f"Vectorstore loaded from {load_path}")
        else:
            chunks = load_split_docs()
            vectorstore = create_vectorstore(chunks, save_path=load_path)
            print(f"Vectorstore created and saved at {load_path}")
        
        return vectorstore
    except Exception as e:
        print(f"Error loading vectorstore: {e}")
        raise


# Set Vectorstore and Retriever
vs = load_vectorstore()
retriever = vs.as_retriever(search_kwargs={"k": 5}) # search_type="mmr": 관련성/다양성 고려 비율 설정 가능


# Set LLM
llm = ChatOpenAI(model=st.session_state.genai_model_name,
                    temperature=0,
                    base_url=GENAI_BASE_URL)

# Set Prompt
retrieval_qa_chat_prompt = PromptTemplate(
    input_variables=["context", "input"],
    template=(
        "Use the following context to answer the question in korean.\n\n"
        "Chat History\n{chat_history}\n\n"
        "Context:\n{context}\n\n"
        "Question: {input}\n\n"
        "Answer:"
    ),
)

# Set Chains
combine_docs_chain = create_stuff_documents_chain(llm, retrieval_qa_chat_prompt)
rag_chain = create_retrieval_chain(retriever, combine_docs_chain)


def format_chat_history(messages):
    """Formats chat history into a single string for RAG input."""
    formatted_history = ""
    for msg in messages:
        role = "User" if msg["role"] == "user" else "Assistant"
        formatted_history += f"{role}: {msg['content']}\n"
    return formatted_history.strip()


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

        # Format chat history
        chat_history = format_chat_history(st.session_state.messages)

        # Combine chat history with context
        input_data = {
            "input": prompt,
            "chat_history": chat_history,
        }

        # Invoke RAG chain
        response = rag_chain.invoke(input_data)
        full_response = response['answer']
        # full_response = llm.invoke(prompt)
        with st.chat_message("assistant"):
            st.write(full_response)

        # Add assistant response to chat history
        st.session_state.messages.append({"role": "assistant", "content": full_response})


def settings():
    st.write(st.session_state.bar)


# Widgets shared by all the pages
st.sidebar.checkbox("Bar", [True, False], key="bar")

pages = [st.Page(chat, title="chat", icon="💬"), st.Page(settings, title="settings", icon="⚙️")]
pg = st.navigation(pages)
pg.run()






