import os
import torch
import streamlit as st

from langchain_community.vectorstores import FAISS
from langchain_community.vectorstores.utils import DistanceStrategy
from langchain.document_loaders.pdf import PyPDFDirectoryLoader

from langchain_huggingface.embeddings import HuggingFaceEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter


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
    

def load_embeddings(model_name="BAAI/bge-base-en-v1.5"): # "BAAI/bge-multilingual-gemma2"):
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
                                        allow_dangerous_deserialization=True)
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
st.session_state.vectorstore = load_vectorstore()
st.session_state.retriver = st.session_state.vectorstore.as_retriever(search_kwargs={"k": 5}) # search_type="mmr": 관련성/다양성 고려 비율 설정 가능
