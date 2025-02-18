import os
import torch
import streamlit as st

from langchain_community.vectorstores import FAISS
from langchain_community.vectorstores.utils import DistanceStrategy
from langchain_community.document_loaders import PyPDFDirectoryLoader

from langchain_huggingface.embeddings import HuggingFaceEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter

import requests
from bs4 import BeautifulSoup  # 웹 페이지에 들어가서 웹 피이지를 document chunk를 변환
from googlesearch import search  # 구글 검색기
from langchain.docstore.document import Document


def load_split_docs(directory_path='/projects/RAG2/kt-1/PDFs'):
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
    

def web_search_split_docs(query):
    """Search (web) and splits documents into smaller chunks."""
    try:
        web_documents = []
        for result_url in search(f'{query} site:kt.com', tld="com", num=5, stop=10, pause=2):
            print(result_url)
            print()
            print()
            
            response = requests.get(result_url)
            response.encoding = 'utf-8'
            soup = BeautifulSoup(response.text, "html.parser")
            print(soup)
            print()
            print()
            web_documents.append(soup.get_text())

        docs = [Document(page_content=web_txt, metadata={"source": "web"}) for web_txt in web_documents]
        print(f"Number of source documents: {len(docs)}")

        text_splitter = RecursiveCharacterTextSplitter(chunk_size=512, chunk_overlap=32)
        chunks = text_splitter.split_documents(docs)

        return chunks
    except Exception as e:
        print(f"Error searching web or splitting documents: {e}")
        return []
    

def load_embeddings(model_name="BAAI/bge-base-en-v1.5"): # "BAAI/bge-base-en-v1.5"
    """Loads HuggingFace embeddings model."""
    try:
        # Set embeddings model args
        model_kwargs = {"device": "cuda" if torch.cuda.is_available() else "cpu", 'trust_remote_code': True}
        encode_kwargs = {"normalize_embeddings": True} # "batch_size": 16

        # Define embeddings
        embeddings = HuggingFaceEmbeddings(model_name=model_name,
                                        model_kwargs=model_kwargs,
                                        encode_kwargs=encode_kwargs)
        
        print(f"Embeddings loaded: {embeddings != None}")

        return embeddings
    except Exception as e:
        print(f"Error loading embeddings: {e}")
        return None


def create_vectorstore(chunks, save_path="./db/faiss", embedding_model="BAAI/bge-base-en-v1.5"):
    """Creates a vectorstore using FAISS and HuggingFace embeddings."""
    try:
        # Load embeddings
        embeddings = load_embeddings(embedding_model)

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


def load_vectorstore(load_path="./db/faiss", isWeb=False, embedding_model="BAAI/bge-base-en-v1.5"):
    embeddings = load_embeddings(embedding_model)

    try:
        if os.path.exists(load_path):

            vectorstore = FAISS.load_local(load_path,
                                        embeddings,
                                        allow_dangerous_deserialization=True)
            print(f"Vectorstore loaded from {load_path}")
        else:
            chunks = load_split_docs() if not isWeb else web_search_split_docs("KT 모바일 요금제")
            vectorstore = create_vectorstore(chunks, save_path=load_path)
            print(f"Vectorstore created and saved at {load_path}")
        
        return vectorstore
    except Exception as e:
        print(f"Error loading vectorstore: {e}")
        raise