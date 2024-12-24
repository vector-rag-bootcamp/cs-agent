import os
from getpass import getpass

from dotenv import load_dotenv
load_dotenv()

GENERATOR_BASE_URL = os.environ.get("OPENAI_BASE_URL")
OPENAI_API_KEY = os.environ.get("OPEN_API_KEY")

EMBEDDING_MODEL_NAME = "BAAI/bge-base-en-v1.5"

from langchain_community.document_loaders import WebBaseLoader
from langchain_community.vectorstores import FAISS
from langchain_openai.chat_models import ChatOpenAI
from langchain_openai.embeddings import OpenAIEmbeddings
from langchain_huggingface.embeddings import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter


loader = WebBaseLoader("https://lilianweng.github.io/posts/2023-06-23-agent/")
data = loader.load()

# Split
text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=0)
all_splits = text_splitter.split_documents(data)

model_kwargs = {'device': 'cuda', 'trust_remote_code': True}
encode_kwargs = {'normalize_embeddings': True} # set True to compute cosine similarity

print(f"Setting up the embeddings model...")
embeddings = HuggingFaceEmbeddings(
    model_name=EMBEDDING_MODEL_NAME,
    model_kwargs=model_kwargs,
    encode_kwargs=encode_kwargs,
)

# Store splits
vectorstore = FAISS.from_documents(documents=all_splits, embedding=embeddings)

# LLM
llm = ChatOpenAI(model="Meta-Llama-3.1-8B-Instruct",
                 temperature=0,
                 max_tokens=None,
                 base_url=GENERATOR_BASE_URL,
                 headers={"User-Agent": "MyApp/1.0"},
                 api_key=OPENAI_API_KEY
                 )


from langchain.prompts import PromptTemplate
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain


# 프롬프트 템플릿 정의 (LangChain Hub의 템플릿을 로컬에서 구현)
retrieval_qa_chat_prompt = PromptTemplate(
    input_variables=["context", "question"],
    template=(
        "Use the following context to answer the question.\n\n"
        "Context:\n{context}\n\n"
        "Question: {question}\n\n"
        "Answer:"
    ),
)


combine_docs_chain = create_stuff_documents_chain(llm, retrieval_qa_chat_prompt)
rag_chain = create_retrieval_chain(vectorstore.as_retriever(), combine_docs_chain)

response = rag_chain.invoke({"input": "What are autonomous agents?"})

print(response["output"])