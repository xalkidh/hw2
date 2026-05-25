import os
from dotenv import load_dotenv
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_community.vectorstores import Chroma
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import TextLoader, DirectoryLoader

load_dotenv()

DOCUMENTS_PATH = "data/documents"
VECTOR_STORE_PATH = "data/vector_store"

def build_vector_store():
    loader = DirectoryLoader(DOCUMENTS_PATH, glob="*.txt", loader_cls=TextLoader)
    documents = loader.load()

    splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    chunks = splitter.split_documents(documents)

    embeddings = GoogleGenerativeAIEmbeddings(
        model="models/embedding-001",
        google_api_key=os.getenv("GOOGLE_API_KEY")
    )

    vector_store = Chroma.from_documents(
        chunks,
        embeddings,
        persist_directory=VECTOR_STORE_PATH
    )

    print(f"Vector store built with {len(chunks)} chunks.")
    return vector_store

def load_vector_store():
    embeddings = GoogleGenerativeAIEmbeddings(
        model="models/embedding-001",
        google_api_key=os.getenv("GOOGLE_API_KEY")
    )

    vector_store = Chroma(
        persist_directory=VECTOR_STORE_PATH,
        embedding_function=embeddings
    )

    return vector_store

def retrieve(query: str, k: int = 3) -> str:
    vector_store = load_vector_store()
    results = vector_store.similarity_search(query, k=k)
    return "\n\n".join([doc.page_content for doc in results])