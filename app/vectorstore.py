from langchain_chroma import Chroma
from app.config import CHROMA_DIR, TOP_K
from app.embedding import get_embeddings

def get_vectorstore():
    return Chroma(
        persist_directory=CHROMA_DIR,
        embedding_function=get_embeddings()
    )

def get_retriever():
    return get_vectorstore().as_retriever(
        search_type="similarity",
        search_kwargs={"k": TOP_K}
    )
