from langchain_openai import OpenAIEmbeddings

from app.config import EMBEDDING_MODEL, EMBEDDING_KEY, EMBEDDING_URL

def get_embeddings():
    return OpenAIEmbeddings(
            model=EMBEDDING_MODEL, 
            openai_api_key=EMBEDDING_KEY, 
            openai_api_base=EMBEDDING_URL
        )