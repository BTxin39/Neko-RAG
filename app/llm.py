from langchain_openai import ChatOpenAI
from app.config import CHAT_URL, CHAT_MODEL_KEY, TEMPERATURE, CHAT_MODEL

def get_llm():
    return ChatOpenAI(
            model=CHAT_MODEL, 
            openai_api_key=CHAT_MODEL_KEY, 
            openai_api_base=CHAT_URL,
            temperature=TEMPERATURE
        )