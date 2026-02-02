import os
from dotenv import load_dotenv

load_dotenv()
EMBEDDING_MODEL = "BAAI/bge-m3"
EMBEDDING_KEY = os.getenv("siliconflow_api_key")
EMBEDDING_URL = os.getenv("siliconflow_base_url")

CHAT_MODEL = "deepseek-chat"
TEMPERATURE = 0
CHAT_MODEL_KEY = os.getenv("deepseek_api_key")
CHAT_URL = os.getenv("deepseek_base_url")


CHROMA_DIR = "chroma_db"
TOP_K = 3