import os
import shutil

from dotenv import load_dotenv
from langchain_openai import OpenAIEmbeddings
from langchain_community.document_loaders import (
    PyPDFLoader, 
    TextLoader, 
    UnstructuredMarkdownLoader,
    WebBaseLoader
)
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma

# 加载环境变量
load_dotenv()
sf_key = os.getenv("siliconflow_api_key")
sf_url = os.getenv("siliconflow_base_url")

class BUILD_DB:
    def __init__(self, db_directory="./chroma_db"):
        print("[debug] RAG系统初始化...")
        self.db_directory = db_directory
        self.embeddings = OpenAIEmbeddings(
            model="BAAI/bge-m3", 
            openai_api_key=sf_key, 
            openai_api_base=sf_url
        )
        self.vector_store = self._get_or_create_db()
        # 定义支持的文件后缀
        self.supported_extensions = ['.pdf', '.md', '.txt']

    def _get_or_create_db(self):
        return Chroma(
            persist_directory=self.db_directory,
            embedding_function=self.embeddings
        )

    def process_directory(self, dir_path: str):
        """遍历文件夹并处理所有支持的文件"""
        if not os.path.isdir(dir_path):
            print("[wrong]")
            return

        print("[debug]")
        
        # 获取所有待处理文件路径
        files_to_process = []
        for root, dirs, files in os.walk(dir_path):
            for file in files:
                ext = os.path.splitext(file)[1].lower()
                if ext in self.supported_extensions:
                    files_to_process.append(os.path.join(root, file))
        
        total_files = len(files_to_process)
        print(f"[debug] 共{total_files} 个文件")

        # 批量处理
        for index, file_path in enumerate(files_to_process):
            print(f"[debug] 进度：[{index + 1}/{total_files}]")
            self.process_document(file_path)
        
        print("[debug] 处理完毕")

    def process_document(self, source_path: str):
        """处理单个文档或网页"""
        try:
            if source_path.startswith(('http://', 'https://')):
                docs = self._web_loader(source_path)
            elif source_path.lower().endswith('.pdf'):
                docs = self._pdf_loader(source_path)
            elif source_path.lower().endswith('.md'):
                docs = self._markdown_loader(source_path)
            elif source_path.lower().endswith('.txt'):
                docs = self._text_loader(source_path)
            else:
                return # 不支持的格式直接跳过

            text_splitter = RecursiveCharacterTextSplitter(chunk_size=800, chunk_overlap=150)
            splits = text_splitter.split_documents(docs)
            
            if splits:
                self.vector_store.add_documents(documents=splits)
                print(f"[debug] 已存入: {os.path.basename(source_path)}")
        except:
            print("[wrong]")

    def _pdf_loader(self, path):
        return PyPDFLoader(path).load()

    def _text_loader(self, path):
        return TextLoader(path, encoding='utf-8').load()

    def _markdown_loader(self, path):
        try: 
            return UnstructuredMarkdownLoader(path).load()
        except: 
            return self._text_loader(path)

    def _web_loader(self, url):
        loader = WebBaseLoader(url)
        loader.requests_kwargs = {'headers': {'User-Agent': 'Mozilla/5.0'}}
        return loader.load()
    

# =========================================================
if __name__ == "__main__":
    if not sf_url or not sf_key:
        print("[wrong] 环境变量缺失")
    else:
        db_manager = BUILD_DB()
        print("\n--- 向量数据库管理系统 ---")
        print("[system] 支持：单个文件路径、文件夹路径、网页URL")
        
        while True:
            user_input = input("\n[system] 请输入路径 (exit/quit): ").strip()
            if user_input.lower() in ['exit', 'quit']: break

            if not user_input: continue
            
            if user_input.startswith('http'):
                db_manager.process_document(user_input)
            elif os.path.isdir(user_input):
                db_manager.process_directory(user_input)
            elif os.path.isfile(user_input):
                db_manager.process_document(user_input)
            else:
                print(f"[wrong] 路径不存在或无效: {user_input}")