import os
from dotenv import load_dotenv
from langchain_chroma import Chroma
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.output_parsers import StrOutputParser
from langchain_core.messages import HumanMessage, AIMessage

# 加载环境变量
load_dotenv()
sf_key = os.getenv("siliconflow_api_key")
sf_url = os.getenv("siliconflow_base_url")
ds_key = os.getenv("deepseek_api_key")
ds_url = os.getenv("deepseek_base_url")

class DeepseekRAG:
    def __init__(self):
        self.embeddings = OpenAIEmbeddings(
            model="BAAI/bge-m3", 
            openai_api_key=sf_key, 
            openai_api_base=sf_url
        )

        self.llm = ChatOpenAI(
            model='deepseek-chat', 
            openai_api_key=ds_key, 
            openai_api_base=ds_url,
            temperature=0.3
        )

        self.vector_store = Chroma(
            persist_directory="./chroma_db",
            embedding_function=self.embeddings
        )
        
        self.retriever = self.vector_store.as_retriever(
            search_type="similarity",
            search_kwargs={"k": 3}
        )

        self.chat_history = []
        self.summary = ""
        self.requirement = "无"

    def _format_docs(self, docs):
        return "\n\n".join(doc.page_content for doc in docs)

    def _summarize_history(self):
        if not self.chat_history:
            return

        print("[debug] 正在压缩记忆...")
        summary_prompt = ChatPromptTemplate.from_messages([
            MessagesPlaceholder(variable_name="chat_history"),
            ("user", f"""
             请根据以上的对话历史，结合现有的摘要内容（如果有：{self.summary}），总结成一段简短的背景信息，保留关键事实，以便后续对话使用。请直接输出总结后的文本。
             """)
        ])
        
        summarize_chain = summary_prompt | self.llm | StrOutputParser()
        
        self.summary = summarize_chain.invoke({"chat_history": self.chat_history})
        
        self.chat_history = []

    def answer(self, question: str):
        if len(self.chat_history) >= 6:
            self._summarize_history()

        prompt = ChatPromptTemplate.from_messages([
            ("system", """你是一个专业的文档分析师。
             请根据提供的【参考文档】、【对话摘要】和【最近对话历史】来回答问题。
             
             【对话摘要（背景信息）】：
             {summary}
             
             要求：
             - 简洁准确，资料中没有的内容请说"不知道"。
             - 如果【对话摘要】中包含用户之前的偏好或信息，请予以参考。"""),
            MessagesPlaceholder(variable_name="chat_history"),
            ("human", """
            - 参考文档内容：
            {context}
            
            - 额外要求：
            {requirement}
            
            - 当前问题：
            {question}
            """)
        ])

        chain = (
            {
                "context": (lambda x: x["question"]) | self.retriever | self._format_docs,
                "requirement": lambda x: x["requirement"],
                "question": lambda x: x["question"],
                "chat_history": lambda x: x["chat_history"],
                "summary": lambda x: x["summary"]
            }
            | prompt
            | self.llm
            | StrOutputParser()
        )

        result = chain.invoke({
            "question": question,
            "requirement": self.requirement,
            "chat_history": self.chat_history,
            "summary": self.summary if self.summary else "暂无背景信息"
        })

        self.chat_history.append(HumanMessage(content=question))
        self.chat_history.append(AIMessage(content=result))

        return result

# ================= 持续对话入口 =================
if __name__ == "__main__":
    rag = DeepseekRAG()
    
    print("\n" + "="*40)
    print("[debug] DeepSeek RAG 初始化完毕")
    print("输入 'exit' 或 'quit' 退出对话")
    print("输入 'clear' 清空所有记忆")
    print("="*40 + "\n")

    while True:
        user_input = input("[local] 用户: ").strip()
        
        if not user_input:
            continue
            
        if user_input.lower() in ['exit', 'quit']:
            print("[system] 回答：再见！")
            break
            
        if user_input.lower() == 'clear':
            rag.chat_history = []
            rag.summary = ""
            print("[system] 记忆已完全清空")
            continue

        try:
            response = rag.answer(user_input)
            print(f"[system] 回答: {response}\n")
        except Exception as e:
            print(f"[wrong] 错误原因: {e}")