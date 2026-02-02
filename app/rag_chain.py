from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.output_parsers import StrOutputParser
from app.vectorstore import get_retriever
from app.llm import get_llm

def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)

def build_rag_chain(chat_history, summary, requirement):
    prompt = ChatPromptTemplate.from_messages([
        ("system", """你是一个专业的文档分析师。
请根据提供的【参考文档】、【对话摘要】和【最近对话历史】来回答问题。

【对话摘要（背景信息）】：
{summary}
"""),
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
            "context": (lambda x: x["question"]) | get_retriever() | format_docs,
            "question": lambda x: x["question"],
            "chat_history": lambda x: x["chat_history"],
            "summary": lambda x: x["summary"],
            "requirement": lambda x: x["requirement"]
        }
        | prompt
        | get_llm()
        | StrOutputParser()
    )

    return chain
