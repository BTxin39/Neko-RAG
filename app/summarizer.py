from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.output_parsers import StrOutputParser
from app.llm import get_llm

def summarize_chat(history, old_summary=""):
    if not history:
        return old_summary

    prompt = ChatPromptTemplate.from_messages([
        MessagesPlaceholder(variable_name="chat_history"),
        ("human", f"""
请将以上对话与已有摘要（如果有：{old_summary}）合并，
压缩成一段简洁、保留关键信息的背景摘要，
用于后续对话。请直接输出摘要内容。
""")
    ])

    chain = prompt | get_llm() | StrOutputParser()
    return chain.invoke({"chat_history": history})
