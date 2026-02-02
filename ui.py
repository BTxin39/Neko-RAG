import streamlit as st
import requests

API_URL = "http://127.0.0.1:8000/chat/stream"

st.set_page_config(page_title="Neko RAG",
                   page_icon="C:/Users/BTxin/Pictures/logo/neko rag.jpg",
                   layout="centered")
st.title("🤖 Neko RAG 聊天")

# session_id（浏览器级别）
if "session_id" not in st.session_state:
    st.session_state.session_id = "user_streamlit"

if "messages" not in st.session_state:
    st.session_state.messages = []

# 显示历史消息
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# 输入框
prompt = st.chat_input("请输入你的问题")

if prompt:
    # 记录用户消息
    st.session_state.messages.append({
        "role": "user",
        "content": prompt
    })

    with st.chat_message("user"):
        st.markdown(prompt)

    # AI 输出占位符
    with st.chat_message("assistant"):
        placeholder = st.empty()
        full_text = ""

        # 调用 FastAPI streaming 接口
        response = requests.post(
            API_URL,
            json={
                "session_id": st.session_state.session_id,
                "question": prompt
            },
            stream=True
        )

        for chunk in response.iter_content(chunk_size=1024):
            if chunk:
                text = chunk.decode("utf-8")
                full_text += text
                placeholder.markdown(full_text)

    # 记录 AI 消息
    st.session_state.messages.append({
        "role": "assistant",
        "content": full_text
    })
