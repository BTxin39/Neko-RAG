# DeepSeek RAG Demo

一个基于 LangChain 的 RAG 问答系统：

- 🔍 Chroma 向量数据库
- 🧠 硅基流动 Embedding（bge-m3）
- 💬 DeepSeek Chat
- ⚡ FastAPI + Streaming
- 🖥 Streamlit 前端

## 功能特性

- **检索增强生成 (RAG)**: 结合向量检索和生成式 AI，提供准确的问答服务
- **会话管理**: 支持多轮对话，自动总结历史记录
- **流式响应**: 实时流式输出，提升用户体验
- **文档处理**: 支持 PDF 和 Markdown 文档的向量化存储
- **模块化设计**: 核心逻辑分离，便于扩展和维护

## 快速开始

### 环境要求

- Python 3.8+
- pip

### 安装依赖

```bash
pip install -r requirements.txt
```

### 环境变量配置

复制 `.env.example` 到 `.env` 并填写你的 API 密钥：

```bash
cp .env.example .env
```

编辑 `.env` 文件，设置以下变量：

```env
deepseek_api_key="你的 DeepSeek API 密钥"
deepseek_base_url="https://api.deepseek.com/v1"
siliconflow_api_key="你的硅基流动 API 密钥"
siliconflow_base_url="https://api.siliconflow.cn/v1"
```

### 构建向量数据库

运行脚本构建文档向量数据库：

```bash
python scripts/build_db.py
```

### 运行后端服务

启动 FastAPI 服务：

```bash
uvicorn api:app --reload
```

服务将在 `http://127.0.0.1:8000` 启动。

### 运行前端界面

在新终端中启动 Streamlit 前端：

```bash
streamlit run ui.py
```

前端将在浏览器中打开。

## 使用方法

1. 打开 Streamlit 前端
2. 在聊天框中输入问题
3. 系统将基于文档内容生成回答
4. 支持多轮对话，系统会记住上下文

## API 接口

### POST /chat

同步问答接口

**请求体：**
```json
{
  "session_id": "string",
  "question": "string"
}
```

**响应：**
```json
{
  "answer": "string"
}
```

### POST /chat/stream

流式问答接口

**请求体：** 同上

**响应：** 服务器发送事件 (SSE) 流

### GET /health

健康检查接口

**响应：**
```json
{
  "status": "ok"
}
```

## 项目结构

```
.
├── api.py              # FastAPI 服务
├── ui.py               # Streamlit 前端
├── requirements.txt    # Python 依赖
├── .env.example        # 环境变量示例
├── app/
│   ├── assistant.py    # DeepseekRAG 核心逻辑
│   ├── config.py       # 配置管理
│   ├── embedding.py    # 嵌入模型
│   ├── exceptions.py   # 自定义异常
│   ├── llm.py          # LLM 模型
│   ├── memory.py       # 会话内存管理
│   ├── rag_chain.py    # RAG 链构建
│   ├── summarizer.py   # 对话总结
│   └── vectorstore.py  # 向量存储
├── scripts/
│   └── build_db.py     # 构建向量数据库脚本
├── test/
│   ├── qa.py           # QA 测试
│   └── test.py         # 单元测试
├── chroma_db/          # Chroma 向量数据库
└── __pycache__/        # Python 缓存
```

## 依赖包

主要依赖包包括：

- `langchain`: LangChain 框架
- `chromadb`: Chroma 向量数据库
- `fastapi`: Web 框架
- `streamlit`: 前端框架
- `pydantic`: 数据验证
- `python-dotenv`: 环境变量管理
- 其他工具包...

完整依赖请查看 `requirements.txt`。

## 开发

### 运行测试

```bash
python -m pytest test/
```

## 贡献

欢迎提交 Issue 和 Pull Request！

## 许可证

MIT License
