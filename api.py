from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from app.assistant import DeepseekRAG
from app.exceptions import RAGException
import logging
from fastapi.responses import StreamingResponse


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="DeepSeek RAG Service")

rag = DeepseekRAG()

class ChatRequest(BaseModel):
    session_id: str = Field(..., min_length=1)
    question: str = Field(..., min_length=1)

class ChatResponse(BaseModel):
    answer: str

@app.post("/chat", response_model=ChatResponse)
def chat(req: ChatRequest):
    try:
        logger.info(f"[chat] session={req.session_id}, question={req.question}")

        answer = rag.answer(
            question=req.question,
            session_id=req.session_id
        )
        return {"answer": answer}

    except RAGException as e:
        logger.error(f"[RAG ERROR] {e}")
        raise HTTPException(status_code=400, detail=str(e))

    except Exception as e:
        logger.exception("[UNEXPECTED ERROR]")
        raise HTTPException(status_code=500, detail="内部服务错误")
    
@app.get("/health")
def health():
    return {"status": "ok"}

@app.post("/chat/stream")
def chat_stream(req: ChatRequest):

    def token_generator():
        try:
            for token in rag.stream_answer(
                question=req.question,
                session_id=req.session_id
            ):
                yield token
        except Exception as e:
            yield f"\n[ERROR] {str(e)}"

    return StreamingResponse(
        token_generator(),
        media_type="text/plain"
    )
