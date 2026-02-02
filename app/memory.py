from langchain_core.messages import HumanMessage, AIMessage

class SessionMemory:
    def __init__(self, max_turns=6):
        self.max_turns = max_turns
        self.sessions = {}

    def get(self, session_id: str):
        return self.sessions.setdefault(
            session_id,
            {
                "history": [],
                "summary": ""
            }
        )

    def append(self, session_id: str, human: str, ai: str):
        session = self.get(session_id)
        session["history"].append(HumanMessage(content=human))
        session["history"].append(AIMessage(content=ai))

    def need_summarize(self, session_id: str) -> bool:
        return len(self.get(session_id)["history"]) >= self.max_turns

    def clear_history(self, session_id: str):
        self.get(session_id)["history"] = []
