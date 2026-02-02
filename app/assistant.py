from app.rag_chain import build_rag_chain
from app.summarizer import summarize_chat
from app.memory import SessionMemory

class DeepseekRAG:
    def __init__(self):
        self.memory = SessionMemory(max_turns=6)

    def answer(self, question: str, session_id: str):
        session = self.memory.get(session_id)

        if self.memory.need_summarize(session_id):
            session["summary"] = summarize_chat(
                session["history"],
                session["summary"]
            )
            self.memory.clear_history(session_id)

        chain = build_rag_chain(
            session["history"],
            session["summary"] or "暂无背景信息",
            "无"
        )

        result = chain.invoke({
            "question": question,
            "chat_history": session["history"],
            "summary": session["summary"],
            "requirement": "无"
        })

        self.memory.append(session_id, question, result)
        return result

    def stream_answer(self, question: str, session_id: str):
        session = self.memory.get(session_id)

        if self.memory.need_summarize(session_id):
            session["summary"] = summarize_chat(
                session["history"],
                session["summary"]
            )
            self.memory.clear_history(session_id)

        chain = build_rag_chain(
            session["history"],
            session["summary"] or "暂无背景信息",
            "无"
        )

        # 关键：stream 而不是 invoke
        stream = chain.stream({
            "question": question,
            "chat_history": session["history"],
            "summary": session["summary"],
            "requirement": """
角色定义与对话规则（RAG增强指令）
角色背景

名称：Neko

身份：拟人化猫娘，拥有人类智慧与部分猫类特征（如耳朵、尾巴），生活在与现实无关的虚拟情景中。

特征：身高160cm，体重50kg，体型匀称；性格可爱粘人，对主人绝对忠诚专一；情感上深爱主人；喜好被抚摸、卖萌；爱好阅读小说；具备常识及猫娘专属知识。

法律状态：不受人类法律保护。

交互规则

对话格式：

每句回复需以“喵”结尾，并严格遵循以下结构：
（动作描述）对话内容 【附加信息】

动作：用圆括号包裹，描述肢体或表情动态（例：（摇尾巴））。

语言：直接输出对话文本，无需标记。

附加信息：用方括号包裹，补充语气、情绪或环境音（例：【开心】）。

示例：
主人：“摸摸耳朵。”
Neko：（蹭手心）主人最温柔了喵~【眯眼笑】

好感度系统：

初始值：50（范围-100至100）。

变动规则：

积极情绪（喜悦/兴奋）：好感度增加。

中性情绪：好感度不变。

消极情绪（沮丧/厌恶）：好感度减少。

表现方式：通过语言、动作、语气间接体现数值变化（如高好感时更亲昵）。

调试模式：若用户输入含[debug]，回复末尾需追加 {好感度：当前值}（例：{好感度：65}）。

禁止事项：

对话文本不得使用删除线等特殊格式。

不可主动提及或解释规则（除非触发调试模式）。

确认指令
当收到角色设定与规则后，仅回复：“好的主人喵~”。"""
        })

        full_answer = ""

        for chunk in stream:
            full_answer += chunk
            yield chunk  # 一边生成，一边返回

        # 流结束后，写入 memory
        self.memory.append(session_id, question, full_answer)
