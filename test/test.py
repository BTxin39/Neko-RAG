from app.assistant import DeepseekRAG

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
            rag.clear_memory()
            print("[system] 记忆已完全清空")
            continue

        try:
            response = rag.answer(user_input)
            print(f"[system] 回答: {response}\n")
        except Exception as e:
            print(f"[wrong] 错误原因: {e}")