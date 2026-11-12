import os
import requests
from slack_bolt import App
from slack_bolt.adapter.socket_mode import SocketModeHandler
from dotenv import load_dotenv

load_dotenv()

app = App(token=os.environ.get("SLACK_BOT_TOKEN"))

def ask_ollama_local(prompt: str) -> str:
    """학교/로컬 환경의 Ollama 모델(gemma:3.4b) 호출"""
    url = "http://localhost:11434/api/generate"
    payload = {
        "model": "gemma:3.4b",
        "prompt": prompt,
        "stream": False
    }
    try:
        response = requests.post(url, json=payload, timeout=60)
        return response.json().get("response", "모델 응답을 받지 못했습니다.")
    except Exception as e:
        # 차안: 로컬 실패 시 Groq API 무료 키 활용 가능
        return f"로컬 모델 호출 오류 발생: {e}"

@app.event("app_mention")
def handle_mention(event, say):
    user_text = event.get("text", "")
    thread_ts = event.get("thread_ts", event.get("ts")) # 스레드 유지
    
    say(text="📰 최신 뉴스를 바탕으로 퀴즈를 생성하고 있습니다 잠시만 기다려주세요...", thread_ts=thread_ts)
    
    # 프롬프트 구성 (Gemma 3.4b 맞춤형)
    prompt = f"""
    사용자 요청: {user_text}
    위 요청에 맞춰 최신 경제/IT 뉴스를 기반으로 4지선다형 시사 상식 퀴즈 1문항, 정답, 그리고 해설을 만들어주세요.
    """
    
    # LLM(Ollama) 호출
    bot_answer = ask_ollama_local(prompt)
    
    # 슬랙 스레드로 답변 반환
    say(text=bot_answer, thread_ts=thread_ts)

if __name__ == "__main__":
    init_db_checker = os.getenv("DATABASE_URL")
    if init_db_checker:
        from database import init_db
        init_db()
        
    handler = SocketModeHandler(app, os.environ["SLACK_APP_TOKEN"])
    handler.start()