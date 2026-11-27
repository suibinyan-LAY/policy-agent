"""FastAPI 入口：一个网页 + 三个接口。

接口：
- GET  /api/config   返回可选模型
- POST /api/chat     流式对话（可选接入知识库）
- POST /api/agent    工具调用（Agent 模式，返回执行轨迹和引用）
"""

import json
from pathlib import Path

from fastapi import FastAPI
from fastapi.responses import HTMLResponse, StreamingResponse

from . import agent
from .config import DEFAULT_MODEL, MODEL_PRESETS, WEB_DIR, clean_model
from .llm import chat_stream
from .rag import build_context, search

app = FastAPI(title="企业制度问答助手")

SYSTEM_PROMPT = (
    "你是一名耐心的公司制度助手。回答简洁，只依据给定资料；"
    "资料里没有的内容直接说不知道，不要编造。"
)


def sse(obj):
    """把我们的一条消息按 SSE 格式包起来。"""
    return "data: " + json.dumps(obj, ensure_ascii=False) + "\n\n"


@app.get("/", response_class=HTMLResponse)
def index():
    return (WEB_DIR / "index.html").read_text(encoding="utf-8")


@app.get("/api/config")
def get_config():
    return {"models": MODEL_PRESETS, "default": DEFAULT_MODEL}


@app.get("/api/health")
def health():
    return {"ok": True}


@app.post("/api/chat")
def chat(body: dict):
    question = (body.get("q") or "").strip()
    raw_model = body.get("model")
    history = body.get("history") or []
    use_kb = bool(body.get("use_kb"))

    def stream():
        try:
            model = clean_model(raw_model)
            messages = [{"role": "system", "content": SYSTEM_PROMPT}]
            messages.extend(history[-10:])  # 只带最近 10 条，控制成本

            if use_kb:
                yield sse({"type": "status", "text": "正在检索资料…"})
                hits = search(question, top_k=2)
                yield sse(
                    {
                        "type": "sources",
                        "sources": [
                            {
                                "score": h["score"],
                                "source": h["source"],
                                "text": h["text"],
                            }
                            for h in hits
                        ],
                    }
                )
                messages.append(
                    {
                        "role": "system",
                        "content": "以下是检索到的公司资料，回答必须依据它：\n\n"
                        + build_context(hits),
                    }
                )

            messages.append({"role": "user", "content": question})

            for piece in chat_stream(messages, model=model):
                yield sse({"type": "delta", "text": piece})

            yield "data: [DONE]\n\n"
        except Exception as exc:  # 把任何异常转成前端能显示的一句话
            yield sse({"type": "error", "text": str(exc)})

    return StreamingResponse(stream(), media_type="text/event-stream")


@app.post("/api/agent")
def run_agent(body: dict):
    question = (body.get("q") or "").strip()
    try:
        return agent.run(question, model=clean_model(body.get("model")))
    except Exception as exc:
        return {"answer": "", "steps": [], "sources": [], "error": str(exc)}


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="127.0.0.1", port=8000)
