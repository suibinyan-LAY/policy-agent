"""Agent：把"查制度"包装成一个工具，让模型自己决定要不要查、查什么。

这里的 while 循环就是所有 Agent 框架的内核。
"""

import json

from .llm import chat_once
from .rag import search

SYSTEM_PROMPT = (
    "你是一名公司制度助手。需要制度信息时，先调用 search_policy 工具查询，"
    "再根据查到的原文回答；查不到就直说没有相关规定，不要编造。"
)

TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "search_policy",
            "description": "在公司制度文档里检索相关条款，例如 住宿标准、报销时限、审批流程、餐补、账号申请",
            "parameters": {
                "type": "object",
                "properties": {
                    "keyword": {
                        "type": "string",
                        "description": "要检索的主题词，尽量简短，例如 住宿",
                    }
                },
                "required": ["keyword"],
            },
        },
    }
]


def _run_tool(name, args):
    if name == "search_policy":
        hits = search(args.get("keyword", ""), top_k=3)
        return json.dumps(hits, ensure_ascii=False), hits
    return "没有这个工具：" + name, []


def run(question, model=None, max_steps=5):
    """跑一轮 Agent，返回最终答案、执行轨迹和引用来源。"""
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": question},
    ]
    steps, sources = [], []

    for _ in range(max_steps):
        msg = chat_once(messages, model=model, tools=TOOLS)

        # 模型认为不需要再查了，直接给答案
        if not msg.get("tool_calls"):
            return {
                "answer": msg.get("content") or "",
                "steps": steps,
                "sources": sources,
            }

        messages.append(msg)

        for call in msg["tool_calls"]:
            name = call["function"]["name"]
            args = json.loads(call["function"]["arguments"] or "{}")
            result, hits = _run_tool(name, args)

            steps.append({"tool": name, "args": args, "result": result[:200]})
            for hit in hits:
                if hit not in sources:
                    sources.append(hit)

            messages.append(
                {"role": "tool", "tool_call_id": call["id"], "content": result}
            )

    return {
        "answer": "（思考步数超过了上限，先回答到这里）",
        "steps": steps,
        "sources": sources,
    }
