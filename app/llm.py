"""模型网关：业务代码只跟这里打交道，换模型、换供应商都只改这一层。

对外提供三个能力：
- chat_stream：流式对话，逐段吐出文字
- chat_once：非流式对话，一次性拿到完整消息（Agent 循环要用）
- embed：把文字变成向量
"""

import json

import requests

from .config import API_KEY, BASE_URL, DEFAULT_MODEL, EMBED_MODEL, require_key


def _headers():
    require_key()
    return {
        "Authorization": "Bearer " + API_KEY,
        "Content-Type": "application/json",
    }


def _raise_for_status(resp):
    if resp.status_code != 200:
        raise RuntimeError("模型调用失败 %s：%s" % (resp.status_code, resp.text[:300]))


def chat_stream(messages, model=None, temperature=0.3):
    """流式对话。上游用 SSE 一段段返回，我们拆出正文再往下传。"""
    resp = requests.post(
        BASE_URL + "/chat/completions",
        headers=_headers(),
        json={
            "model": model or DEFAULT_MODEL,
            "messages": messages,
            "stream": True,
            "temperature": temperature,
        },
        stream=True,
        timeout=120,
    )
    _raise_for_status(resp)

    for line in resp.iter_lines(decode_unicode=True):
        if not line or not line.startswith("data:"):
            continue
        payload = line[5:].strip()
        if payload == "[DONE]":
            break
        delta = json.loads(payload)["choices"][0]["delta"]
        piece = delta.get("content") or ""
        if piece:
            yield piece


def chat_once(messages, model=None, tools=None, temperature=0):
    """非流式对话，返回完整的 message 对象（其中可能带 tool_calls）。"""
    body = {
        "model": model or DEFAULT_MODEL,
        "messages": messages,
        "temperature": temperature,
    }
    if tools:
        body["tools"] = tools

    resp = requests.post(
        BASE_URL + "/chat/completions", headers=_headers(), json=body, timeout=120
    )
    _raise_for_status(resp)
    return resp.json()["choices"][0]["message"]


def embed(texts, model=None):
    """把一批文字变成向量，返回 list[list[float]]。"""
    resp = requests.post(
        BASE_URL + "/embeddings",
        headers=_headers(),
        json={"model": model or EMBED_MODEL, "input": texts},
        timeout=120,
    )
    _raise_for_status(resp)
    return [item["embedding"] for item in resp.json()["data"]]
