"""最小可用的 RAG：读文档 → 递归切块 → 向量化 → 余弦检索。

切分策略（三级，逐级降粒度）：
1. 先按空行切段落；
2. 段落超过 MAX_CHUNK 时按句号、问号、分号切成句子；
3. 单句仍然超长时硬切。
最后把句子按顺序打包成不超过 MAX_CHUNK 的块，避免出现大量碎片。
"""

import math
import re
from functools import lru_cache

from .config import DATA_DIR
from .llm import embed

MAX_CHUNK = 300


def _hard_split(text, size=MAX_CHUNK):
    return [text[i : i + size] for i in range(0, len(text), size)]


def _split_sentences(paragraph):
    """把段落切成句子（保留句末标点），超长句硬切。"""
    pieces = [p.strip() for p in re.split(r"(?<=[。！？；])|\n", paragraph)]
    out = []
    for piece in pieces:
        if not piece:
            continue
        if len(piece) > MAX_CHUNK:
            out.extend(_hard_split(piece))
        else:
            out.append(piece)
    return out


def _pack(sentences, max_len=MAX_CHUNK):
    """把句子按顺序拼成不超过 max_len 的块。"""
    chunks, buf = [], ""
    for sentence in sentences:
        if buf and len(buf) + len(sentence) > max_len:
            chunks.append(buf)
            buf = sentence
        else:
            buf += sentence
    if buf:
        chunks.append(buf)
    return chunks


def split_text(text):
    paragraphs = [p.strip() for p in re.split(r"\n\s*\n", text) if p.strip()]
    sentences = []
    for para in paragraphs:
        sentences.extend(_split_sentences(para))
    return _pack(sentences)


def _load_chunks():
    chunks = []
    for path in sorted(DATA_DIR.glob("*.txt")):
        text = path.read_text(encoding="utf-8")
        for piece in split_text(text):
            chunks.append({"text": piece, "source": path.name})
    return chunks


@lru_cache(maxsize=1)
def _index():
    """第一次检索时才建索引，之后复用（进程内缓存）。"""
    chunks = _load_chunks()
    vectors = embed([c["text"] for c in chunks])
    return chunks, vectors


def cosine(a, b):
    dot = sum(x * y for x, y in zip(a, b))
    na = math.sqrt(sum(x * x for x in a))
    nb = math.sqrt(sum(y * y for y in b))
    return dot / (na * nb) if na and nb else 0.0


def search(query, top_k=2):
    """返回最相关的几段资料，带相似度、原文和出处文件名。"""
    chunks, vectors = _index()
    qv = embed([query])[0]
    scored = sorted(
        ((cosine(qv, vec), chunk) for chunk, vec in zip(chunks, vectors)),
        key=lambda pair: pair[0],
        reverse=True,
    )
    return [
        {"score": round(score, 3), "text": chunk["text"], "source": chunk["source"]}
        for score, chunk in scored[:top_k]
    ]


def build_context(hits):
    """把检索结果拼成一段可以直接塞进提示词的文字。"""
    return "\n\n".join(hit["text"] for hit in hits)
