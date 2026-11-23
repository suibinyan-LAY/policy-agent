"""最小可用的 RAG：读文档 → 切块 → 向量化 → 余弦检索。

真实项目会换成递归切分 + 专业向量库（Chroma、pgvector）+ 重排序，
但骨架和这里完全一样。
"""

import math
from functools import lru_cache

from .config import DATA_DIR
from .llm import embed


def _load_chunks():
    chunks = []
    for path in sorted(DATA_DIR.glob("*.txt")):
        text = path.read_text(encoding="utf-8")
        for para in text.split("\n\n"):
            para = para.strip()
            if para:
                chunks.append({"text": para, "source": path.name})
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
    """返回最相关的几段资料，带相似度和出处文件名。"""
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
