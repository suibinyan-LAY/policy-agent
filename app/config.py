"""集中管理配置：所有密钥和模型名都从环境变量读取，绝不写死在代码里。"""

import os
from pathlib import Path

from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parent.parent
load_dotenv(ROOT / ".env")

API_KEY = os.getenv("DASHSCOPE_API_KEY", "")
BASE_URL = os.getenv("LLM_BASE_URL", "https://dashscope.aliyuncs.com/compatible-mode/v1")
DEFAULT_MODEL = os.getenv("LLM_MODEL", "deepseek-v4.1-flash")
EMBED_MODEL = os.getenv("EMBED_MODEL", "text-embedding-v3")

AVAILABLE_MODELS = [
    m.strip()
    for m in os.getenv(
        "LLM_MODELS", "deepseek-v4.1-flash,qwen-turbo,qwen-plus"
    ).split(",")
    if m.strip()
]

DATA_DIR = ROOT / "data"
WEB_DIR = ROOT / "web"


def require_key():
    """没有配置密钥时，给出人能看懂的错误，而不是一堆堆栈。"""
    if not API_KEY:
        raise RuntimeError(
            "没有找到 DASHSCOPE_API_KEY。请把 .env.example 复制为 .env，并填入你的 Key。"
        )
