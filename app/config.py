"""集中管理配置：所有密钥和模型名都从环境变量读取，绝不写死在代码里。"""

import os
import re
from pathlib import Path

from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parent.parent
load_dotenv(ROOT / ".env")

API_KEY = os.getenv("DASHSCOPE_API_KEY", "")
BASE_URL = os.getenv("LLM_BASE_URL", "https://dashscope.aliyuncs.com/compatible-mode/v1")
DEFAULT_MODEL = os.getenv("LLM_MODEL", "deepseek-v4.1-flash")
EMBED_MODEL = os.getenv("EMBED_MODEL", "text-embedding-v3")

DEFAULT_PRESETS = [
    {"id": "deepseek-v4.1-flash", "label": "deepseek-v4.1-flash · 默认"},
    {"id": "qwen-turbo", "label": "qwen-turbo · 快而省"},
    {"id": "qwen-plus", "label": "qwen-plus · 均衡"},
    {"id": "qwen-max", "label": "qwen-max · 最强"},
    {"id": "qwen-long", "label": "qwen-long · 长文档"},
    {"id": "deepseek-v3", "label": "deepseek-v3 · 通用"},
    {"id": "deepseek-r1", "label": "deepseek-r1 · 推理"},
]

# 环境变量 LLM_MODELS 可覆盖预设（逗号分隔），换平台时不用改代码
_env_models = [m.strip() for m in os.getenv("LLM_MODELS", "").split(",") if m.strip()]
MODEL_PRESETS = (
    [{"id": m, "label": m} for m in _env_models] if _env_models else DEFAULT_PRESETS
)

MODEL_PATTERN = re.compile(r"^[A-Za-z0-9._:/-]{1,64}$")


def clean_model(name):
    """校验模型名：既支持下拉里的预设，也支持用户自己填写的模型 ID。"""
    model = (name or DEFAULT_MODEL).strip()
    if not MODEL_PATTERN.match(model):
        raise ValueError("模型名不合法，请从下拉框选择，或填写正确的模型 ID")
    return model

DATA_DIR = ROOT / "data"
WEB_DIR = ROOT / "web"


def require_key():
    """没有配置密钥时，给出人能看懂的错误，而不是一堆堆栈。"""
    if not API_KEY:
        raise RuntimeError(
            "没有找到 DASHSCOPE_API_KEY。请把 .env.example 复制为 .env，并填入你的 Key。"
        )
