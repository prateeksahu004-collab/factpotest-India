from __future__ import annotations

import os
import tomllib
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data"
KB_DIR = DATA_DIR / "kb"
CORPUS_DIR = DATA_DIR / "corpus"
ARTIFACTS_DIR = ROOT / "artifacts"
EVAL_DIR = ROOT / "evals"

SECRET_KEYS = ("GROQ_API_KEY", "SUPABASE_URL", "SUPABASE_KEY", "AUTHORITY_PWD")


@dataclass(frozen=True)
class Models:
    analyze: str = "openai/gpt-oss-20b"
    verify: str = "openai/gpt-oss-120b"
    agent: str = "openai/gpt-oss-120b"
    generator: str = "openai/gpt-oss-120b"
    gold_generator: str = "qwen/qwen3.8-27b"
    prompt_guard: str = "meta-llama/llama-prompt-guard-2-86m"
    safeguard: str = "openai/gpt-oss-safeguard-20b"
    embedding: str = "BAAI/bge-small-en-v1.5"


MODELS = Models()


@lru_cache
def secrets() -> dict[str, str]:
    """Environment variables win, then .streamlit/secrets.toml, then st.secrets (Streamlit Cloud)."""
    found: dict[str, str] = {}
    path = ROOT / ".streamlit" / "secrets.toml"
    if path.exists():
        found.update({k: str(v) for k, v in tomllib.loads(path.read_text()).items() if k in SECRET_KEYS})
    else:
        try:
            import streamlit as st

            found.update({k: str(st.secrets[k]) for k in SECRET_KEYS if k in st.secrets})
        except Exception:
            pass
    found.update({k: os.environ[k] for k in SECRET_KEYS if os.environ.get(k)})
    return found


def secret(name: str) -> str:
    value = secrets().get(name)
    if not value:
        raise KeyError(f"Missing secret {name}: set it in .streamlit/secrets.toml or the environment")
    return value
