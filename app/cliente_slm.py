"""Cliente para a SLM local rodando no Ollama (via Docker), API HTTP.

OLLAMA_HOST  (padrao: http://localhost:11434)
OLLAMA_MODEL (padrao: qwen2.5:1.5b)
"""

import os
import time

import requests

from app.prompts import SYSTEM_PROMPT, montar_prompt_usuario
from app.util_json import ErroExtracaoJSON, extrair_json

OLLAMA_HOST = os.getenv("OLLAMA_HOST", "http://localhost:11434")
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "qwen2.5:1.5b")


class ErroClienteSLM(RuntimeError):
    pass


def extrair(texto_ocr: str) -> dict:
    """Extrai campos estruturados da nota fiscal usando o modelo local via Ollama."""
    inicio = time.perf_counter()
    try:
        resp = requests.post(
            f"{OLLAMA_HOST}/api/chat",
            json={
                "model": OLLAMA_MODEL,
                "messages": [
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": montar_prompt_usuario(texto_ocr)},
                ],
                "format": "json",
                "stream": False,
                "options": {"temperature": 0},
            },
            timeout=120,
        )
        resp.raise_for_status()
    except requests.RequestException as e:
        raise ErroClienteSLM(
            f"Falha ao chamar o Ollama em {OLLAMA_HOST} (container esta rodando?): {e}"
        ) from e
    latencia = time.perf_counter() - inicio

    dados = resp.json()
    conteudo = dados.get("message", {}).get("content", "")
    try:
        campos = extrair_json(conteudo)
    except ErroExtracaoJSON as e:
        raise ErroClienteSLM(str(e)) from e

    tokens_entrada = dados.get("prompt_eval_count", 0)
    tokens_saida = dados.get("eval_count", 0)

    return {
        "provider": "ollama",
        "model": OLLAMA_MODEL,
        "campos": campos,
        "latencia_s": round(latencia, 3),
        "custo_brl": 0.0,
        "tokens_entrada": tokens_entrada,
        "tokens_saida": tokens_saida,
    }
