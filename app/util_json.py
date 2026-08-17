"""Extracao tolerante de JSON de respostas de LLM (uteis quando o modelo nao
respeita 100% um modo JSON estrito, comum em modelos menores/roteados)."""

import json
import re


class ErroExtracaoJSON(RuntimeError):
    pass


def extrair_json(texto: str) -> dict:
    try:
        return json.loads(texto)
    except json.JSONDecodeError:
        pass
    match = re.search(r"\{.*\}", texto, re.DOTALL)
    if not match:
        raise ErroExtracaoJSON(f"Nao foi possivel extrair JSON da resposta do modelo: {texto!r}")
    return json.loads(match.group(0))
