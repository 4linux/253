"""Cliente para a LLM proprietaria (Anthropic, OpenAI ou OpenRouter), configuravel via env.

LLM_PROVIDER=anthropic (padrao) | openai | openrouter
ANTHROPIC_MODEL  (padrao: claude-opus-4-8)
OPENAI_MODEL     (padrao: gpt-4o-mini)
OPENROUTER_MODEL (padrao: anthropic/claude-3.5-sonnet)
USD_BRL (padrao: 5.40) - cotacao aproximada para estimar custo em R$ quando o
                         provedor nao retorna o custo real da chamada
"""

import json
import os
import time

import requests

from app.prompts import JSON_SCHEMA, SYSTEM_PROMPT, montar_prompt_usuario
from app.util_json import ErroExtracaoJSON, extrair_json

USD_BRL = float(os.getenv("USD_BRL", "5.40"))

# Precos em USD por 1M de tokens (input, output). Valores aproximados —
# confira o preco vigente antes de usar isso para decisao de negocio real.
PRECOS_USD_POR_MTOK = {
    "claude-opus-4-8": (5.00, 25.00),
    "claude-sonnet-5": (3.00, 15.00),
    "claude-haiku-4-5": (1.00, 5.00),
    "gpt-4o-mini": (0.15, 0.60),
    "gpt-4o": (2.50, 10.00),
    # slugs do OpenRouter (usados so como fallback — o custo real vem de usage.cost)
    "anthropic/claude-sonnet-5": (3.00, 15.00),
    "anthropic/claude-opus-4.8": (5.00, 25.00),
}


class ErroClienteProprietario(RuntimeError):
    pass


def _custo_brl(modelo: str, tokens_entrada: int, tokens_saida: int) -> float:
    preco_in, preco_out = PRECOS_USD_POR_MTOK.get(modelo, (0.0, 0.0))
    custo_usd = (tokens_entrada / 1_000_000) * preco_in + (tokens_saida / 1_000_000) * preco_out
    return round(custo_usd * USD_BRL, 4)


def _extrair_anthropic(texto_ocr: str) -> dict:
    import anthropic

    if not (os.getenv("ANTHROPIC_API_KEY") or os.getenv("ANTHROPIC_AUTH_TOKEN")):
        raise ErroClienteProprietario(
            "ANTHROPIC_API_KEY (ou ANTHROPIC_AUTH_TOKEN) nao esta definida no ambiente."
        )

    modelo = os.getenv("ANTHROPIC_MODEL", "claude-opus-4-8")
    client = anthropic.Anthropic()

    inicio = time.perf_counter()
    resposta = client.messages.create(
        model=modelo,
        max_tokens=1024,
        system=SYSTEM_PROMPT,
        output_config={"format": {"type": "json_schema", "schema": JSON_SCHEMA}},
        messages=[{"role": "user", "content": montar_prompt_usuario(texto_ocr)}],
    )
    latencia = time.perf_counter() - inicio

    texto = next(b.text for b in resposta.content if b.type == "text")
    campos = json.loads(texto)

    return {
        "provider": "anthropic",
        "model": modelo,
        "campos": campos,
        "latencia_s": round(latencia, 3),
        "custo_brl": _custo_brl(modelo, resposta.usage.input_tokens, resposta.usage.output_tokens),
        "tokens_entrada": resposta.usage.input_tokens,
        "tokens_saida": resposta.usage.output_tokens,
    }


def _extrair_openai(texto_ocr: str) -> dict:
    from openai import OpenAI

    if not os.getenv("OPENAI_API_KEY"):
        raise ErroClienteProprietario("OPENAI_API_KEY nao esta definida no ambiente.")

    modelo = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
    client = OpenAI()

    inicio = time.perf_counter()
    resposta = client.chat.completions.create(
        model=modelo,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": montar_prompt_usuario(texto_ocr)},
        ],
        response_format={"type": "json_object"},
    )
    latencia = time.perf_counter() - inicio

    campos = json.loads(resposta.choices[0].message.content)
    uso = resposta.usage

    return {
        "provider": "openai",
        "model": modelo,
        "campos": campos,
        "latencia_s": round(latencia, 3),
        "custo_brl": _custo_brl(modelo, uso.prompt_tokens, uso.completion_tokens),
        "tokens_entrada": uso.prompt_tokens,
        "tokens_saida": uso.completion_tokens,
    }


def _extrair_openrouter(texto_ocr: str) -> dict:
    api_key = os.getenv("OPENROUTER_API_KEY")
    if not api_key:
        raise ErroClienteProprietario("OPENROUTER_API_KEY nao esta definida no ambiente.")

    modelo = os.getenv("OPENROUTER_MODEL", "anthropic/claude-sonnet-5")
    base_url = os.getenv("OPENROUTER_BASE_URL", "https://openrouter.ai/api/v1")

    inicio = time.perf_counter()
    try:
        resp = requests.post(
            f"{base_url}/chat/completions",
            headers={"Authorization": f"Bearer {api_key}"},
            json={
                "model": modelo,
                "messages": [
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": montar_prompt_usuario(texto_ocr)},
                ],
                "response_format": {"type": "json_object"},
                # extensao do OpenRouter: pede o custo real (USD) da chamada no usage
                "usage": {"include": True},
            },
            timeout=60,
        )
        resp.raise_for_status()
    except requests.RequestException as e:
        raise ErroClienteProprietario(f"Falha ao chamar o OpenRouter: {e}") from e
    latencia = time.perf_counter() - inicio

    dados = resp.json()
    conteudo = dados["choices"][0]["message"]["content"]
    try:
        campos = extrair_json(conteudo)
    except ErroExtracaoJSON as e:
        raise ErroClienteProprietario(str(e)) from e

    uso = dados.get("usage", {})
    custo_usd = uso.get("cost")  # presente quando "usage": {"include": True} e suportado
    if custo_usd is None:
        custo_brl = _custo_brl(modelo, uso.get("prompt_tokens", 0), uso.get("completion_tokens", 0))
    else:
        custo_brl = round(custo_usd * USD_BRL, 4)

    return {
        "provider": "openrouter",
        "model": modelo,
        "campos": campos,
        "latencia_s": round(latencia, 3),
        "custo_brl": custo_brl,
        "tokens_entrada": uso.get("prompt_tokens", 0),
        "tokens_saida": uso.get("completion_tokens", 0),
    }


def extrair(texto_ocr: str) -> dict:
    """Extrai campos estruturados da nota fiscal usando a LLM proprietaria configurada."""
    provider = os.getenv("LLM_PROVIDER", "anthropic").strip().lower()
    if provider == "anthropic":
        return _extrair_anthropic(texto_ocr)
    if provider == "openai":
        return _extrair_openai(texto_ocr)
    if provider == "openrouter":
        return _extrair_openrouter(texto_ocr)
    raise ErroClienteProprietario(
        f"LLM_PROVIDER invalido: {provider!r} (use 'anthropic', 'openai' ou 'openrouter')"
    )
