"""Selecao do backend 'ativo' para o fluxo de producao — um unico modelo por
vez, escolhido por configuracao (nao por request). Isso simula a decisao real
de uma migracao: antes do migrate, BACKEND_ATIVO=proprietaria; depois de
validar o SLM (via app/comparador.py), BACKEND_ATIVO=slm.

BACKEND_ATIVO=slm (padrao) ou proprietaria
"""

import os

from app import cliente_proprietario, cliente_slm

BACKEND_ATIVO = os.getenv("BACKEND_ATIVO", "slm").strip().lower()


class ErroBackendAtivo(RuntimeError):
    pass


def extrair(texto_ocr: str) -> dict:
    """Extrai os campos usando apenas o backend configurado como ativo."""
    if BACKEND_ATIVO == "slm":
        return cliente_slm.extrair(texto_ocr)
    if BACKEND_ATIVO == "proprietaria":
        return cliente_proprietario.extrair(texto_ocr)
    raise ErroBackendAtivo(
        f"BACKEND_ATIVO invalido: {BACKEND_ATIVO!r} (use 'slm' ou 'proprietaria')"
    )
