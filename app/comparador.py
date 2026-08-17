"""Compara os campos extraidos por dois backends diferentes."""

from app.prompts import CAMPOS

TOLERANCIA_NUMERICA = 0.01  # tolerancia para valores monetarios (arredondamento)


def _normalizar(valor):
    if isinstance(valor, str):
        return valor.strip().lower()
    return valor


def _campos_batem(a, b) -> bool:
    a, b = _normalizar(a), _normalizar(b)
    if isinstance(a, (int, float)) and isinstance(b, (int, float)):
        return abs(a - b) <= TOLERANCIA_NUMERICA
    return a == b


def comparar(campos_proprietario: dict, campos_slm: dict) -> dict:
    """Retorna o resultado da comparacao campo a campo entre os dois outputs."""
    resultado = {}
    for campo in CAMPOS:
        valor_prop = campos_proprietario.get(campo)
        valor_slm = campos_slm.get(campo)
        resultado[campo] = {
            "proprietario": valor_prop,
            "slm": valor_slm,
            "bateu": _campos_batem(valor_prop, valor_slm),
        }

    total = len(CAMPOS)
    acertos = sum(1 for r in resultado.values() if r["bateu"])
    return {
        "campos": resultado,
        "acertos": acertos,
        "total": total,
        "taxa_acerto": round(acertos / total, 3) if total else 0.0,
    }
