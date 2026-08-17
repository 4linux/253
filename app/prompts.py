"""Prompt e schema compartilhados pelos dois backends (proprietario e SLM).

Manter o mesmo prompt/schema nos dois clientes e o que torna a comparacao
campo a campo justa.
"""

CAMPOS = ["numero", "emitente", "cnpj", "data_emissao", "valor_bruto", "iss_retido", "valor_liquido"]

JSON_SCHEMA = {
    "type": "object",
    "properties": {
        "numero": {"type": "string", "description": "Numero da nota fiscal"},
        "emitente": {"type": "string", "description": "Razao social do prestador de servicos"},
        "cnpj": {"type": "string", "description": "CNPJ do emitente, no formato XX.XXX.XXX/XXXX-XX"},
        "data_emissao": {"type": "string", "description": "Data de emissao no formato DD/MM/AAAA"},
        "valor_bruto": {"type": "number", "description": "Valor bruto dos servicos, sem simbolo de moeda"},
        "iss_retido": {"type": "number", "description": "Valor do ISS retido na fonte, sem simbolo de moeda"},
        "valor_liquido": {"type": "number", "description": "Valor liquido a receber, sem simbolo de moeda"},
    },
    "required": CAMPOS,
    "additionalProperties": False,
}

SYSTEM_PROMPT = (
    "Voce e um sistema de extracao de dados de notas fiscais de servico brasileiras. "
    "Receba o texto bruto (originado de OCR, podendo ter formatacao irregular) e "
    "extraia os campos estruturados solicitados. Valores monetarios devem ser numeros "
    "(ponto como separador decimal, sem R$, sem separador de milhar). "
    "Responda APENAS com o JSON, sem nenhum texto adicional antes ou depois."
)


def montar_prompt_usuario(texto_ocr: str) -> str:
    campos_fmt = "\n".join(f"- {c}" for c in CAMPOS)
    return (
        "Extraia os seguintes campos deste texto de nota fiscal:\n"
        f"{campos_fmt}\n\n"
        "Texto da nota fiscal (via OCR):\n"
        "-----\n"
        f"{texto_ocr}\n"
        "-----\n\n"
        "Responda apenas com um objeto JSON contendo exatamente esses campos."
    )
