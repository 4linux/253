"""Monta o prompt de treino/avaliacao exatamente como o app usa em inferencia
(app/prompts.py + template de chat do proprio tokenizer do Qwen2.5-Instruct).

Centralizado aqui porque treinar.py e avaliar.py precisam gerar o MESMO
formato de prompt — qualquer divergencia faz o fine-tune aprender um formato
de entrada diferente do que ele vai receber na hora de usar de verdade.
"""

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.prompts import SYSTEM_PROMPT, montar_prompt_usuario

MAX_SEQ_LENGTH = 512


def mensagens(texto_ocr: str) -> list[dict]:
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": montar_prompt_usuario(texto_ocr)},
    ]


def prompt_inferencia(tokenizer, texto_ocr: str) -> str:
    """Prompt pronto para geracao (termina com o marcador do turno do assistente)."""
    return tokenizer.apply_chat_template(
        mensagens(texto_ocr), tokenize=False, add_generation_prompt=True
    )


def tokenizar_exemplo(tokenizer, texto_ocr: str, gabarito: dict, max_length: int = MAX_SEQ_LENGTH) -> dict:
    """Tokeniza um exemplo de treino, mascarando o prompt (labels=-100) para
    que a loss seja calculada só sobre o JSON de resposta."""
    prompt = prompt_inferencia(tokenizer, texto_ocr)
    completude = json.dumps(gabarito, ensure_ascii=False) + tokenizer.eos_token

    prompt_ids = tokenizer(prompt, add_special_tokens=False)["input_ids"]
    completude_ids = tokenizer(completude, add_special_tokens=False)["input_ids"]

    input_ids = (prompt_ids + completude_ids)[:max_length]
    labels = ([-100] * len(prompt_ids) + completude_ids)[:max_length]

    if len(input_ids) >= max_length:
        print(
            f"[aviso] exemplo truncado em max_length={max_length} "
            f"(prompt={len(prompt_ids)} + completude={len(completude_ids)} tokens)",
            file=sys.stderr,
        )

    return {
        "input_ids": input_ids,
        "attention_mask": [1] * len(input_ids),
        "labels": labels,
    }
