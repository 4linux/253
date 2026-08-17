"""Avalia a acuracia (campo a campo, contra o gabarito) do modelo base e,
opcionalmente, do modelo fine-tunado — reaproveitando app/comparador.py.

Uso:
    python -m finetune.avaliar                              # so o baseline
    python -m finetune.avaliar --fine-tunado finetune/saida/modelo_mesclado  # antes/depois

Aviso: a geracao aqui usa `transformers` puro (nao a versao quantizada que
o Ollama roda) — o numero nao e 1:1 com o que o /comparativo do app mostra,
mas e uma comparacao interna consistente (mesmo metodo pros dois lados).
"""

import argparse
import json
import sys
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.comparador import comparar
from app.prompts import CAMPOS
from app.tema import console
from app.util_json import ErroExtracaoJSON, extrair_json
from finetune.formatador import prompt_inferencia
from finetune.treinar import DATASET_DIR, MODELO_BASE

MAX_NOVOS_TOKENS = 200
TAMANHO_LOTE = 8


def carregar_teste() -> list[dict]:
    with (DATASET_DIR / "test.jsonl").open(encoding="utf-8") as f:
        return [json.loads(linha) for linha in f]


def gerar_em_lotes(modelo, tokenizer, notas: list[dict]) -> list[dict]:
    """Gera a extracao para cada nota, em lotes, e retorna os campos parseados
    (dict vazio em caso de falha de parse — conta como erro em todos os campos)."""
    resultados = []
    for inicio in range(0, len(notas), TAMANHO_LOTE):
        lote = notas[inicio : inicio + TAMANHO_LOTE]
        prompts = [prompt_inferencia(tokenizer, n["texto_ocr"]) for n in lote]

        entradas = tokenizer(prompts, return_tensors="pt", padding=True, add_special_tokens=False)
        with torch.no_grad():
            saida_ids = modelo.generate(
                **entradas,
                max_new_tokens=MAX_NOVOS_TOKENS,
                do_sample=False,
                pad_token_id=tokenizer.pad_token_id,
            )

        for i, ids in enumerate(saida_ids):
            gerado = ids[entradas["input_ids"].shape[1] :]
            texto = tokenizer.decode(gerado, skip_special_tokens=True)
            try:
                resultados.append(extrair_json(texto))
            except ErroExtracaoJSON:
                resultados.append({})

        print(f"  processadas {min(inicio + TAMANHO_LOTE, len(notas))}/{len(notas)}")
    return resultados


def avaliar_modelo(caminho_modelo: str, notas: list[dict]) -> float:
    print(f"Carregando {caminho_modelo}...")
    tokenizer = AutoTokenizer.from_pretrained(caminho_modelo)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"

    # fp32 pra bater com o dtype do treino/merge (ver treinar.py) e evitar o
    # caminho lento de bf16 sem aceleracao de hardware em CPUs sem AVX-512/AMX
    modelo = AutoModelForCausalLM.from_pretrained(caminho_modelo, dtype=torch.float32)
    modelo.eval()

    print(f"Gerando extracoes para {len(notas)} exemplos de teste...")
    extraidos = gerar_em_lotes(modelo, tokenizer, notas)

    acertos_totais = 0
    campos_totais = 0
    for nota, campos in zip(notas, extraidos):
        resultado = comparar(nota["gabarito"], campos)
        acertos_totais += resultado["acertos"]
        campos_totais += resultado["total"]

    return acertos_totais / campos_totais if campos_totais else 0.0


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fine-tunado", type=Path, default=None, help="caminho do modelo fundido (opcional)")
    args = parser.parse_args()

    notas = carregar_teste()
    console.print(f"[titulo]Avaliando sobre {len(notas)} exemplos de teste[/titulo] (campos: {', '.join(CAMPOS)})")

    acuracia_base = avaliar_modelo(MODELO_BASE, notas)
    console.print(f"\n[secundario]Acuracia do modelo BASE ({MODELO_BASE}):[/secundario] [destaque]{acuracia_base:.1%}[/destaque]")

    if args.fine_tunado:
        acuracia_ft = avaliar_modelo(str(args.fine_tunado), notas)
        console.print(f"[secundario]Acuracia do modelo FINE-TUNADO:[/secundario] [destaque]{acuracia_ft:.1%}[/destaque]")
        delta = acuracia_ft - acuracia_base
        sinal = "+" if delta >= 0 else ""
        console.print(f"\n[titulo]Diferenca:[/titulo] {sinal}{delta:.1%}")

    console.print(
        "\n[secundario]Aviso: esta avaliacao usa transformers puro (nao a versao quantizada do "
        "Ollama) — comparacao interna consistente, nao 1:1 com o /comparativo do app.[/secundario]"
    )


if __name__ == "__main__":
    main()
