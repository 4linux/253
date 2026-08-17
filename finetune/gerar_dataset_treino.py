"""Gera o dataset de treino/teste para o fine-tuning, reaproveitando o mesmo
gerador sintetico usado na demo (dataset/gerar_notas.py), com seed e
quantidade diferentes para nao sobrepor com as 15 notas do dataset de demo.

O gabarito de cada nota ja bate 1:1 com CAMPOS/JSON_SCHEMA de app/prompts.py
(mesma ordem de chaves, mesmos tipos) — por isso vira o alvo de treino
direto, sem precisar gastar com a LLM proprietaria para rotular nada.
"""

import argparse
import json
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from dataset.gerar_notas import gerar_notas

SAIDA_DIR = Path(__file__).resolve().parent / "dataset_treino"

# Confusoes de OCR tipicas, usadas so quando --ruido esta ligado.
_CONFUSOES = {"0": "O", "O": "0", "1": "l", "l": "1", "5": "S", "S": "5"}


def _aplicar_ruido(texto: str, rng: random.Random, intensidade: float = 0.02) -> str:
    """Injeta variacoes leves (case aleatorio + confusoes de caractere) —
    mitigacao para o caso do modelo base ja acertar quase tudo sem ruido."""
    chars = list(texto)
    for i, c in enumerate(chars):
        if rng.random() < intensidade:
            if c in _CONFUSOES:
                chars[i] = _CONFUSOES[c]
            elif c.isalpha():
                chars[i] = c.upper() if c.islower() else c.lower()
    return "".join(chars)


def gerar(quantidade: int, seed: int, treino_frac: float, ruido: bool) -> tuple[list[dict], list[dict]]:
    notas = gerar_notas(quantidade=quantidade, seed=seed)
    rng = random.Random(seed)

    if ruido:
        for nota in notas:
            nota["texto_ocr"] = _aplicar_ruido(nota["texto_ocr"], rng)

    corte = int(quantidade * treino_frac)
    return notas[:corte], notas[corte:]


def salvar_jsonl(notas: list[dict], caminho: Path):
    with caminho.open("w", encoding="utf-8") as f:
        for nota in notas:
            f.write(json.dumps(nota, ensure_ascii=False) + "\n")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--quantidade", type=int, default=200, help="total de notas geradas")
    parser.add_argument("--seed", type=int, default=999, help="seed (diferente da demo, que usa 42)")
    parser.add_argument("--treino-frac", type=float, default=0.8, help="fracao para treino (resto vai pro teste)")
    parser.add_argument(
        "--ruido",
        action="store_true",
        help="injeta ruido leve no texto OCR (desligado por padrao — so ligar se o baseline sair muito bom)",
    )
    args = parser.parse_args()

    treino, teste = gerar(args.quantidade, args.seed, args.treino_frac, args.ruido)

    SAIDA_DIR.mkdir(parents=True, exist_ok=True)
    salvar_jsonl(treino, SAIDA_DIR / "train.jsonl")
    salvar_jsonl(teste, SAIDA_DIR / "test.jsonl")

    print(f"{len(treino)} exemplos de treino em {SAIDA_DIR / 'train.jsonl'}")
    print(f"{len(teste)} exemplos de teste em {SAIDA_DIR / 'test.jsonl'}")
    if args.ruido:
        print("Ruido de OCR ATIVADO nesta geracao.")


if __name__ == "__main__":
    main()
