"""Funde o adapter LoRA treinado no modelo base, gerando um modelo unico
(sem dependencia do peft em tempo de inferencia) — pronto pra conversao GGUF.

Uso: python -m finetune.mesclar_modelo
"""

import argparse
import json
from pathlib import Path

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

from finetune.treinar import MODELO_BASE, SAIDA_ADAPTER

SAIDA_MESCLADA = Path(__file__).resolve().parent / "saida" / "modelo_mesclado"


def _corrigir_tokenizer_config_para_gguf(saida: Path) -> None:
    """`transformers` recentes salvam `extra_special_tokens` como lista; o
    conversor `convert_hf_to_gguf.py` do llama.cpp usa uma versao mais antiga
    do `transformers`, que espera um dict e quebra com
    `AttributeError: 'list' object has no attribute 'keys'`. Os tokens ja
    estao no vocabulario (tokenizer.json) de qualquer forma — o campo e so
    metadado redundante, seguro de remover."""
    caminho = saida / "tokenizer_config.json"
    if not caminho.exists():
        return
    config = json.loads(caminho.read_text(encoding="utf-8"))
    if isinstance(config.get("extra_special_tokens"), list):
        del config["extra_special_tokens"]
        caminho.write_text(json.dumps(config, ensure_ascii=False, indent=2), encoding="utf-8")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--adapter", type=Path, default=SAIDA_ADAPTER)
    parser.add_argument("--saida", type=Path, default=SAIDA_MESCLADA)
    args = parser.parse_args()

    print(f"Carregando modelo base ({MODELO_BASE})...")
    # fp32 pra bater com o dtype usado no treino (ver treinar.py)
    base = AutoModelForCausalLM.from_pretrained(MODELO_BASE, dtype=torch.float32)
    tokenizer = AutoTokenizer.from_pretrained(MODELO_BASE)

    print(f"Aplicando adapter LoRA de {args.adapter}...")
    modelo = PeftModel.from_pretrained(base, str(args.adapter))

    print("Fundindo pesos (merge_and_unload)...")
    modelo = modelo.merge_and_unload()

    args.saida.mkdir(parents=True, exist_ok=True)
    modelo.save_pretrained(str(args.saida))
    tokenizer.save_pretrained(str(args.saida))
    _corrigir_tokenizer_config_para_gguf(args.saida)
    print(f"Modelo fundido salvo em {args.saida}")


if __name__ == "__main__":
    main()
