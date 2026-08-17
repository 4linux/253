"""Fine-tuning LoRA do Qwen2.5-1.5B-Instruct em CPU (sem GPU, sem bitsandbytes).

Uso:
    python -m finetune.treinar                 # treino completo
    python -m finetune.treinar --max-steps 5   # dry-run rapido, so pra validar o pipeline
"""

import argparse
import json
import os
import sys
from pathlib import Path

import torch
from datasets import Dataset
from peft import LoraConfig, get_peft_model
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    DataCollatorForSeq2Seq,
    Trainer,
    TrainingArguments,
)

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from finetune.formatador import tokenizar_exemplo

# Limita threads do torch ao que o container do docker-compose realmente tem
# via `deploy.resources.limits.cpus` — sem isso o torch enxerga todos os
# nucleos do HOST (o cgroup so limita tempo de CPU, nao a contagem visivel de
# nucleos) e tenta usar todos eles, disputando com o resto da maquina do
# aluno mesmo com o limite de CPU do container ativo.
_NUM_THREADS = int(os.environ.get("FINETUNE_NUM_THREADS", "2"))
torch.set_num_threads(_NUM_THREADS)

MODELO_BASE = "Qwen/Qwen2.5-1.5B-Instruct"
DATASET_DIR = Path(__file__).resolve().parent / "dataset_treino"
SAIDA_ADAPTER = Path(__file__).resolve().parent / "saida" / "adapter_lora"

TARGET_MODULES = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]


def carregar_jsonl(caminho: Path) -> list[dict]:
    with caminho.open(encoding="utf-8") as f:
        return [json.loads(linha) for linha in f]


def montar_dataset(tokenizer, notas: list[dict]) -> Dataset:
    exemplos = [tokenizar_exemplo(tokenizer, n["texto_ocr"], n["gabarito"]) for n in notas]
    return Dataset.from_list(exemplos)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--max-steps", type=int, default=-1, help="limita o numero de passos (dry-run)")
    parser.add_argument("--epochs", type=float, default=3.0)
    parser.add_argument("--saida", type=Path, default=SAIDA_ADAPTER)
    args = parser.parse_args()

    print(f"Carregando tokenizer e modelo base ({MODELO_BASE})...")
    tokenizer = AutoTokenizer.from_pretrained(MODELO_BASE)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # fp32 (nao bf16): CPUs sem AVX-512/AMX (comum ate em notebooks recentes,
    # ex. Intel hibridos 12a/13a geracao) nao aceleram bf16 em hardware — o
    # treino em bf16 nesse caso cai num caminho sem vetorizacao e fica
    # ordens de magnitude mais lento. fp32 usa AVX2/FMA, suportado
    # universalmente, e e o que realmente roda rapido em CPU generica.
    modelo = AutoModelForCausalLM.from_pretrained(MODELO_BASE, dtype=torch.float32)

    lora_config = LoraConfig(
        r=16,
        lora_alpha=32,
        lora_dropout=0.05,
        target_modules=TARGET_MODULES,
        task_type="CAUSAL_LM",
    )
    modelo = get_peft_model(modelo, lora_config)
    modelo.print_trainable_parameters()
    modelo.enable_input_require_grads()  # necessario p/ gradient checkpointing + peft

    print("Montando dataset de treino...")
    treino = montar_dataset(tokenizer, carregar_jsonl(DATASET_DIR / "train.jsonl"))

    training_args = TrainingArguments(
        output_dir=str(args.saida.parent / "checkpoints"),
        use_cpu=True,
        num_train_epochs=args.epochs,
        max_steps=args.max_steps,
        # batch 1 + accumulation 8 (em vez de batch 4) e gradient checkpointing:
        # bem mais lento, mas o pico de memoria de ativacoes cai muito — prioridade
        # aqui e a maquina do aluno continuar usavel durante o treino, nao velocidade.
        per_device_train_batch_size=1,
        gradient_accumulation_steps=8,
        gradient_checkpointing=True,
        learning_rate=2e-4,
        lr_scheduler_type="cosine",
        warmup_ratio=0.03,
        optim="adamw_torch",
        logging_steps=5,
        save_strategy="no",
        report_to=[],
    )

    trainer = Trainer(
        model=modelo,
        args=training_args,
        train_dataset=treino,
        data_collator=DataCollatorForSeq2Seq(tokenizer, padding=True, label_pad_token_id=-100),
    )

    print("Iniciando treino (CPU) — isso pode levar de 15 a 45+ minutos...")
    trainer.train()

    args.saida.mkdir(parents=True, exist_ok=True)
    modelo.save_pretrained(str(args.saida))
    tokenizer.save_pretrained(str(args.saida))
    print(f"Adapter LoRA salvo em {args.saida}")


if __name__ == "__main__":
    main()
