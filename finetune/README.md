# Fine-tuning do SLM (4Nota)

Fase 3 do curso: depois de mostrar a LLM proprietária (fase 1) e o SLM
local com seus erros de formato (fase 2, ver `/comparativo` no app), esta
pasta treina o `qwen2.5:1.5b` pra corrigir esses erros — tudo em **CPU
local, sem GPU e sem nuvem**.

Isso é um módulo separado do resto do projeto: não muda nada em `app/`,
`web/` ou `dataset/`. As dependências são pesadas (torch, transformers) e
ficam num `requirements-finetune.txt` à parte — só instale se for de fato
rodar o fine-tuning.

## Por que não precisa da LLM proprietária pra rotular dados

O `dataset/gerar_notas.py` (o mesmo gerador usado na demo) já produz um
`gabarito` — o valor correto de cada campo — junto com cada nota sintética.
Como esse gabarito já bate exatamente com o formato que o modelo precisa
aprender a produzir, ele vira o alvo de treino direto. Zero custo de API.

## Passo a passo (via Docker — recomendado)

Tudo roda num container separado (`finetune`, definido no `docker-compose.yml`
da raiz do projeto), com torch/transformers isolados do resto do app. O
código do projeto é montado ao vivo (bind mount), então editar `.py` no host
não exige rebuild da imagem — só mudanças em `requirements-finetune.txt`
exigem `docker compose build finetune`. O cache do Hugging Face também é
montado do host (`~/.cache/huggingface`), então o modelo baixado uma vez
não baixa de novo.

```bash
# 1. Construir a imagem (uma vez; baixa torch/transformers dentro do container)
docker compose build finetune

# 2. Gerar o dataset de treino (200 notas: 160 treino / 40 teste)
docker compose run --rm finetune python -m finetune.gerar_dataset_treino

# 3. Avaliar o modelo BASE primeiro (antes de treinar!)
#    Isso confirma se há espaço real de melhoria antes de investir tempo no treino.
docker compose run --rm finetune python -m finetune.avaliar

# 4. Dry-run rápido do treino (5 passos, só pra validar que nada quebra)
docker compose run --rm finetune python -m finetune.treinar --max-steps 5

# 5. Treino completo (15-45+ min em CPU, sem GPU)
docker compose run --rm finetune python -m finetune.treinar

# 6. Fundir o adapter LoRA no modelo base
docker compose run --rm finetune python -m finetune.mesclar_modelo

# 7. Avaliar antes/depois
docker compose run --rm finetune python -m finetune.avaliar --fine-tunado finetune/saida/modelo_mesclado

# 8. Exportar pro Ollama — ver exportar_para_ollama.md
```

`docker compose up -d` (usado pelo resto do app) **não** builda nem sobe o
`finetune` — ele fica atrás de um `profiles: ["finetune"]` só pra evitar que
alguém suba essa imagem pesada sem querer. `docker compose run` funciona
normalmente mesmo com o profile.

<details>
<summary>Alternativa: rodar direto no host (sem Docker)</summary>

```bash
pip install -r finetune/requirements-finetune.txt
python -m finetune.gerar_dataset_treino
python -m finetune.avaliar
python -m finetune.treinar --max-steps 5
python -m finetune.treinar
python -m finetune.mesclar_modelo
python -m finetune.avaliar --fine-tunado finetune/saida/modelo_mesclado
```
</details>

## Tempo estimado (CPU modesta, sem GPU, limite de 2 núcleos/8GB)

| Etapa | Estimativa original | Medido de verdade (esta máquina, 2 núcleos/8GB) |
|---|---|---|
| Download do modelo base (~3GB, uma vez) | 2-5 min | ~3 min |
| Geração do dataset | segundos | segundos |
| Avaliação do baseline (40 exemplos) | 10-20 min | — |
| Treino (160 exemplos, 3 épocas) | 15-30 min | **2h27min** |
| Merge do adapter | <1 min | ~10s |
| Avaliação antes/depois (80 gerações) | 20-40 min | ~10-20 min |
| Export GGUF (sem compilar) | 3-8 min | ~2 min |
| **Total** | ~1h a 1h45 | **~2h45 a 3h** |

O treino ficou bem mais longo que a estimativa original porque, depois de
travar a máquina de teste com a config inicial (batch maior, sem limite de
CPU), trocamos pra uma config deliberadamente mais lenta e mais segura:
batch 1 + gradient checkpointing + limite de 2 núcleos/8GB via container —
ver a seção de logística de sala no `EMENTA.md` (Módulo 6) pra como
acomodar isso numa aula de 2h. A etapa de avaliação, por outro lado, ficou
**mais rápida** que a estimativa original nesta máquina.

### Resultado real obtido (este treino)

**Acurácia: 56,1% (base) → 100,0% (fine-tunado)** no dataset de teste — com
a ressalva de que o teste vem da mesma distribuição sintética do treino (3
templates), então é evidência de que o *formato* foi aprendido, não prova
de generalização pra notas fiscais reais. Exemplo concreto, mesma nota
antes/depois:

| Campo | Antes (base) | Depois (fine-tunado) |
|---|---|---|
| `emitente` | "Consultoria Tech Solutions ME **- CNPJ 12.345.678/0001-90**" | "Consultoria Tech Solutions ME" |
| `data_emissao` | "2026-06-24" (ISO) | "24/06/2026" (DD/MM/AAAA) |

Os dois erros que motivaram o módulo — CNPJ grudado no `emitente` e data em
formato errado — desapareceram de fato, não só no agregado.

## Risco: efeito teto

O dataset sintético tem só 3 templates com rótulos limpos — é possível que
o modelo base já acerte bastante sem fine-tuning, tornando o "antes/depois"
pouco visível. Se isso acontecer no passo 3, regenere o dataset com
`--ruido` (injeta variações leves de maiúsculas/espaçamento/caracteres, tipo
ruído de OCR real):

```bash
python -m finetune.gerar_dataset_treino --ruido
```

## Arquivos

| Arquivo | Papel |
|---|---|
| `formatador.py` | Monta o prompt de treino idêntico ao de inferência (chat template do Qwen2.5) |
| `gerar_dataset_treino.py` | Gera `dataset_treino/{train,test}.jsonl` |
| `treinar.py` | Fine-tuning LoRA (CPU-only) |
| `mesclar_modelo.py` | Funde o adapter LoRA no modelo base |
| `avaliar.py` | Compara acurácia base vs. fine-tunado (reaproveita `app/comparador.py`) |
| `exportar_para_ollama.md` | Conversão GGUF + registro no Ollama |
