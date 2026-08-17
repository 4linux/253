# 4Nota

Demonstração (projeto 4Linux) de migração de uma LLM proprietária (Claude,
OpenAI ou OpenRouter) para uma **SLM (Small Language Model) local**, rodando
via Docker/Ollama, no caso de uso de **extração de dados de nota fiscal para
JSON estruturado**.

Fase atual: aplicação funcional, sem fine-tune (isso vem em uma fase
posterior). Sem GPU — pensado para rodar bem em CPU.

## Arquitetura

```
slm-migration/
├── docker-compose.yml       # sobe o Ollama (porta 11434)
├── requirements.txt
├── dataset/
│   ├── gerar_notas.py       # gera notas fiscais sintéticas (texto tipo OCR)
│   └── notas_fiscais.json   # gerado na primeira execução
├── app/
│   ├── prompts.py           # prompt e schema JSON compartilhados
│   ├── cliente_proprietario.py  # chama Claude, OpenAI ou OpenRouter
│   ├── cliente_slm.py      # chama o Ollama local via API HTTP
│   ├── producao.py          # escolhe o backend ATIVO (um só) — simula produção
│   ├── comparador.py        # compara os dois outputs campo a campo
│   ├── util_json.py          # extração tolerante de JSON das respostas
│   ├── tema.py               # paleta de cores 4Linux para o terminal
│   └── main.py               # CLI: roda a comparação completa e imprime no terminal
└── web/
    ├── app.py                # interface web (Flask) — reaproveita app/
    ├── templates/             # HTML (Jinja2)
    └── static/style.css       # paleta 4Linux, minimalista
```

A aplicação tem duas frentes bem separadas (na CLI e na web):

- **Produção** (`app/producao.py`, rota `/extrair`): usa **um único backend**,
  escolhido por configuração (`BACKEND_ATIVO`) — é o que rodaria de verdade
  atendendo um usuário, sem custo/latência duplicados.
- **Comparativo** (`app/main.py`, rotas `/comparativo/*`): roda os **dois**
  backends lado a lado e compara — ferramenta de avaliação/QA para decidir
  se a migração já está boa o suficiente, não o fluxo real de produção.

## Pré-requisitos

- Docker instalado e rodando
- Python 3.10+
- Chave de API configurada (uma das opções):
  - `ANTHROPIC_API_KEY` (padrão, `LLM_PROVIDER=anthropic`)
  - `OPENAI_API_KEY` (`LLM_PROVIDER=openai`)
  - `OPENROUTER_API_KEY` (`LLM_PROVIDER=openrouter`) — usa a API compatível
    com OpenAI do [OpenRouter](https://openrouter.ai), útil se você não tem
    chave direta da Anthropic/OpenAI. Modelo padrão: `anthropic/claude-sonnet-5`
    (troque via `OPENROUTER_MODEL`; confira os slugs disponíveis em
    `https://openrouter.ai/api/v1/models`, pois eles mudam com o tempo).

> **Nunca cole chaves de API em mensagens de chat/conversas com IA** — elas
> ficam registradas no histórico. Sempre configure via variável de ambiente
> ou arquivo `.env` (não versionado).

## Subindo o Ollama

```bash
docker compose up -d
docker exec slm-ollama ollama pull qwen2.5:1.5b
```

`qwen2.5:1.5b` foi escolhido por ser leve o suficiente para rodar bem em CPU,
mantendo qualidade razoável para extração estruturada. Para mais qualidade
(com mais custo de CPU/RAM), experimente `qwen2.5:3b` — basta trocar a
variável `OLLAMA_MODEL`.

## Instalando dependências e rodando

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

# Gera o dataset sintético (opcional — gerado automaticamente se faltar)
python -m dataset.gerar_notas

# CLI: roda a comparação completa no terminal
python -m app.main

# Interface web: http://127.0.0.1:5000
python web/app.py
```

## Variáveis de ambiente

| Variável           | Padrão                  | Descrição                                      |
|--------------------|-------------------------|-------------------------------------------------|
| `BACKEND_ATIVO`    | `slm`                  | Backend usado no fluxo de **produção** (`/extrair`): `slm` ou `proprietaria` |
| `LLM_PROVIDER`     | `anthropic`             | Provedor da LLM proprietária: `anthropic`, `openai` ou `openrouter` |
| `ANTHROPIC_MODEL`  | `claude-opus-4-8`       | Modelo Claude usado como baseline proprietário  |
| `OPENAI_MODEL`     | `gpt-4o-mini`           | Modelo OpenAI usado como baseline proprietário  |
| `OPENROUTER_MODEL` | `anthropic/claude-sonnet-5` | Modelo roteado via OpenRouter                 |
| `OLLAMA_HOST`      | `http://localhost:11434`| Endpoint do Ollama                              |
| `OLLAMA_MODEL`     | `qwen2.5:1.5b`          | Modelo local usado no Ollama                    |
| `USD_BRL`          | `5.40`                  | Cotação usada para converter custo USD → R$     |

> Os preços por token usados na estimativa de custo são aproximados
> (`app/cliente_proprietario.py`) — confira os valores vigentes antes de usar
> isso para qualquer decisão de negócio real.

## O que a aplicação faz

**Comparativo** (CLI `python -m app.main`, ou web em `/comparativo`) — para
cada nota fiscal do dataset:
1. Envia o mesmo prompt para a LLM proprietária e para o SLM local.
2. Compara os campos extraídos (número, emitente, CNPJ, data de emissão,
   valor bruto, ISS retido, valor líquido) campo a campo.
3. Mostra latência e custo estimado (R$) de cada chamada.
4. Ao final, imprime um resumo com custo total, latência média e acurácia
   média do SLM em relação à LLM proprietária.

**Produção** (web em `/extrair`) — extrai os campos de uma nota usando
apenas o backend configurado em `BACKEND_ATIVO`, sem chamar os dois modelos.

## Fine-tuning (fase 3)

Depois de mostrar a LLM proprietária e o SLM local com seus erros de
formato (ex.: `emitente` grudando o CNPJ, valores monetários em formato
brasileiro em vez de número puro), a fase de fine-tuning treina o
`qwen2.5:1.5b` pra corrigir esses erros — em CPU local, sem GPU, via um
container Docker separado. Ver `finetune/README.md`.
