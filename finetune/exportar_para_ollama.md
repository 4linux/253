# Exportando o modelo fine-tunado para o Ollama

Depois de `treinar.py` + `mesclar_modelo.py`, o modelo fundido está em
`finetune/saida/modelo_mesclado/` (formato Hugging Face). Pra rodar no
Ollama (e no app `4Nota`), precisa converter pra GGUF.

**Não precisa compilar nada** — o `convert_hf_to_gguf.py` do llama.cpp
converte direto pra `q8_0` (quantização 8-bit) num caminho 100% Python.

## Passo a passo

Os passos 1-4 rodam dentro do container `finetune` (já tem `git` e Python);
como o repo inteiro está montado em `/workspace`, o `.gguf` gerado aparece
normalmente no host em `finetune/saida/gguf/`.

```bash
# 1. Clonar o llama.cpp dentro do container (só a lógica de conversão, sem compilar)
docker compose run --rm finetune git clone --depth 1 https://github.com/ggml-org/llama.cpp

# 2. Instalar as dependências do conversor (dentro do container)
docker compose run --rm finetune pip install -r llama.cpp/requirements/requirements-convert_hf_to_gguf.txt

# 3. Converter o modelo fundido pra GGUF (q8_0 — sem compilar nada)
docker compose run --rm finetune python llama.cpp/convert_hf_to_gguf.py finetune/saida/modelo_mesclado \
    --outtype q8_0 \
    --outfile finetune/saida/gguf/qwen2.5-1.5b-notafiscal-q8_0.gguf

# 4. Criar o Modelfile do Ollama (no host, é só um arquivo de texto)
cat > finetune/saida/gguf/Modelfile <<'EOF'
FROM ./qwen2.5-1.5b-notafiscal-q8_0.gguf
PARAMETER temperature 0
EOF

# 5. Registrar o modelo no container do Ollama (slm-ollama)
docker cp finetune/saida/gguf/qwen2.5-1.5b-notafiscal-q8_0.gguf slm-ollama:/tmp/
docker cp finetune/saida/gguf/Modelfile slm-ollama:/tmp/
docker exec slm-ollama ollama create nf-qwen2.5-1.5b -f /tmp/Modelfile
```

## Usando no app

```bash
export OLLAMA_MODEL=nf-qwen2.5-1.5b
python -m app.main            # ou: python web/app.py
```

O `/extrair` e o `/comparativo` passam a usar o modelo fine-tunado sem
nenhuma mudança de código.

## Quantização menor (opcional, avançado)

O `q8_0` gera um arquivo de ~1.6 GB (o `qwen2.5:1.5b` original do Ollama usa
uma quantização 4-bit de ~1 GB). Pra chegar num tamanho parecido, é preciso
compilar o `llama-quantize` do llama.cpp — a imagem `finetune` não vem com
`cmake`/compilador por padrão (só é preciso pra esse passo opcional), então
instale ad-hoc dentro do container:

```bash
docker compose run --rm finetune bash -c "
    apt-get update && apt-get install -y --no-install-recommends cmake build-essential &&
    cmake -B llama.cpp/build llama.cpp &&
    cmake --build llama.cpp/build --target llama-quantize -j &&
    ./llama.cpp/build/bin/llama-quantize \
        finetune/saida/gguf/qwen2.5-1.5b-notafiscal-q8_0.gguf \
        finetune/saida/gguf/qwen2.5-1.5b-notafiscal-q4_k_m.gguf \
        Q4_K_M
"
```

Isso é opcional — o `q8_0` já funciona bem para a demo, só ocupa mais disco.
