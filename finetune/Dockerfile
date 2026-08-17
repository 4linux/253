# Imagem so com as dependencias de fine-tuning (torch CPU + transformers/peft).
# O codigo do projeto NAO e copiado pra imagem — vem do bind mount do
# docker-compose (repo inteiro em /workspace), assim editar .py no host nao
# exige rebuild; so mudar requirements-finetune.txt exige.
FROM python:3.11-slim

RUN apt-get update && apt-get install -y --no-install-recommends \
    libgomp1 \
    git \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /workspace

COPY requirements-finetune.txt /tmp/requirements-finetune.txt
RUN pip install --no-cache-dir -r /tmp/requirements-finetune.txt

CMD ["bash"]
