"""Interface web: uma secao de 'producao' (um unico backend ativo, escolhido
por configuracao) e uma secao de 'comparativo' (avaliacao lado a lado entre
a LLM proprietaria e o SLM local).

Roda com: python web/app.py  (ou: flask --app web.app run)
"""

import json
import os
import sys
from pathlib import Path

from flask import Flask, render_template, request

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from app import cliente_proprietario, cliente_slm, producao  # noqa: E402
from app.comparador import comparar  # noqa: E402
from app.prompts import CAMPOS  # noqa: E402
from dataset.gerar_notas import gerar_notas  # noqa: E402

DATASET_PATH = ROOT / "dataset" / "notas_fiscais.json"

app = Flask(__name__)


def carregar_notas() -> list[dict]:
    if DATASET_PATH.exists():
        return json.loads(DATASET_PATH.read_text(encoding="utf-8"))
    notas = gerar_notas()
    DATASET_PATH.write_text(json.dumps(notas, ensure_ascii=False, indent=2), encoding="utf-8")
    return notas


def _resolver_texto(notas: list[dict], form) -> tuple[str, str]:
    """Resolve o texto OCR a partir do form (nota escolhida ou texto colado)."""
    texto_custom = form.get("texto_custom", "").strip()
    if texto_custom:
        return texto_custom, "Texto personalizado"
    nota_id = form.get("nota_id")
    nota = next(n for n in notas if str(n["id"]) == nota_id)
    return nota["texto_ocr"], f"Nota #{nota['id']} — {nota['gabarito']['numero']}"


def _comparar_dois_backends(texto_ocr: str):
    """Chama os dois backends e compara. Retorna (resultado_prop, resultado_slm, comparacao, erro)."""
    try:
        resultado_prop = cliente_proprietario.extrair(texto_ocr)
        resultado_slm = cliente_slm.extrair(texto_ocr)
    except Exception as e:
        return None, None, None, str(e)
    comparacao = comparar(resultado_prop["campos"], resultado_slm["campos"])
    return resultado_prop, resultado_slm, comparacao, None


# --------------------------------------------------------------------------
# Home
# --------------------------------------------------------------------------


@app.route("/", methods=["GET"])
def index():
    return render_template("index.html", backend_ativo=producao.BACKEND_ATIVO)


# --------------------------------------------------------------------------
# Producao: um unico backend ativo (escolhido por configuracao)
# --------------------------------------------------------------------------


@app.route("/extrair", methods=["GET"])
def extrair_form():
    notas = carregar_notas()
    return render_template("extrair.html", notas=notas, backend_ativo=producao.BACKEND_ATIVO)


@app.route("/extrair", methods=["POST"])
def extrair_view():
    notas = carregar_notas()
    texto_ocr, nota_label = _resolver_texto(notas, request.form)

    resultado = erro = None
    try:
        resultado = producao.extrair(texto_ocr)
    except Exception as e:
        erro = str(e)

    return render_template(
        "extrair_resultado.html",
        nota_label=nota_label,
        texto_ocr=texto_ocr,
        resultado=resultado,
        campos=CAMPOS,
        erro=erro,
        backend_ativo=producao.BACKEND_ATIVO,
    )


# --------------------------------------------------------------------------
# Comparativo: avaliacao lado a lado (proprietaria vs. SLM)
# --------------------------------------------------------------------------


@app.route("/comparativo", methods=["GET"])
def comparativo_form():
    notas = carregar_notas()
    provider = os.getenv("LLM_PROVIDER", "anthropic")
    return render_template("comparativo.html", notas=notas, provider=provider)


@app.route("/comparativo/resultado", methods=["POST"])
def comparativo_resultado():
    notas = carregar_notas()
    texto_ocr, nota_label = _resolver_texto(notas, request.form)
    resultado_prop, resultado_slm, comparacao, erro = _comparar_dois_backends(texto_ocr)

    return render_template(
        "comparativo_resultado.html",
        nota_label=nota_label,
        texto_ocr=texto_ocr,
        resultado_prop=resultado_prop,
        resultado_slm=resultado_slm,
        comparacao=comparacao,
        campos=CAMPOS,
        erro=erro,
    )


@app.route("/comparativo/resumo", methods=["POST"])
def comparativo_resumo():
    notas = carregar_notas()
    resultados = []
    erro = None

    for nota in notas:
        resultado_prop, resultado_slm, comparacao, erro_nota = _comparar_dois_backends(
            nota["texto_ocr"]
        )
        if erro_nota:
            erro = erro_nota
            break
        resultados.append(
            {"nota": nota, "prop": resultado_prop, "slm": resultado_slm, "comparacao": comparacao}
        )

    resumo = None
    if resultados:
        total = len(resultados)
        resumo = {
            "total": total,
            "custo_prop": sum(r["prop"]["custo_brl"] for r in resultados),
            "custo_slm": sum(r["slm"]["custo_brl"] for r in resultados),
            "latencia_prop": sum(r["prop"]["latencia_s"] for r in resultados) / total,
            "latencia_slm": sum(r["slm"]["latencia_s"] for r in resultados) / total,
            "taxa_media": sum(r["comparacao"]["taxa_acerto"] for r in resultados) / total,
        }

    return render_template(
        "comparativo_resumo.html", resultados=resultados, resumo=resumo, erro=erro, campos=CAMPOS
    )


if __name__ == "__main__":
    app.run(debug=True, port=5000)
