"""Roda o fluxo completo: para cada nota fiscal sintetica, chama os dois
backends (LLM proprietaria e SLM local), compara os resultados campo a
campo e imprime uma comparacao lado a lado no terminal.
"""

import json
import os
import sys
from pathlib import Path

from rich.panel import Panel
from rich.table import Table
from rich.text import Text

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app import cliente_proprietario, cliente_slm
from app.comparador import comparar
from app.prompts import CAMPOS
from app.tema import console
from dataset.gerar_notas import gerar_notas

DATASET_PATH = Path(__file__).resolve().parent.parent / "dataset" / "notas_fiscais.json"


def carregar_notas() -> list[dict]:
    if DATASET_PATH.exists():
        return json.loads(DATASET_PATH.read_text(encoding="utf-8"))
    notas = gerar_notas()
    DATASET_PATH.write_text(json.dumps(notas, ensure_ascii=False, indent=2), encoding="utf-8")
    return notas


def imprimir_comparacao_nota(nota: dict, resultado_prop: dict, resultado_slm: dict, comparacao: dict):
    tabela = Table(
        title=f"Nota Fiscal #{nota['id']} — {nota['gabarito']['numero']}",
        title_style="titulo",
        show_lines=True,
    )
    tabela.add_column("Campo", style="secundario")
    tabela.add_column(f"Proprietaria ({resultado_prop['model']})", style="proprietario")
    tabela.add_column(f"SLM ({resultado_slm['model']})", style="slm")
    tabela.add_column("Bateu?", justify="center")

    for campo in CAMPOS:
        c = comparacao["campos"][campo]
        marcador = Text("✔", style="ok") if c["bateu"] else Text("✘", style="falha")
        tabela.add_row(campo, str(c["proprietario"]), str(c["slm"]), marcador)

    console.print(tabela)

    resumo = Table.grid(padding=(0, 2))
    resumo.add_column(style="secundario")
    resumo.add_column(style="proprietario")
    resumo.add_column(style="slm")
    resumo.add_row("", f"Proprietaria ({resultado_prop['provider']})", f"SLM ({resultado_slm['provider']})")
    resumo.add_row(
        "Latencia",
        f"{resultado_prop['latencia_s']}s",
        f"{resultado_slm['latencia_s']}s",
    )
    resumo.add_row(
        "Custo estimado",
        f"R$ {resultado_prop['custo_brl']:.4f}",
        f"R$ {resultado_slm['custo_brl']:.4f}",
    )
    resumo.add_row(
        "Acurácia campo a campo",
        f"{comparacao['acertos']}/{comparacao['total']} ({comparacao['taxa_acerto']:.0%})",
        "",
    )
    console.print(resumo)
    console.print()


def imprimir_resumo_final(resultados: list[dict]):
    total_notas = len(resultados)
    custo_total_prop = sum(r["prop"]["custo_brl"] for r in resultados)
    custo_total_slm = sum(r["slm"]["custo_brl"] for r in resultados)
    latencia_media_prop = sum(r["prop"]["latencia_s"] for r in resultados) / total_notas
    latencia_media_slm = sum(r["slm"]["latencia_s"] for r in resultados) / total_notas
    taxa_media = sum(r["comparacao"]["taxa_acerto"] for r in resultados) / total_notas

    tabela = Table(title="Resumo Final", title_style="titulo", show_lines=True)
    tabela.add_column("Métrica", style="secundario")
    tabela.add_column("Proprietária", style="proprietario", justify="right")
    tabela.add_column("SLM local", style="slm", justify="right")

    tabela.add_row("Notas processadas", str(total_notas), str(total_notas))
    tabela.add_row("Custo total estimado", f"R$ {custo_total_prop:.4f}", f"R$ {custo_total_slm:.4f}")
    tabela.add_row("Latência média", f"{latencia_media_prop:.3f}s", f"{latencia_media_slm:.3f}s")
    tabela.add_row("Acurácia média (vs. proprietária)", "—", f"{taxa_media:.1%}")

    console.print(tabela)

    economia = custo_total_prop - custo_total_slm
    console.print(
        Panel(
            f"Economia estimada rodando localmente: R$ {economia:.4f} "
            f"({(economia / custo_total_prop * 100) if custo_total_prop else 0:.1f}% do custo da LLM proprietária)",
            style="destaque",
        )
    )


def verificar_credenciais(provider: str):
    if provider == "anthropic" and not (os.getenv("ANTHROPIC_API_KEY") or os.getenv("ANTHROPIC_AUTH_TOKEN")):
        console.print(
            "[falha]ANTHROPIC_API_KEY não está definida no ambiente.[/falha]\n"
            "Defina-a (ex.: export ANTHROPIC_API_KEY=sk-ant-...) ou troque para "
            "LLM_PROVIDER=openai antes de rodar novamente."
        )
        sys.exit(1)
    if provider == "openai" and not os.getenv("OPENAI_API_KEY"):
        console.print(
            "[falha]OPENAI_API_KEY não está definida no ambiente.[/falha]\n"
            "Defina-a (ex.: export OPENAI_API_KEY=sk-...) antes de rodar novamente."
        )
        sys.exit(1)
    if provider == "openrouter" and not os.getenv("OPENROUTER_API_KEY"):
        console.print(
            "[falha]OPENROUTER_API_KEY não está definida no ambiente.[/falha]\n"
            "Defina-a (ex.: export OPENROUTER_API_KEY=sk-or-...) antes de rodar novamente."
        )
        sys.exit(1)


def main():
    provider = os.getenv("LLM_PROVIDER", "anthropic").strip().lower()
    verificar_credenciais(provider)

    notas = carregar_notas()
    console.print(
        Panel(
            f"Migração LLM proprietária ({provider}) → SLM local (Ollama)\n"
            f"Caso de uso: extração de dados de nota fiscal para JSON estruturado\n"
            f"Notas no dataset: {len(notas)}",
            title="4·Nota",
            title_align="left",
            style="titulo",
        )
    )

    resultados = []
    for nota in notas:
        try:
            resultado_prop = cliente_proprietario.extrair(nota["texto_ocr"])
        except Exception as e:
            console.print(f"[falha]Erro na LLM proprietária (nota #{nota['id']}): {e}[/falha]")
            continue

        try:
            resultado_slm = cliente_slm.extrair(nota["texto_ocr"])
        except Exception as e:
            console.print(f"[falha]Erro no SLM local (nota #{nota['id']}): {e}[/falha]")
            continue

        comparacao = comparar(resultado_prop["campos"], resultado_slm["campos"])
        imprimir_comparacao_nota(nota, resultado_prop, resultado_slm, comparacao)
        resultados.append({"prop": resultado_prop, "slm": resultado_slm, "comparacao": comparacao})

    if resultados:
        imprimir_resumo_final(resultados)
    else:
        console.print("[falha]Nenhuma nota foi processada com sucesso.[/falha]")


if __name__ == "__main__":
    main()
