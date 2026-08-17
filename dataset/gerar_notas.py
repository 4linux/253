"""Gera notas fiscais de servico sinteticas simulando texto bruto de OCR.

Cada nota tem um "gabarito" (valores reais) e um "texto_ocr" (o texto como
seria extraido por um OCR real: formatacao inconsistente, rotulos variados,
quebras de linha irregulares). O gabarito nao e usado para validar os LLMs
neste momento (isso fica para a fase de fine-tuning); serve como referencia
para quem for revisar o dataset.
"""

import json
import random
from pathlib import Path

EMITENTES = [
    ("4Linux Treinamentos e Consultoria Ltda", "07.194.517/0001-70"),
    ("Consultoria Tech Solutions ME", "12.345.678/0001-90"),
    ("Cloud Nine Servicos de TI Ltda", "23.456.789/0001-01"),
    ("DataMinds Analytics EIRELI", "34.567.890/0001-12"),
    ("Infraestrutura Azul Sistemas Ltda", "45.678.901/0001-23"),
    ("Servicos Digitais Horizonte ME", "56.789.012/0001-34"),
    ("Nexus Integracao de Sistemas Ltda", "67.890.123/0001-45"),
    ("Codebase Treinamentos Corporativos", "78.901.234/0001-56"),
    ("Vertex Consultoria em TI Ltda", "89.012.345/0001-67"),
    ("Pioneira Solucoes Open Source ME", "90.123.456/0001-78"),
]

DESCRICOES = [
    "Prestacao de servicos de treinamento em tecnologia da informacao",
    "Consultoria tecnica especializada em infraestrutura Linux",
    "Servicos de suporte tecnico e manutencao de sistemas",
    "Ministracao de curso corporativo in company",
    "Servicos de consultoria em migracao de sistemas",
]

ALIQUOTA_ISS = 0.05


def _gerar_valor(minimo=1500, maximo=45000):
    return round(random.uniform(minimo, maximo), 2)


def _fmt_moeda(valor: float) -> str:
    inteiro, centavos = f"{valor:.2f}".split(".")
    inteiro_fmt = f"{int(inteiro):,}".replace(",", ".")
    return f"{inteiro_fmt},{centavos}"


def _gerar_data(ano=2026):
    mes = random.randint(1, 6)
    dia = random.randint(1, 28)
    return f"{dia:02d}/{mes:02d}/{ano}"


TEMPLATES_OCR = [
    """NOTA FISCAL DE SERVICOS ELETRONICA - NFS-e
Numero: {numero}          Data de Emissao: {data}
PRESTADOR DE SERVICOS
{emitente}
CNPJ: {cnpj}
DISCRIMINACAO DOS SERVICOS
{descricao}
VALORES
Valor Bruto dos Servicos .......... R$ {valor_bruto}
ISS Retido na Fonte ({aliquota}%) ... R$ {iss}
Valor Liquido ...................... R$ {valor_liquido}
""",
    """=== NFS-e Nº {numero} ===
Emitente: {emitente} - CNPJ {cnpj}
Emissao: {data}
Descricao: {descricao}
--- Valores ---
Bruto: R${valor_bruto}
ISS Ret.: R${iss}
Liquido: R${valor_liquido}
""",
    """PREFEITURA MUNICIPAL - NOTA FISCAL DE SERVICOS N. {numero}
Data emissao: {data}
Prestador: {emitente}
CNPJ: {cnpj}
Servicos prestados: {descricao}
Valor bruto: {valor_bruto}
Iss retido: {iss}
Valor liquido a receber: {valor_liquido}
(documento digitalizado via OCR - qualidade variavel)
""",
]


def gerar_notas(quantidade: int = 15, seed: int = 42) -> list[dict]:
    random.seed(seed)
    notas = []
    for i in range(quantidade):
        numero = str(1000 + i)
        emitente, cnpj = random.choice(EMITENTES)
        data = _gerar_data()
        descricao = random.choice(DESCRICOES)
        valor_bruto = _gerar_valor()
        iss = round(valor_bruto * ALIQUOTA_ISS, 2)
        valor_liquido = round(valor_bruto - iss, 2)

        template = random.choice(TEMPLATES_OCR)
        texto_ocr = template.format(
            numero=numero,
            data=data,
            emitente=emitente,
            cnpj=cnpj,
            descricao=descricao,
            valor_bruto=_fmt_moeda(valor_bruto),
            iss=_fmt_moeda(iss),
            valor_liquido=_fmt_moeda(valor_liquido),
            aliquota=ALIQUOTA_ISS * 100,
        )

        notas.append(
            {
                "id": i + 1,
                "texto_ocr": texto_ocr.strip(),
                "gabarito": {
                    "numero": numero,
                    "emitente": emitente,
                    "cnpj": cnpj,
                    "data_emissao": data,
                    "valor_bruto": valor_bruto,
                    "iss_retido": iss,
                    "valor_liquido": valor_liquido,
                },
            }
        )
    return notas


def main():
    notas = gerar_notas()
    saida = Path(__file__).parent / "notas_fiscais.json"
    saida.write_text(json.dumps(notas, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"{len(notas)} notas fiscais sinteticas geradas em {saida}")


if __name__ == "__main__":
    main()
