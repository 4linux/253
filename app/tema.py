"""Tema de cores para a saida no terminal, baseado na paleta da 4Linux.

Cores extraidas do logo oficial (imgs/logo-4linux.png):
- teal/ciano (destaque): #12D7D3
- preto:                 #000000
- branco:                #FFFFFF
- cinza (texto sec.):    #757575
"""

from rich.theme import Theme
from rich.console import Console

TEAL = "#12D7D3"
CINZA = "#757575"

TEMA_4LINUX = Theme(
    {
        "titulo": f"bold {TEAL}",
        "destaque": TEAL,
        "ok": "bold green",
        "falha": "bold red",
        "secundario": CINZA,
        "proprietario": f"bold {TEAL}",
        "slm": "bold white",
    }
)

console = Console(theme=TEMA_4LINUX)
