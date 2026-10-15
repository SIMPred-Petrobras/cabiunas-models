"""O controle: o detector v2 da pesquisa, rodado de forma CAUSAL (replay semanal do protocolo da sombra) e lido como série de alarme.

O controle é uma CAIXA-PRETA: o que entra no harness é só (alarme, vigiado) na grade de 2 min, o mesmo formato de qualquer candidato.
Três variantes existem (atual, G168, B800); a `atual` é a linha de base de toda tabela.
"""
from __future__ import annotations
from pathlib import Path
import pandas as pd

PASTA = Path(__file__).resolve().parents[3] / "data_local" / "controle_v2"


def carrega_controle(variante: str = "atual", pasta: Path = PASTA) -> tuple[pd.Series, pd.Series]:
    """Devolve (alarme, vigiado) booleanos na grade completa de 2 min, do primeiro ao último instante registrado."""
    s = pd.read_csv(pasta / variante / "serie.csv.gz")
    s["ts"] = pd.to_datetime(s["ts"], utc=True)
    s = s.set_index("ts").sort_index()
    grade = pd.date_range(s.index[0], s.index[-1], freq="2min")
    s = s.reindex(grade)
    return s["alarme"].fillna(0).astype(bool), s["vigiado"].fillna(0).astype(bool)
