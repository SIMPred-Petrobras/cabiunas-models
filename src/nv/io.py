"""Contrato de dados do TC-33003A.

A grade é de 2 minutos, em UTC. As fontes são só LIDAS (nunca escritas) e ficam fora do git:
  GRADE_PESQUISA  grade de 2 min de jan/2024 a abr/2026 derivada na pesquisa (artefato, não fonte de verdade)
  CSV_BRUTO       export bruto de 30 s de 2025-01-01 a 2026-04-30 (fonte de verdade para a linhagem)
  REGISTRO        registro de trips de produção (instantes reais)

`verifica_linhagem` compara a grade derivada com o CSV bruto reamostrado por ESTE módulo, de modo independente:
um artefato derivado não é aceito sem conferência.
"""
from __future__ import annotations
import os
from pathlib import Path
import numpy as np
import pandas as pd

RAIZ = Path(os.environ.get("NV_DADOS", "/home/thallys/Documents/projeto-petrobras/Analise-exploratoria-dos-dados/analise_cabiunas/cabv2/cabiunas-models"))
GRADE_PESQUISA = RAIZ / "scripts" / "pdm_fisico" / "grade2min.parquet"
CSV_BRUTO = RAIZ / "estrutura_pra_prod" / "Cabiunas" / "dados" / "2025_2026" / "data_2025-01-01_2026-04-30_raw.csv"
REGISTRO = RAIZ / "estrutura_pra_prod" / "Cabiunas" / "registro_trips.csv"

PASSO = pd.Timedelta("2min")
T5_ESTAVEL = 300.0
TAGS_MASCARA = ("RUNNING_A", "T5_AVG_A", "HSX_6240001A")
PREFIXO = "954005_624_"


def grupos(colunas) -> dict[str, list[str]]:
    """Famílias físicas das tags (o contrato). `outras` guarda o que não tem dono físico claro."""
    c = list(colunas)
    g = {
        "exaustao_tc": sorted(x for x in c if x.startswith("TC382_")),
        "vibracao": sorted(x for x in c if x.startswith("TV_")),
        "processo": sorted(x for x in c if x.startswith(PREFIXO)),
    }
    usados = {t for v in g.values() for t in v} | set(TAGS_MASCARA)
    g["mascara"] = [t for t in TAGS_MASCARA if t in c]
    g["outras"] = [x for x in c if x not in usados]
    return g


def carrega_grade(caminho: Path = GRADE_PESQUISA) -> pd.DataFrame:
    g = pd.read_parquet(caminho)
    g.index = pd.DatetimeIndex(g.index)
    if g.index.tz is None:
        g.index = g.index.tz_localize("UTC")
    return g.sort_index()


def mascaras(g: pd.DataFrame) -> pd.DataFrame:
    """ligada (RUNNING_A > 0,5), estável (ligada e T5 > 300) e partida (subida de ligada). Booleanos na grade."""
    ligada = (g["RUNNING_A"] > 0.5).fillna(False)
    estavel = ligada & (g["T5_AVG_A"] > T5_ESTAVEL)
    partida = ligada & ~ligada.shift(fill_value=False)
    return pd.DataFrame({"ligada": ligada, "estavel": estavel, "partida": partida}, index=g.index)


def carrega_csv_bruto_2min(caminho: Path = CSV_BRUTO) -> pd.DataFrame:
    """O export de 30 s reamostrado para 2 min pela média (a mesma convenção da grade de produção: rótulo à esquerda)."""
    d = pd.read_csv(caminho)
    d["ts"] = pd.to_datetime(d["ts"], utc=True)
    d = d.set_index("ts").sort_index()
    return d.resample(PASSO).mean()


def verifica_linhagem(grade: pd.DataFrame, bruto: pd.DataFrame, deslocamentos=(-1, 0, 1)) -> dict:
    """Compara, por tag, a grade derivada com o bruto reamostrado, no período comum, e detecta desalinhamento de 1 passo.

    Devolve, por deslocamento, a mediana (entre tags) da diferença relativa mediana, e as tags piores no deslocamento 0.
    """
    comuns = [c for c in grade.columns if c in bruto.columns]
    ini = max(grade.index[0], bruto.index[0]); fim = min(grade.index[-1], bruto.index[-1])
    a = grade.loc[ini:fim, comuns]; b = bruto.loc[ini:fim, comuns]
    saida = {}
    for k in deslocamentos:
        bb = b.shift(k)
        rel = []
        for c in comuns:
            x, y = a[c].to_numpy(float), bb[c].to_numpy(float)
            m = np.isfinite(x) & np.isfinite(y)
            if m.sum() < 100:
                rel.append(np.nan); continue
            esc = max(float(np.nanmedian(np.abs(x[m]))), 1e-9)
            rel.append(float(np.median(np.abs(x[m] - y[m])) / esc))
        saida[k] = pd.Series(rel, index=comuns)
    melhor = min(saida, key=lambda k: np.nanmedian(saida[k]))
    return dict(tags_comuns=len(comuns), periodo=(ini, fim), mediana_por_deslocamento={k: float(np.nanmedian(v)) for k, v in saida.items()},
                melhor_deslocamento=melhor, piores_tags=saida[0].sort_values(ascending=False).head(5).to_dict(),
                frac_tags_ate_1e3=float((saida[0] <= 1e-3).mean()))
