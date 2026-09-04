#!/usr/bin/env python3
"""Fecha a lista de pendencias, cada uma COM e SEM EWMA.

Bracos:
  base          4 sinais, como esta
  +tendencia    5o sinal: z da variacao em 24h das grandezas fisicas
                (mancal, oleo, vibracao) -- taxa, nao nivel. Quatro dos nove
                eventos sao 'Temp.Mt.Alta Manc.Rad'; aquecimento de mancal tem
                taxa caracteristica, e nivel/oscilacao ja foram testados.
  +ensemble     referencia de 667h E 1333h ao mesmo tempo: o sinal so conta se
                dispara nas duas. Evita escolher entre elas e, em tese, corta o
                que e artefato de uma referencia so.

Cada braco roda com EWMA (1h/30min, como o detector) e sem EWMA nenhuma (a
sustentacao sozinha faz a suavizacao). Sem esse controle nao da pra saber se a
EWMA esta ajudando ou so mascarando.

DUAS metricas sempre juntas -- episodios de FP E horas em alarme. A rodada
anterior mostrou que contagem de episodios sozinha e manipulavel por
suavizacao: meia-vida de 16h dava 'menos FP' fundindo alarmes vizinhos, com as
horas triplicando. Comparacao a HORAS igualadas (~90 h/mes, o custo atual).
"""
from __future__ import annotations
import sys
import numpy as np, pandas as pd

# O pacote `cabiunas_pdm` vive agora em ./cabiunas_pdm, restaurado da branch
# do Francisco (ver cabiunas_pdm/__init__.py). O caminho antigo era um
# diretorio temporario que foi apagado; nao ha mais sys.path a inserir.
from cabiunas_pdm import config as C, detector as DET
import avalia as A
from ablacao import canonico, roda, mascara_pontuacao
from ablacao4 import BRACO
from varre_referencia import roda_param

K_VIB = 5.5
KS = [0.4, 0.55, 0.7, 0.85, 1.0, 1.15, 1.3, 1.5, 1.7, 2.0, 2.2, 2.6, 3.0]
H_ALVO = 90.0     # horas de alarme por mes do detector atual
MANCAL = ["954005_624_TI_0301", "954005_624_TI_0303",
          "954005_624_TI_0305", "954005_624_TI_0307"]
OLEO = ["954005_624_PI_0307", "954005_624_PI_0308", "954005_624_PDI_0301",
        "954005_624_PDI_0302", "954005_624_PDI_0317", "954005_624_PDI_0338"]


def sinal_tendencia(df, stable, falhas):
    """5o sinal: z robusto da variacao em 24 h (taxa), nao do nivel."""
    n24 = int(24 * 60 / 2)
    partes = {}
    for nome, tags, agg in [("manc", MANCAL, "mean"), ("oleo", OLEO, "mean"),
                             ("vib", list(C.VIBRATION_TAGS), "max")]:
        s = df[tags].mean(axis=1) if agg == "mean" else df[tags].max(axis=1)
        d = s.diff(n24).where(stable)
        base = d[stable]
        m = base.median(); mad = (base - m).abs().median() * 1.4826
        partes[nome] = ((d - m) / (mad if mad and mad > 0 else np.nan)).abs()
    return pd.DataFrame(partes).max(axis=1)


def monta(df, falhas, stable):
    """Devolve os dicionarios de sinais brutos de cada braco."""
    out_667 = roda(BRACO, df, falhas)
    base = {"t": out_667["t"], "p": out_667["p"],
            "sp": out_667["sp"], "vb": out_667["vb"]}

    tend = dict(base); tend["tr"] = sinal_tendencia(df, stable, falhas)

    out_1333 = roda_param(df, falhas, 400.0, 40_000)
    ens = {"t": (out_667["t"], out_1333["t"]), "p": (out_667["p"], out_1333["p"]),
           "sp": (out_667["sp"], out_1333["sp"]), "vb": (out_667["vb"], out_1333["vb"])}
    return {"base": base, "+tendencia": tend, "+ensemble": ens}


def avalia_braco(sinais, mask, k, com_ewma, falhas, meses, jan48, ensemble=False):
    idx = mask.index
    HL = {"t": "1h", "p": "1h", "sp": "30min", "vb": "30min", "tr": "1h"}
    THR = {"t": DET.THR_FAM, "p": DET.THR_FAM, "sp": DET.THR_SPREAD,
           "vb": 3.0 * K_VIB / k, "tr": DET.THR_FAM}

    def prep(s, canal):
        s = s.where(mask)
        if com_ewma:
            s = s.ewm(halflife=pd.Timedelta(HL[canal]), times=idx).mean().where(mask)
        return s

    n = 0
    for canal, val in sinais.items():
        lim = THR[canal] * k
        if ensemble:
            a = DET._sustained(prep(val[0], canal), lim)
            b = DET._sustained(prep(val[1], canal), lim)
            n = n + (a & b).astype(int)
        else:
            n = n + DET._sustained(prep(val, canal), lim).astype(int)
    al = (n >= 2) & mask
    eps = A.episodios(al)
    fp = [(a, b) for a, b in eps if not any((a <= t1) and (b >= t0) for t0, t1 in jan48)]
    det = [t.strftime("%Y-%m-%d") for t in falhas
           if al[(al.index >= t - pd.Timedelta(hours=48)) & (al.index < t)].any()]
    h = sum((b - a).total_seconds() / 3600 + 2 / 60 for a, b in fp)
    return len(fp), len(det), h / meses, det


def main():
    df = canonico()
    stable = df["stable"].astype(bool)
    falhas = pd.read_csv("falhas.csv", parse_dates=["evento"])["evento"].dt.tz_convert("UTC")
    mask = mascara_pontuacao(df)
    meses = mask.sum() * 2 / 60 / 730
    jan48 = [(t - pd.Timedelta(hours=48), t) for t in falhas]

    print("montando sinais ...", flush=True)
    bracos = monta(df, falhas, stable)

    linhas = []
    for nome, sinais in bracos.items():
        ens = nome == "+ensemble"
        for com in [True, False]:
            for k in KS:
                fp, nd, hm, det = avalia_braco(sinais, mask, k, com, falhas,
                                                meses, jan48, ensemble=ens)
                linhas.append(dict(braco=nome, ewma="com" if com else "sem", k=k,
                                    fp=fp, det=nd, h_mes=hm,
                                    perdidos=",".join(t.strftime("%Y-%m-%d") for t in falhas
                                                       if t.strftime("%Y-%m-%d") not in det)))
            print(f"  {nome} ewma={'com' if com else 'sem'} ok", flush=True)

    T = pd.DataFrame(linhas)
    T.to_csv("lista_final.csv", index=False)

    print(f"\n=== comparacao a HORAS IGUALADAS (~{H_ALVO:.0f} h/mes, o custo atual) ===")
    print(f"{'braco':>12} {'ewma':>5} {'k':>5} {'FP':>4} {'h/mes':>7} {'det':>5}  perdidos")
    for nome in bracos:
        for ew in ["com", "sem"]:
            s = T[(T.braco == nome) & (T.ewma == ew)].copy()
            s["d"] = (s.h_mes - H_ALVO).abs()
            r = s.sort_values("d").iloc[0]
            print(f"{nome:>12} {ew:>5} {r.k:5.2f} {int(r.fp):4d} {r.h_mes:7.1f} "
                  f"{int(r.det):3d}/9  {r.perdidos}")

    print(f"\n=== algum ponto bate o atual em AMBAS (FP<=84 e horas<=90) com det>=8? ===")
    b = T[(T.det >= 8) & (T.fp <= 84) & (T.h_mes <= 90)]
    print(b[["braco", "ewma", "k", "fp", "det", "h_mes"]]
          .to_string(index=False, float_format=lambda v: f"{v:.1f}") if not b.empty else "  NENHUM")


main()
