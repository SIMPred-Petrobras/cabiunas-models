#!/usr/bin/env python3
"""EXP 48b — EIXOS POR EVENTO (ANÁLISE PÓS-HOC, DESCRITIVA, SEM PRÉ-REGISTRO): algum modelo é melhor que o PCA em algum eixo?

Feita DEPOIS de ver os EXP 44, 46, 47 e 48, a pedido do usuário ("tem algum eixo em que outro modelo é melhor?"); não tem critério de ganho e nada aqui se adota. Escores: PCA (escores_sub34) e K-Means, análise fatorial e LSTM (artefatos do EXP 44), n = 20.000 e 19.000, alarme do EXP 46 (EWMA 1 h, persistência 2 h, refratário 72 h, limiar por bisseção para o orçamento de 1.457 h sem rótulos, piso no quantil 5%).
Mede, por canal e modelo: detecção por evento (soma dos 2 n), início dentro da janela de 48 h, FP médio e lead médio. Ressalva: o lead inclui alarmes já ligados antes da janela (o PCA tem lead longo em parte por isso).
Uso:  PYTHONPATH=src .venv/bin/python experiments/48b_eixos_por_evento_pos_hoc.py   (precisa de data_local/escores_lit44_w, gerado pelo EXP 48)
"""
import sys, os, importlib.util
import numpy as np, pandas as pd, warnings
warnings.filterwarnings("ignore"); sys.path.insert(0, "experiments"); sys.path.insert(0, "src")
spec = importlib.util.spec_from_file_location("exp46", "experiments/46_optuna_modelos_loeo_placebo.py"); E = importlib.util.module_from_spec(spec); spec.loader.exec_module(E)
from nv.avalia import avalia, carrega_controle
from nv.causal import carrega_registro
from nv.politica import ewma_vigiado, episodios, alarme_de
ROOT = E.ROOT; INICIO8 = E.INICIO8
reg = [t for t in carrega_registro() if t >= pd.Timestamp("2025-02-27", tz="UTC")]; al_c, vig_c = carrega_controle("atual"); vig = vig_c; fim = vig_c.index[-1]; ini = INICIO8 - pd.Timedelta(days=10)
def carrega(c, nome, n):
    f = ROOT / "escores_sub34" / f"{c}_A_{n}.parquet" if nome == "PCA" else ROOT / "escores_lit44_w" / f"{c}_{nome}_{n}.parquet"
    return pd.read_parquet(f)["s"].loc[ini:fim]
def alarme(s, hl=1.0, pers=2.0, orc=1457.0):
    e = ewma_vigiado(s, vig.loc[s.index], f"{hl}h"); ed = e.dropna(); lo, hi = float(ed.quantile(.05)), float(ed.max()); dm = pd.Timedelta(hours=pers)
    h = lambda t: avalia(alarme_de(episodios(e, t, dur_min=dm), e.index), vig, [], inicio_periodo=INICIO8).horas_alarme
    for _ in range(18):
        mid = float(np.sqrt(max(lo, 1e-12) * hi)); lo, hi = (mid, hi) if h(mid) > orc else (lo, mid)
    thr = float(np.sqrt(max(lo, 1e-12) * hi)); return alarme_de(episodios(e, thr, dur_min=dm), e.index)
rows = []
for c in ("t", "p"):
    for nome in ("PCA", "KM", "FA", "LSTM"):
        for n in (20000, 19000):
            al = alarme(carrega(c, nome, n)); r = avalia(al, vig, reg, inicio_periodo=INICIO8)
            for t in reg:
                d = r.por_evento.get(str(t), {}); rows.append(dict(canal=c, modelo=nome, n=n, evento=t.strftime("%d/%m/%Y"), det=int(bool(d.get("det"))), inicio=int(bool(d.get("inicio"))), lead=d.get("lead_h")))
            rows.append(dict(canal=c, modelo=nome, n=n, evento="__FP", det=r.fp, inicio=0, lead=r.lead_medio_h))
R = pd.DataFrame(rows); R.to_csv("data_local/eixos44.csv", index=False)
pd.set_option("display.width", 220)
for c in ("t", "p"):
    d = R[(R.canal == c) & (R.evento != "__FP")]; tab = d.pivot_table(index="evento", columns="modelo", values="det", aggfunc="sum")[["PCA", "KM", "FA", "LSTM"]]
    ordem = [t.strftime("%d/%m/%Y") for t in reg]; tab = tab.loc[ordem]; print(f"\n[{c}] eventos detectados (soma dos 2 n; 2 = nos dois):"); print(tab.to_string())
    ini_ = d.pivot_table(index="evento", columns="modelo", values="inicio", aggfunc="sum")[["PCA", "KM", "FA", "LSTM"]].loc[ordem]; print(f"[{c}] início dentro da janela (nasce nas 48 h antes):"); print(ini_.to_string())
    f = R[(R.canal == c) & (R.evento == "__FP")].groupby("modelo").agg(FP=("det", "mean"), lead_medio_h=("lead", "mean")).loc[["PCA", "KM", "FA", "LSTM"]]; print(f"[{c}] FP médio (2 n) e lead médio (h):"); print(f.round(1).to_string())
