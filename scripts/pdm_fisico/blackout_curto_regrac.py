#!/usr/bin/env python3
"""Encurtar o blackout torna 04/11/2025 robusto? E a que custo?

DE ONDE VEM A PERGUNTA. `diagnostico_0411.py` mostrou que o evento fragil nao e
fragil por falta de sinal -- e por falta de OBSERVACAO. A maquina rodou 15,3 h das
48 h antes do trip e o blackout de 6 h descartou 6,0 h delas: 39% de toda a
evidencia disponivel. Nos outros sete eventos o detector ve de 41,6 a 48,0 h.

O BLACKOUT LONGO JA FOI MEDIDO (9/12/18/24 h) e custa deteccao. O CURTO nunca foi
medido NESTE ponto de operacao sob a REGRA C -- a medicao anterior usou a regua
crua e outro ponto, e concluiu "encurtar nao compra deteccao, compra custo".
Vale refazer, porque a pergunta agora e outra: nao e quantos eventos o ponto pega,
e sim se 04/11 deixa de depender de sorte.

A METRICA QUE IMPORTA. Nao e `det` no melhor ponto -- e a FRACAO das configuracoes
dentro do orcamento que detectam cada evento. Um evento detectado por 24% das
configuracoes e fragil mesmo que o ponto escolhido o pegue; detectado por 70% e
robusto. E esse numero que separa "8/8 observado" de "8/8 de verdade".

O RISCO da regra, e por que ele tem que ser medido junto: o blackout existe porque
todo religamento produz transiente mecanico real. Encurta-lo devolve evidencia mas
tambem devolve transiente, e 8 dos 12 falsos positivos do ponto atual ja nascem na
borda do blackout.
"""
from __future__ import annotations
import numpy as np, pandas as pd, avalia as AV
from pos_processamento import EW, BASE, sel, T0
from publica_clearml import GRID, SIN, KAPPA, H_CUSUM, REFRAT_H, DUR_MIN
from blackout_curto import cusum
from plota_estilo_francisco import paradas_reais_2h, classifica_regra_c

BLACKOUTS = ["0min", "1h", "2h", "3h", "4h", "6h"]     # 6h = ponto atual
KBS = [1.3, 1.5, 1.7, 2.0, 2.4]
KVS = [1.8, 2.2, 2.8, 3.5]
TETO_FP = 1.0                                          # orcamento de falso positivo

g = pd.read_parquet("grade2min.parquet")
idx = g.index
op = (g["RUNNING_A"] > 0.5).fillna(False)
estavel = op & (g["T5_AVG_A"] > 300)
part = op & ~op.shift(fill_value=False)
fal = pd.read_csv("falhas.csv", parse_dates=["evento"])["evento"].dt.tz_convert("UTC")
alvo = pd.Series(list(fal[fal >= T0]))
paradas = paradas_reais_2h()
JAN = pd.Timedelta(hours=48)
jw = [(t - JAN, t) for t in alvo]
COLS = [t.strftime("%d/%m/%Y") for t in alvo]
FRAGIL = "04/11/2025"


def pos_(voto, mask):
    al = pd.Series(False, index=idx); bloq = None
    for a, b in AV.episodios(voto):
        if bloq is not None and a <= bloq:
            continue
        al.loc[a:b] = True; bloq = b + pd.Timedelta(hours=REFRAT_H)
    fin = pd.Series(False, index=idx)
    for a, b in AV.episodios(al):
        if (b - a).total_seconds() / 60 + 2 >= DUR_MIN:
            fin.loc[a:b] = True
    return fin & sel


linhas = []
for bl in BLACKOUTS:
    n_bl = int(pd.Timedelta(bl) / pd.Timedelta(GRID))
    blk = (part.rolling(n_bl, min_periods=1).max().astype(bool) if n_bl > 0
           else pd.Series(False, index=idx))
    mask = (estavel & ~blk) & sel
    reset = ((~mask) | part).to_numpy()
    for kb in KBS:
        for kv in KVS:
            K_ = {"t": kb, "p": kb, "sp": kb, "vb": kv}
            ON = {}
            for c in SIN:
                thr = BASE[c] * K_[c]
                E = EW[c].where(mask)
                deg = ((E > thr).astype(int).rolling(15, min_periods=15).sum() >= 15)
                cu = pd.Series(cusum(((E / thr).clip(upper=20) - KAPPA).fillna(0.0).to_numpy(),
                                     reset) > H_CUSUM, index=idx)
                ON[c] = (deg | cu) & mask
            ns = sum(ON[c].astype(int) for c in SIN)
            v = pd.Series(ns >= 2, index=idx) & mask & (ON["sp"] | ON["vb"])
            al = pos_(v, mask)
            eps = AV.episodios(al)
            if not eps:
                continue
            m = AV.avalia(al, alvo, mask); meses = m["horas_op"] / 730.0
            cl = classifica_regra_c(eps, paradas)
            n_fp = sum(1 for a, b, c, l in cl if c == "FP")
            h_fp = sum((b - a).total_seconds() / 3600 for a, b, c, l in cl if c == "FP")
            pega = {t.strftime("%d/%m/%Y"): any(a <= t1 and b >= t0 for a, b in eps)
                    for t, (t0, t1) in zip(alvo, jw)}
            linhas.append(dict(bl=bl, kb=kb, kv=kv, det=sum(pega.values()),
                               fp_mes=n_fp / meses, h_mes=h_fp / meses,
                               lead=m["lead_med"], **pega))
T = pd.DataFrame(linhas)
T.to_csv("blackout_curto_regrac.csv", index=False)

print("FRAGILIDADE POR EVENTO x TAMANHO DO BLACKOUT")
print("(fracao das configuracoes dentro do teto de 1 FP/mes que detectam cada evento)")
print("=" * 104)
print(f"{'blackout':>9} {'cfgs':>6} " + "".join(f"{c[:5]:>8}" for c in COLS) + f"{'melhor':>9}")
for bl in BLACKOUTS:
    d = T[(T.bl == bl) & (T.fp_mes <= TETO_FP)]
    if d.empty:
        print(f"{bl:>9}   nenhuma configuracao dentro do teto"); continue
    fr = "".join(f"{100*d[c].mean():7.0f}%" for c in COLS)
    marca = "   <<< ATUAL" if bl == "6h" else ""
    print(f"{bl:>9} {len(d):6d} {fr} {d.det.max():7.0f}/8{marca}")

print("\n" + "=" * 104)
print(f"O EVENTO FRAGIL ({FRAGIL}) EM DETALHE")
print("=" * 104)
print(f"{'blackout':>9} {'robustez':>10} {'melhor ponto que o pega':>28} {'FP/mes':>8} {'h/mes':>8} {'det':>5}")
for bl in BLACKOUTS:
    d = T[(T.bl == bl) & (T.fp_mes <= TETO_FP)]
    if d.empty:
        continue
    rob = 100 * d[FRAGIL].mean()
    com = d[d[FRAGIL]]
    if com.empty:
        print(f"{bl:>9} {rob:9.0f}%   nenhuma configuracao no teto o detecta"); continue
    b = com.sort_values(["det", "h_mes"], ascending=[False, True]).iloc[0]
    print(f"{bl:>9} {rob:9.0f}% {f'kb={b.kb} kv={b.kv}':>28} {b.fp_mes:8.3f} {b.h_mes:8.2f} {b.det:4.0f}/8")

print("\n" + "=" * 104)
print("O CUSTO DE ENCURTAR -- melhor configuracao de cada blackout, criterio 8/8 e menor h/mes")
print("=" * 104)
print(f"{'blackout':>9} {'det':>5} {'kb':>5} {'kv':>5} {'FP/mes':>8} {'h/mes':>8} {'lead':>7}")
for bl in BLACKOUTS:
    d = T[(T.bl == bl) & (T.fp_mes <= TETO_FP)]
    if d.empty:
        continue
    b = d.sort_values(["det", "h_mes"], ascending=[False, True]).iloc[0]
    marca = "   <<< ATUAL" if bl == "6h" else ""
    print(f"{bl:>9} {b.det:4.0f}/8 {b.kb:5.1f} {b.kv:5.1f} {b.fp_mes:8.3f} {b.h_mes:8.2f} "
          f"{b.lead:6.1f}h{marca}")
print("\n-> blackout_curto_regrac.csv")
