#!/usr/bin/env python3
"""Plota o nosso melhor resultado no mesmo estilo do grafico que o Francisco/Lara
mandaram (linha de T5, bandas cinzas de maquina parada, pontos coloridos por
classificacao, linhas verticais laranja nas falhas reais).

Classificacao usa a REGRA C (Francisco): parada real durante o alerta ou ate 48h
depois do fim conta como "antes de parada (nao contado)", nao como falso positivo.
Ver [[fp-bruto-contra-fp-liquido]] e a conversa que fechou nisso.
"""
from __future__ import annotations
import numpy as np, pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import avalia as AV
from pos_processamento import partes, pos, mask, idx, alvo, op, g
from publica_clearml import SIN, REFRAT_H, DUR_MIN

KB, KV = 1.7, 2.2
INK, INK2, MUTED = "#131a20", "#3d4a55", "#6b7885"
RULE, GROUND = "#dde2e6", "#f4f6f7"
AMP, TP_COR, FP_COR, NEU_COR, LINHA = "#b8792a", "#c0392b", "#131a20", "#d68a1f", "#2b6ca3"
STOP_COR = "#c9cfd4"

def alarme():
    ON = partes(KB, KV)
    ns = sum(ON[c].astype(int) for c in SIN)
    v = pd.Series(ns >= 2, index=idx) & mask & (ON["sp"] | ON["vb"])
    return pos(v, ns, REFRAT_H, DUR_MIN, False)


def paradas_reais_2h():
    g_id = (op != op.shift()).cumsum()
    runs = pd.DataFrame({"tempo": idx, "op": op.values, "grupo": g_id.values})
    resumo = runs.groupby("grupo").agg(ini=("tempo", "first"), fim=("tempo", "last"), op=("op", "first"))
    resumo["dur_h"] = (resumo.fim - resumo.ini).dt.total_seconds() / 3600 + (2 / 60)
    return resumo[(resumo.op == False) & (resumo.dur_h >= 2.0)]


def classifica_regra_c(eps, paradas):
    JAN = pd.Timedelta(hours=48)
    jw = [(t - JAN, t) for t in alvo]
    out = []
    for a, b in eps:
        tp_alvo = [t for t, (t0, t1) in zip(alvo, jw) if a <= t1 and b >= t0]
        if tp_alvo:
            lead_h = (tp_alvo[0] - a).total_seconds() / 3600
            out.append((a, b, "TP", lead_h))
            continue
        cand = paradas[(paradas.ini >= a) & (paradas.ini <= b + JAN)]
        if len(cand):
            out.append((a, b, "NEUTRO", None))
        else:
            out.append((a, b, "FP", None))
    return out


if __name__ == "__main__":
    al = alarme()
    eps = AV.episodios(al)
    paradas = paradas_reais_2h()
    classif = classifica_regra_c(eps, paradas)

    n_tp = sum(1 for *_, c, _ in [(a, b, c, l) for a, b, c, l in classif] if c == "TP")
    n_fp = sum(1 for a, b, c, l in classif if c == "FP")
    n_ne = sum(1 for a, b, c, l in classif if c == "NEUTRO")
    m = AV.avalia(al, alvo, mask)
    meses = m["horas_op"] / 730.0
    h_fp = sum((b - a).total_seconds() / 3600 for a, b, c, l in classif if c == "FP")
    print(f"controle: TP={n_tp} FP={n_fp} NEUTRO={n_ne}  FP/mes={n_fp/meses:.3f}  "
          f"h/mes={h_fp/meses:.2f}  lead_med={m['lead_med']:.1f}h")

    # --- serie de fundo: T5 diario (mediana), igual ao "TI_0305 mediana diaria" deles ---
    # filtra para T0 em diante -- grade2min.parquet tem historico anterior que nao entra
    # na avaliacao (idx e op tambem ja vem filtrados pelas mesmas datas usadas no resto)
    from publica_clearml import T0
    t5 = g.loc[g.index >= T0, "T5_AVG_A"].resample("1D").median()

    # --- bandas de maquina parada, em resolucao diaria ---
    parada_diaria = (~op.loc[op.index >= T0]).resample("1D").mean() > 0.5

    fig, ax = plt.subplots(figsize=(15, 5.0), facecolor="white")
    ax.set_facecolor("white")

    # bandas cinzas
    dias = parada_diaria.index
    em_parada = False
    ini_b = None
    for i, v_ in enumerate(parada_diaria.to_numpy()):
        if v_ and not em_parada:
            ini_b = dias[i]; em_parada = True
        elif not v_ and em_parada:
            ax.axvspan(ini_b, dias[i], color=STOP_COR, alpha=0.55, lw=0,
                      label="Máquina parada" if ini_b == dias[parada_diaria.to_numpy().argmax()] else None)
            em_parada = False
    if em_parada:
        ax.axvspan(ini_b, dias[-1], color=STOP_COR, alpha=0.55, lw=0)

    # linha T5
    ax.plot(t5.index, t5.to_numpy(), color=LINHA, lw=1.0, label="T5_AVG_A (mediana diária)", zorder=3)

    # falhas reais
    for i, t in enumerate(alvo):
        ax.axvline(t, color=AMP, ls="--", lw=1.1, alpha=0.9, zorder=4,
                  label="falha real" if i == 0 else None)

    # pontos por episodio, um por dia dentro do episodio (igual ao estilo deles)
    cores = {"TP": TP_COR, "FP": FP_COR, "NEUTRO": NEU_COR}
    rotulos = {"TP": "antecipou falha", "FP": "falso positivo", "NEUTRO": "antes de parada (não contado)"}
    contagem_dias = {"TP": 0, "FP": 0, "NEUTRO": 0}
    primeiro = {"TP": True, "FP": True, "NEUTRO": True}
    for a, b, c, lead_h in classif:
        dias_ep = pd.date_range(a.floor("D"), b.floor("D"), freq="1D")
        for d in dias_ep:
            if d not in t5.index or pd.isna(t5.loc[d]):
                continue
            contagem_dias[c] += 1
            ax.scatter([d], [t5.loc[d]], color=cores[c], s=42, zorder=6,
                      edgecolor="white", linewidth=0.6,
                      label=None if not primeiro[c] else None)
            primeiro[c] = False

    h_leg = [
        plt.Rectangle((0, 0), 1, 1, color=STOP_COR, alpha=0.55, label="Máquina parada"),
        plt.Line2D([0], [0], color=LINHA, lw=1.2, label="T5_AVG_A (mediana diária)"),
        plt.Line2D([0], [0], color=AMP, ls="--", lw=1.2, label="falha real"),
        plt.Line2D([0], [0], marker="o", color="w", markerfacecolor=TP_COR, markersize=7,
                   label=f"antecipou falha ({contagem_dias['TP']} dias)"),
        plt.Line2D([0], [0], marker="o", color="w", markerfacecolor=FP_COR, markersize=7,
                   label=f"falso positivo ({contagem_dias['FP']} dias)"),
        plt.Line2D([0], [0], marker="o", color="w", markerfacecolor=NEU_COR, markersize=7,
                   label=f"antes de parada, não contado ({contagem_dias['NEUTRO']} dias)"),
    ]
    ax.legend(handles=h_leg, loc="upper left", fontsize=8.3, ncol=3, frameon=True,
             framealpha=0.92, edgecolor=RULE, bbox_to_anchor=(0.0, 1.08))

    ax.xaxis.set_major_locator(mdates.MonthLocator(interval=2))
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%b/%y"))
    ax.tick_params(labelsize=8.5, colors=INK2)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(RULE)
    ax.set_ylabel("T5_AVG_A (°C)", fontsize=9.5, color=INK2)
    ax.set_xlabel("Tempo (data/hora)", fontsize=9.5, color=INK2)
    ax.grid(axis="y", color=RULE, lw=0.7)

    titulo = (f"Nosso resultado — 4 sinais físicos, 16 meses — {n_tp}/8 falhas · "
             f"lead {m['lead_med']:.1f} h · {n_fp/meses:.2f} FP/mês · {h_fp/meses:.1f} h/mês "
             f"em alarme falso  (regra C)")
    fig.suptitle(titulo, fontsize=12.5, fontweight="bold", color=INK, y=1.06, x=0.02, ha="left")

    fig.tight_layout(rect=(0, 0, 1, 0.93))
    fig.savefig("fig_nosso_estilo_francisco.png", dpi=150, facecolor="white", bbox_inches="tight")
    plt.close(fig)
    print("-> fig_nosso_estilo_francisco.png")
