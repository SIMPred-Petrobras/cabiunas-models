#!/usr/bin/env python3
"""Figuras da predicao para o relatorio em PDF do detector de 4 sinais.

Tres figuras, cada uma respondendo a uma pergunta diferente sobre a predicao:

  fig1  serie completa -- cada sinal dividido pelo PROPRIO limiar, entao 1,0 e a linha
        de disparo para os quatro numa escala so. Sem isso os quatro sinais teriam
        unidades diferentes e um eixo duplo, que e proibido.
  fig2  zoom em dois eventos detectados -- mostra o sinal subindo antes da parada e o
        lead efetivo.
  fig3  antecedencia por evento -- barras horizontais, um evento por linha, incluindo o
        que nao foi detectado (barra vazia). E a figura que nao deixa esconder o erro.

Reaproveita o cache de piso_fisico.py (t, p, spread bruto + med/mad, referencia rolante
da vibracao) para nao refazer o walk-forward mensal.
"""
from __future__ import annotations
import sys
import numpy as np, pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.dates import DateFormatter

# O pacote `cabiunas_pdm` vive agora em ./cabiunas_pdm, restaurado da branch
# do Francisco (ver cabiunas_pdm/__init__.py). O caminho antigo era um
# diretorio temporario que foi apagado; nao ha mais sys.path a inserir.
from cabiunas_pdm import detector as DET
import avalia as A
import piso_fisico as PF
from ablacao import canonico, mascara_pontuacao
from portoes import K_BASE, K_VIB

# paleta categorica validada (slots 1-4, modo claro), ordem fixa
C1, C2, C3, C4 = "#2a78d6", "#eb6834", "#1baf7a", "#eda100"
INK, INK2, MUTED = "#1a1a1a", "#4a4a4a", "#8a8a8a"
GRID = "#dcdcdc"
SINAIS = [("t", "Temperatura (PCA)", C1, DET.THR_FAM * K_BASE),
          ("p", "Pressão (PCA)",     C2, DET.THR_FAM * K_BASE),
          ("sp", "Spread do mancal", C3, DET.THR_SPREAD * K_BASE),
          ("vb", "Vibração",         C4, 3.0 * K_VIB)]

plt.rcParams.update({
    "font.family": "DejaVu Sans", "font.size": 8,
    "axes.edgecolor": MUTED, "axes.linewidth": 0.6, "axes.labelcolor": INK2,
    "xtick.color": INK2, "ytick.color": INK2, "text.color": INK,
    "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.5,
    "figure.facecolor": "white", "axes.facecolor": "white",
})


def prepara():
    df = canonico()
    falhas = pd.read_csv("falhas.csv", parse_dates=["evento"])["evento"].dt.tz_convert("UTC")
    d = PF.pre(df, falhas)
    mask = mascara_pontuacao(df)
    out = PF.sinais(d, df.index, 0.0, 0.0)
    E = PF.ewmas(out, mask)
    al = PF.alerta_rapido(E, mask, K_BASE, K_VIB)
    # ponto de operacao recomendado, o mesmo do relatorio
    import reduz_fp as RF
    al = RF.dur_min(RF.refratario(al, 48), 60)
    return df, falhas, mask, E, al


# o detector so existe a partir daqui: primeiro instante com os 4 sinais validos dentro
# da mascara (janela_valida.py). 2024-01-16 fica FORA do alvo de avaliacao.
T0 = pd.Timestamp("2024-02-01 00:00", tz="UTC")


def razao(E):
    """Cada sinal dividido pelo proprio limiar: 1,0 = disparo, escala unica."""
    return pd.DataFrame({c: E[c] / thr for c, _, _, thr in SINAIS})


def fig1(R, E, falhas, al, df, path):
    """Duas faixas: o estado fisico da maquina em cima, a decisao do detector embaixo.

    A versao anterior plotava os 4 sinais crus reamostrados por max horario -- ilegivel,
    porque todo transiente vira um pico de altura cheia. A variavel que importa nao e o
    sinal, e o VOTO: quantos dos quatro estao sustentados acima do proprio limiar. Ela e
    discreta (0 a 4), tem o limiar de decisao embutido (>=2) e cabe num eixo so.
    """
    votos = sum(DET._sustained(E[c], thr).astype(int) for c, _, _, thr in SINAIS)
    vd = votos.resample("1D").max()
    t5 = df["T5_AVG_A"].resample("1D").median() if "T5_AVG_A" in df else None

    fig, (a1, a2) = plt.subplots(2, 1, figsize=(7.4, 3.4), sharex=True,
                                 gridspec_kw=dict(height_ratios=[1, 1.5], hspace=0.12))
    if t5 is not None:
        a1.fill_between(t5.index, 0, t5.values, color="#cfcfcf", lw=0, zorder=2)
        a1.plot(t5.index, t5.values, color=INK2, lw=0.6, zorder=3)
    a1.axhline(300, color=C2, lw=0.9, ls="--", zorder=4)
    a1.text(t5.index[3], 330, "300 °C — piso da máscara", fontsize=6.2, color=C2, va="bottom")
    a1.set_ylabel("T5 (°C)", fontsize=7.2); a1.set_ylim(0, 780)
    a1.spines[["top", "right"]].set_visible(False)
    a1.grid(axis="x", visible=False)

    a2.fill_between(vd.index, 0, vd.values, step="mid", color=C1, alpha=0.30, lw=0, zorder=2)
    a2.step(vd.index, vd.values, where="mid", color=C1, lw=0.8, zorder=3)
    a2.axhline(2, color=INK, lw=1.1, ls="--", zorder=4)
    a2.text(vd.index[3], 2.12, "confirmado: 2 ou mais sinais", fontsize=6.4, color=INK, va="bottom")
    for x, y in A.episodios(al):
        a2.axvspan(x, y, color="#e34948", alpha=0.20, lw=0, zorder=1)
    for ax in (a1, a2):
        ax.axvspan(vd.index[0], T0, color="#f0f0f0", lw=0, zorder=0)
    a2.text(T0, 4.0, " detector operante →", fontsize=6.3, color=MUTED, va="top")
    for t in falhas:
        cor = MUTED if t < T0 else INK
        for ax in (a1, a2):
            ax.axvline(t, color=cor, lw=0.9, ls=":" if t < T0 else "-", zorder=5)
        a1.plot([t], [800], marker="v", ms=5, color=cor, clip_on=False, zorder=6)
    a2.set_ylim(0, 4.35); a2.set_yticks([0, 1, 2, 3, 4])
    a2.set_ylabel("sinais em concordância", fontsize=7.2)
    a2.xaxis.set_major_formatter(DateFormatter("%b/%y"))
    a2.spines[["top", "right"]].set_visible(False)

    from matplotlib.patches import Patch
    from matplotlib.lines import Line2D
    a2.legend(handles=[Line2D([], [], color=INK, lw=0.9, label="parada real (8 no alvo)"),
                       Patch(facecolor="#e34948", alpha=0.20, label="alarme ativo"),
                       Patch(facecolor="#f0f0f0", label="antes do detector existir")],
              loc="upper left", ncol=2, fontsize=6.6, frameon=False,
              handlelength=1.4, columnspacing=1.2, borderaxespad=0.3)
    fig.tight_layout(pad=0.4); fig.savefig(path, dpi=200); plt.close(fig)


def fig2(R, falhas, al, mask, path):
    """Zoom em dois eventos. O lead e medido na MESMA janela de 48 h da regua --
    a versao anterior media na janela plotada e reportava 51,8 h, incompativel com a
    tabela do relatorio."""
    alvos = [pd.Timestamp("2025-12-09", tz="UTC"), pd.Timestamp("2026-02-26", tz="UTC")]
    ts = [min(falhas, key=lambda x: abs((x - a).total_seconds())) for a in alvos]
    fig, axs = plt.subplots(1, 2, figsize=(7.4, 2.7))
    for ax, t in zip(axs, ts):
        a0, b0 = t - pd.Timedelta(hours=48), t + pd.Timedelta(hours=4)
        w = R.loc[a0:b0].clip(upper=4.4)
        for (c, rot, cor, _) in SINAIS:
            ax.plot(w.index, w[c], color=cor, lw=1.2, label=rot, zorder=3)
        ax.axhline(1.0, color=INK, lw=1.0, ls="--", zorder=4)
        for x, y in A.episodios(al.loc[a0:b0]):
            ax.axvspan(x, y, color="#e34948", alpha=0.18, lw=0, zorder=1)
        ax.axvline(t, color=INK, lw=1.3, zorder=5)
        jan = al.loc[t - pd.Timedelta(hours=48):t]
        on = jan[jan.fillna(False)]
        if len(on):
            lead = (t - on.index[0]).total_seconds() / 3600
            ax.annotate("", xy=(on.index[0], 3.55), xytext=(t, 3.55),
                        arrowprops=dict(arrowstyle="<->", color=INK, lw=0.9))
            ax.text(on.index[0] + (t - on.index[0]) / 2, 3.68, f"{lead:.1f} h de antecedência",
                    ha="center", fontsize=6.6, color=INK)
        ax.set_title(f"parada de {t:%d/%m/%Y}", fontsize=7.6, color=INK, pad=3)
        ax.set_ylim(0, 4.4)
        ax.xaxis.set_major_locator(matplotlib.dates.HourLocator(byhour=[0, 12]))
        ax.xaxis.set_major_formatter(DateFormatter("%d/%m\n%Hh"))
        ax.tick_params(axis="x", labelsize=6.4)
        ax.spines[["top", "right"]].set_visible(False)
    axs[0].set_ylabel("sinal ÷ limiar", fontsize=7.5)
    h, l = axs[0].get_legend_handles_labels()
    fig.legend(h, l, loc="lower center", ncol=4, fontsize=6.8, frameon=False,
               handlelength=1.4, columnspacing=1.6, bbox_to_anchor=(0.5, -0.02))
    fig.tight_layout(pad=0.4, rect=(0, 0.07, 1, 1)); fig.savefig(path, dpi=200); plt.close(fig)


def fig3(falhas, al, mask, path):
    linhas = []
    for t in falhas:
        jan = al.loc[t - pd.Timedelta(hours=48):t]
        on = jan[jan.fillna(False)]
        linhas.append((t, (t - on.index[0]).total_seconds() / 3600 if len(on) else np.nan))
    fig, ax = plt.subplots(figsize=(7.4, 2.6))
    rot = [f"{t:%d/%m/%Y}" for t, _ in linhas]
    y = np.arange(len(linhas))[::-1]
    for (t, lead), yy in zip(linhas, y):
        if t < T0:
            ax.barh(yy, 48, height=0.55, color="#fafafa", edgecolor="#cfcfcf", lw=0.6,
                    ls=":", zorder=2)
            ax.text(24, yy, "fora da janela — detector ainda sem histórico",
                    ha="center", va="center", fontsize=6.6, color="#8a8a8a", zorder=4)
        elif np.isnan(lead):
            ax.barh(yy, 48, height=0.55, color="#f0f0f0", edgecolor="#cfcfcf", lw=0.6, zorder=2)
            ax.text(24, yy, "sem detecção", ha="center", va="center",
                    fontsize=6.8, color="#8a8a8a", zorder=4)
        else:
            ax.barh(yy, lead, height=0.55, color=C1, zorder=3)
            ax.text(lead + 0.8, yy, f"{lead:.1f} h", va="center", fontsize=6.8, color=INK2, zorder=4)
    ax.set_yticks(y); ax.set_yticklabels(rot, fontsize=7)
    ax.set_xlim(0, 52); ax.set_xlabel("horas de antecedência da primeira detecção", fontsize=7.5)
    ax.axvline(48, color=MUTED, lw=0.8, ls=":")
    ax.text(48, len(linhas) - 0.2, "limite da janela (48 h)", fontsize=6.4,
            color=MUTED, ha="right", va="bottom")
    ax.grid(axis="y", visible=False)
    ax.spines[["top", "right", "left"]].set_visible(False)
    fig.tight_layout(pad=0.4); fig.savefig(path, dpi=200); plt.close(fig)


def main():
    df, falhas, mask, E, al = prepara()
    R = razao(E)
    dentro = pd.Series(df.index >= T0, index=df.index)
    alvo = falhas[falhas >= T0]
    x = A.avalia(al[dentro.values], alvo, (mask & dentro)[dentro.values])
    print(f"detector: {x['det']}/{len(alvo)}  {x['episodios']} eps  {x['fp_mes']:.2f} FP/mes  "
          f"{x['h_fp_mes']:.1f} h/mes  lead {x['lead_med']:.1f} h", flush=True)
    fig1(R, E, falhas, al, df, "fig_pdf_serie.png"); print("fig_pdf_serie.png", flush=True)
    fig2(R, falhas, al, mask, "fig_pdf_zoom.png"); print("fig_pdf_zoom.png", flush=True)
    fig3(falhas, al, mask, "fig_pdf_lead.png"); print("fig_pdf_lead.png", flush=True)


main()
