"""Serie temporal COMPLETA das grandezas fisicas, com os trips marcados.

A figura anterior mostra a saida do detector (contagem de sinais). Esta mostra
o que os sensores mediram de fato, em unidade fisica, ao longo dos 28 meses --
que e o que um publico de engenharia espera ver antes de acreditar em qualquer
detector. Mediana horaria para nao virar borrao; so operacao quente-estavel.
"""
import sys
# O pacote `cabiunas_pdm` vive agora em ./cabiunas_pdm, restaurado da branch
# do Francisco (ver cabiunas_pdm/__init__.py). O caminho antigo era um
# diretorio temporario que foi apagado; nao ha mais sys.path a inserir.
import numpy as np, pandas as pd
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt, matplotlib.dates as mdates
from cabiunas_pdm import config as C
from ablacao import canonico, mascara_pontuacao

VERM, VERDE = "#9E2B2B", "#1F7A5E"
df = canonico(); mask = mascara_pontuacao(df); idx = df.index
ftodas = pd.read_csv("falhas.csv", parse_dates=["evento"])["evento"].dt.tz_convert("UTC")
op = df["in_operation"].astype(bool)

MANC = ["954005_624_TI_0301","954005_624_TI_0303","954005_624_TI_0305","954005_624_TI_0307"]
OLEO = ["954005_624_PI_0307"]   # PI_0308 mede outra grandeza (~1,4 bar), nao divide eixo
paineis = [
  ("Temperatura de exaustão  (T5 médio)", df["T5_AVG_A"].where(mask), "#B5651D", "°C"),
  ("Termopares do mancal  (4 sensores)", df[MANC].where(mask, axis=0), "#1F7A5E", "°C"),
  ("Pressão de óleo lubrificante  (PI_0307)", df[OLEO[0]].where(mask), "#2E5E6E", "bar"),
  ("Vibração  (máximo das 10 sondas)", df[C.VIBRATION_TAGS].where(mask, axis=0).max(axis=1), "#7A3E8F", "µm"),
]
fig, axes = plt.subplots(5, 1, figsize=(15, 10.5), sharex=True,
                         gridspec_kw={"height_ratios":[1,1,1,1,.42]})
for ax, (tit, serie, cor, un) in zip(axes, paineis):
    s = serie.resample("1h").median()
    if isinstance(s, pd.DataFrame):
        for i, c in enumerate(s.columns):
            ax.plot(s.index, s[c], lw=.55, color=cor, alpha=.55 if len(s.columns)>2 else .8)
    else:
        ax.plot(s.index, s, lw=.6, color=cor, alpha=.9)
    for ev in ftodas:
        ax.axvline(ev, color=VERM, lw=1.1, ls="--", alpha=.75)
    if "mancal" in tit:
        pico = pd.Timestamp("2025-10-31", tz="UTC")
        ax.annotate("TI_0305 sobe a 120 °C\n(normal 71 °C) — trip de 04/11",
                    xy=(pico, 118), xytext=(pd.Timestamp("2025-06-01", tz="UTC"), 112),
                    fontsize=8, color="#9E2B2B",
                    arrowprops=dict(arrowstyle="->", color="#9E2B2B", lw=1))
    ax.set_title(tit, fontsize=10, loc="left")
    ax.set_ylabel(un, fontsize=8.5); ax.grid(alpha=.18, lw=.5)
ax = axes[4]
ax.fill_between(idx, 0, op.astype(int), step="mid", color="#556", lw=0, alpha=.5)
for ev in ftodas: ax.axvline(ev, color=VERM, lw=1.1, ls="--", alpha=.75)
ax.set_ylim(0,1.15); ax.set_yticks([0,1]); ax.set_yticklabels(["parada","operando"], fontsize=8)
ax.set_title("Estado da máquina", fontsize=10, loc="left"); ax.grid(alpha=.18, lw=.5)
ax.xaxis.set_major_locator(mdates.MonthLocator(interval=2))
ax.xaxis.set_major_formatter(mdates.DateFormatter("%b/%y"))
ax.set_xlim(idx[0], idx[-1])
axes[0].plot([],[], color=VERM, ls="--", lw=1.1, label=f"trip ({len(ftodas)} eventos)")
axes[0].legend(loc="upper right", fontsize=8.5, framealpha=.92)
fig.autofmt_xdate()
fig.suptitle("TC-330.03A — grandezas físicas medidas, jan/2024 a abr/2026  "
             "(mediana horária, apenas operação quente-estável)", fontsize=12.5, y=.997)
fig.tight_layout()
fig.savefig("fig_apresentacao_fisica.png", dpi=140)
print("fig_apresentacao_fisica.png")
