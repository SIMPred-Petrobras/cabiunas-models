"""Zoom nos eventos curados: mostra o score de temperatura e o z do
mancal_spread ao redor de cada evento, com o limiar, a sustentacao e o
alerta E (producao) marcados -- pra ver visualmente se "acertamos o
alarme" e por que sim/nao."""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

BLUE = "#2a78d6"
ORANGE = "#eb6834"
GREEN = "#0ca30c"
CRITICAL = "#d03b3b"
TEXT_SECONDARY = "#52514e"
GRID = "#e5e4e0"

print("lendo series (pode demorar, ~40MB comprimido)...")
df = pd.read_csv("series_v6_para_plots.csv.gz", parse_dates=["timestamp"]).set_index("timestamp")
eventos = pd.read_csv("eventos_v6_hit_miss.csv", parse_dates=["evento"])
print("series:", df.shape, " eventos:", len(eventos))


def plot_zoom(evento_ts, titulo, fname, dias_antes=4, dias_depois=1.5):
    evento_ts = pd.Timestamp(evento_ts)
    t0 = evento_ts - pd.Timedelta(days=dias_antes)
    t1 = evento_ts + pd.Timedelta(days=dias_depois)
    janela = df.loc[t0:t1]
    if janela.empty:
        print(f"  [{fname}] janela vazia, pulando")
        return

    fig, axes = plt.subplots(3, 1, figsize=(11, 8), sharex=True,
                              gridspec_kw={"height_ratios": [1.1, 1.1, 0.45]})

    ax = axes[0]
    ax.plot(janela.index, janela["T5_AVG_A_raw"], color=TEXT_SECONDARY, linewidth=0.8, alpha=0.85)
    ax.set_ylabel("T5_AVG_A (°C)\n[contexto bruto]")
    ax.axvline(evento_ts, color=CRITICAL, linestyle="--", linewidth=1.6, zorder=5)

    ax = axes[1]
    ax.plot(janela.index, janela["temp_score"] + 1.0, color=BLUE, linewidth=1.0, label="score temperatura (PCA-Q, EWMA 1h)")
    ax.plot(janela.index, janela["temp_thr"] + 1.0, color=BLUE, linewidth=1.0, linestyle=":", label="limiar (2,0×p99)")
    ax.set_yscale("log")
    alerta_temp = janela["temp_alert"].astype(bool)
    if alerta_temp.any():
        ax.fill_between(janela.index, 1.0, (janela["temp_score"].max() + 1.0) * 1.5,
                         where=alerta_temp, color=BLUE, alpha=0.13, step="mid")
    ax.axvline(evento_ts, color=CRITICAL, linestyle="--", linewidth=1.6, zorder=5)
    ax.set_ylabel("score temperatura\n(escala log)")
    ax.legend(loc="upper left", fontsize=8.5, framealpha=0.9)

    ax2 = axes[1].twinx()
    ax2.plot(janela.index, janela["mancal_z"], color=ORANGE, linewidth=1.0, label="z mancal_spread (EWMA 30min)")
    ax2.axhline(3.0, color=ORANGE, linestyle=":", linewidth=1.0)
    ax2.axhline(-3.0, color=ORANGE, linestyle=":", linewidth=1.0)
    ax2.set_ylim(-20, 20)
    alerta_mancal = janela["mancal_alert"].astype(bool)
    ax2.set_ylabel("z mancal_spread\n(cortado em ±20)", color=ORANGE)
    ax2.tick_params(axis="y", colors=ORANGE)
    ax2.legend(loc="upper right", fontsize=8.5, framealpha=0.9)

    ax = axes[2]
    and_prod = janela["alert_and_producao"].astype(bool)
    ax.fill_between(janela.index, 0, 1, where=alerta_temp.values, color=BLUE, alpha=0.5,
                     step="mid", label="temperatura sustentado")
    ax.fill_between(janela.index, 1, 2, where=alerta_mancal.values, color=ORANGE, alpha=0.5,
                     step="mid", label="mancal_spread sustentado")
    ax.fill_between(janela.index, 2, 3, where=and_prod.values, color=CRITICAL, alpha=0.7,
                     step="mid", label="ALERTA CONFIRMADO (E)")
    ax.axvline(evento_ts, color=CRITICAL, linestyle="--", linewidth=1.6, zorder=5)
    ax.set_ylim(0, 3)
    ax.set_yticks([0.5, 1.5, 2.5])
    ax.set_yticklabels(["temp.", "mancal", "E"], fontsize=8.5)
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%d/%m %Hh"))
    plt.setp(ax.get_xticklabels(), rotation=30, ha="right")

    fig.suptitle(titulo, fontsize=12, color="#222")
    plt.tight_layout()
    fig.savefig(fname, dpi=160, facecolor="white", bbox_inches="tight")
    plt.close(fig)
    print(f"  ok: {fname}")


# --- 2 acertos da politica de producao (E entre 2 sinais) -------------
plot_zoom("2025-04-07 21:18:00",
          "Acerto — evento de 2025-04-07 (mancal), detectado pelos 2 sinais com 12,0h de antecedência",
          "fig_zoom_hit_2025-04-07.png")
plot_zoom("2025-11-04 06:22:30",
          "Acerto — evento de 2025-11-04 (óleo lub.), detectado pelos 2 sinais com 14,3h de antecedência",
          "fig_zoom_hit_2025-11-04.png")

# --- 2 erros ilustrativos ----------------------------------------------
plot_zoom("2025-04-11 17:03:00",
          "Erro — evento de 2025-04-11: temperatura E mancal_spread dispararam,\nmas em janelas que não se sobrepuseram — a confirmação E não fechou",
          "fig_zoom_miss_2025-04-11.png")
plot_zoom("2025-02-27 08:38:00",
          "Erro — evento de 2025-02-27 (selagem): só a temperatura antecipou (33,7h);\nmancal_spread não é o sinal certo para esse mecanismo de falha",
          "fig_zoom_miss_2025-02-27.png")

print("\nOK - zooms gerados")
