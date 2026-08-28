"""Comparativo FP/mes x eventos detectados entre os 4 sinais individuais,
a politica de producao (E entre 2) e a votacao N-de-4 -- avaliados contra
o ground-truth curado (estilo Frnacisco), com a referencia dele (0,94
FP/mes, 6/8 eventos) marcada como linha pontilhada."""
import matplotlib.pyplot as plt
import numpy as np

BLUE = "#2a78d6"
ORANGE = "#eb6834"
GREEN = "#0ca30c"
CRITICAL = "#d03b3b"
TEXT_SECONDARY = "#52514e"
GRID = "#e5e4e0"

# dados: resultado real da task v6 (id 15f89b2c...), estilo Francisco (eventos curados)
labels = ["temperatura\n(sozinho)", "pressao_oleo\n(sozinho)", "mancal_spread\n(sozinho)",
          "selagem_z\n(sozinho)", "produção_2sinais\n(E, política dele)", "votação\n2-de-4",
          "votação\n3-de-4", "votação\n4-de-4"]
eventos_pct = [54.5, 0.0, 36.4, 45.5, 18.2, 45.5, 0.0, 0.0]
fp_mes = [2.81, 0.23, 1.18, 5.85, 0.90, 1.91, 0.17, 0.00]
cores = [BLUE, BLUE, ORANGE, ORANGE, CRITICAL, "#7a5ea8", "#7a5ea8", "#7a5ea8"]

REF_FP = 0.94
REF_EVENTOS = 75.0  # 6/8

fig, axes = plt.subplots(2, 1, figsize=(11, 8.5), sharex=True)
x = np.arange(len(labels))

ax = axes[0]
bars = ax.bar(x, eventos_pct, color=cores, width=0.62)
ax.axhline(REF_EVENTOS, color=TEXT_SECONDARY, linestyle="--", linewidth=1.3)
ax.text(len(labels) - 0.4, REF_EVENTOS + 2, f"Francisco: 6/8 = {REF_EVENTOS:.0f}%",
        color=TEXT_SECONDARY, fontsize=9.5, ha="right")
ax.set_ylabel("Eventos curados detectados (%)")
ax.set_ylim(0, 90)
ax.grid(axis="y", color=GRID, linewidth=0.8, zorder=0)
ax.set_axisbelow(True)
for b, v in zip(bars, eventos_pct):
    ax.text(b.get_x() + b.get_width() / 2, v + 1.5, f"{v:.1f}%", ha="center", fontsize=9)
ax.set_title("Nosso v6 (reprodução das inovações do Francisco) — avaliado contra 11 eventos curados",
             fontsize=12, color="#222", loc="left")

ax = axes[1]
bars = ax.bar(x, fp_mes, color=cores, width=0.62)
ax.axhline(REF_FP, color=TEXT_SECONDARY, linestyle="--", linewidth=1.3)
ax.text(len(labels) - 0.4, REF_FP + 0.15, f"Francisco: {REF_FP:.2f} FP/mês",
        color=TEXT_SECONDARY, fontsize=9.5, ha="right")
ax.set_ylabel("Falso positivo (episódios / mês)")
ax.set_ylim(0, 6.5)
ax.grid(axis="y", color=GRID, linewidth=0.8, zorder=0)
ax.set_axisbelow(True)
for b, v in zip(bars, fp_mes):
    ax.text(b.get_x() + b.get_width() / 2, v + 0.12, f"{v:.2f}", ha="center", fontsize=9)
ax.set_xticks(x)
ax.set_xticklabels(labels, fontsize=9.3)

plt.tight_layout()
fig.savefig("fig_comparativo_sinais.png", dpi=170, facecolor="white", bbox_inches="tight")
print("ok: fig_comparativo_sinais.png")
