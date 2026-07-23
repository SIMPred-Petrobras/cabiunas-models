"""Compara visualmente, para as 7 janelas de `TC382_03_A_recorded_janelas_2025.xlsx`, o dado
realmente gravado no PI (pontos) contra a série interpolada de produção
(`sensores_filtrados_Interpolados_2025.csv`, dataset ClearML e2765c3eef2349cda5f5cbcb0fcd5a40,
usado por configs/calibracao_v4_eq/record2025_tzm3_interpolado_v4_TC382_03_A.json).

Eixo X em UTC (convenção confirmada empiricamente do arquivo interpolado — ver
memória/justificativa na sessão). O trecho sem nenhum dado real dentro da janela nominal é
sombreado em vermelho.

Uso:
  PYTHONPATH=. python scripts/plot_tc382_03_a_recorded_vs_interpolado.py
"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

REC_XLSX = "../dados/TC382_03_A_recorded_janelas_2025.xlsx"
INTERP_CSV = "/home/thallys/.clearml/cache/storage_manager/datasets/ds_e2765c3eef2349cda5f5cbcb0fcd5a40/sensores_filtrados_Interpolados_2025.csv"
OUT_PNG = "eda_load_residual_out/tc382_03_a_gap_check/recorded_vs_interpolado.png"


def _plot_one(ax, j, row, g, w, nom_ini_utc, nom_fim_utc):
    ax.plot(w.index, w.values, lw=1.2, color="steelblue", alpha=0.85,
           label="interpolado (produção, ClearML)", zorder=2)
    ax.scatter(g["t_utc"], g["value"], s=16, color="crimson", zorder=3,
              label="gravado real (PI, recorded)")

    real_first, real_last = g["t_utc"].min(), g["t_utc"].max()
    lead_gap_min = (real_first - nom_ini_utc).total_seconds() / 60
    trail_gap_min = (nom_fim_utc - real_last).total_seconds() / 60
    if lead_gap_min > 5:
        ax.axvspan(nom_ini_utc, real_first, color="red", alpha=0.08,
                  label=f"sem dado real ({lead_gap_min:.0f}min)")
    if trail_gap_min > 5:
        ax.axvspan(real_last, nom_fim_utc, color="red", alpha=0.08,
                  label=f"sem dado real ({trail_gap_min:.0f}min)")

    cov_pct = row["registros_unicos"] / len(w) * 100
    ax.set_title(f"{j} — {row['inicio']} → {row['fim']} (cobertura real: {cov_pct:.0f}%)", fontsize=11)
    ax.set_ylabel("TC382_03_A")
    ax.legend(fontsize=8, loc="best")
    ax.tick_params(axis="x", labelsize=8, rotation=20)


def main():
    rec = pd.read_excel(REC_XLSX, sheet_name="dados_recorded")
    rec["t_utc"] = pd.to_datetime(rec["timestamp_utc"])
    rec["value"] = pd.to_numeric(rec["value"], errors="coerce")
    res = pd.read_excel(REC_XLSX, sheet_name="resumo_janelas")

    interp = pd.read_csv(INTERP_CSV, usecols=["data_datetime", "TC382_03_A"])
    interp["data_datetime"] = pd.to_datetime(interp["data_datetime"], errors="coerce")
    interp = interp.dropna(subset=["data_datetime"]).set_index("data_datetime").sort_index()["TC382_03_A"]

    janelas = res["janela"].tolist()

    # grid resumo (como antes)
    fig, axes = plt.subplots(4, 2, figsize=(15, 16))
    axes = axes.flatten()
    for ax, j in zip(axes, janelas):
        row = res[res["janela"] == j].iloc[0]
        nom_ini_utc = pd.Timestamp(row["inicio"]) + pd.Timedelta(hours=3)
        nom_fim_utc = pd.Timestamp(row["fim"]) + pd.Timedelta(hours=3)
        g = rec[rec["janela"] == j].sort_values("t_utc")
        w = interp.loc[nom_ini_utc:nom_fim_utc]
        _plot_one(ax, j, row, g, w, nom_ini_utc, nom_fim_utc)
    for ax in axes[len(janelas):]:
        ax.axis("off")
    fig.suptitle("TC382_03_A — dado real gravado (PI) vs. série interpolada de produção (eixo X em UTC)",
                fontsize=13, y=1.00)
    fig.tight_layout()
    fig.savefig(OUT_PNG, dpi=120, bbox_inches="tight")
    plt.close(fig)
    print(f"salvo: {OUT_PNG}")

    # uma figura individual, maior, por janela
    for j in janelas:
        row = res[res["janela"] == j].iloc[0]
        nom_ini_utc = pd.Timestamp(row["inicio"]) + pd.Timedelta(hours=3)
        nom_fim_utc = pd.Timestamp(row["fim"]) + pd.Timedelta(hours=3)
        g = rec[rec["janela"] == j].sort_values("t_utc")
        w = interp.loc[nom_ini_utc:nom_fim_utc]

        fig, ax = plt.subplots(figsize=(13, 5))
        _plot_one(ax, j, row, g, w, nom_ini_utc, nom_fim_utc)
        fig.tight_layout()
        safe = j.replace(" ", "_")
        out = f"eda_load_residual_out/tc382_03_a_gap_check/{safe}.png"
        fig.savefig(out, dpi=130, bbox_inches="tight")
        plt.close(fig)
        print(f"salvo: {out}")


if __name__ == "__main__":
    main()
