"""Índice de SAÚDE/condição por sensor (data-free, sem retreino).

Diferente do alarme de evento: é um indicador LENTO de degradação para manutenção.
A partir do MAE de reconstrução já calculado, suaviza com half-life longo (condição,
não evento), exclui OFF, e mapeia para um percentil 0-100 relativo à distribuição
saudável do próprio sensor. Saídas: heatmap sensor×mês + tendência por sensor.

Uso: PYTHONPATH=. python scripts/health_index.py
"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from clearml import Task

import scripts.eval_per_sensor_level as E

DS = "/home/thallys/.clearml/cache/storage_manager/datasets/ds_424e5b589e13402d9d95371a317e85c9"
SENS2025 = f"{DS}/sensores_filtrados_Interpolados_2025.csv"
TASK = "58bc393c1d7a4e42815236e8897abc88"
OUT = "eval_predictive_out/health_index_2025"
RUN_THR = 50.0
COND_HALFLIFE_H = 24.0          # half-life longo: condição, não evento
SENSORS = ["T5_AVG_A", "TC382_01_A", "TC382_02_A", "TC382_03_A",
           "TC382_04_A", "TC382_05_A", "TC382_06_A"]


def main():
    task = Task.get_task(task_id=TASK)
    mae_all = E.load_mae_series(task, SENSORS)
    ngp = pd.read_csv(SENS2025, usecols=["data_datetime", "NGP_A"])
    ngp["data_datetime"] = pd.to_datetime(ngp["data_datetime"], utc=True, errors="coerce")
    ngp = ngp.dropna(subset=["data_datetime"]).set_index("data_datetime").sort_index()["NGP_A"]

    monthly = {}      # sensor -> Series (mês -> índice 0-100)
    summary = []
    for s in SENSORS:
        mae = mae_all.get(s)
        if mae is None or mae.empty:
            continue
        dt_s = pd.Series(mae.index).diff().dt.total_seconds().median() or 300.0
        on = ngp.reindex(mae.index, method="nearest") > RUN_THR
        cond = mae.where(on.values).ewm(halflife=max(1, int(round(COND_HALFLIFE_H * 3600 / dt_s)))).mean()
        cond = cond.dropna()
        # índice 0-100 = percentil do valor atual na distribuição (ON) do próprio sensor
        idx = cond.rank(pct=True) * 100.0
        m = idx.resample("MS").median()
        monthly[s] = m
        # tendência: inclinação (pontos/mês) por regressão linear nos meses
        if len(m.dropna()) >= 3:
            x = np.arange(len(m)); y = m.values
            ok = ~np.isnan(y)
            slope = np.polyfit(x[ok], y[ok], 1)[0] if ok.sum() >= 2 else 0.0
        else:
            slope = 0.0
        cur = float(m.dropna().iloc[-1]) if len(m.dropna()) else float("nan")
        tend = "PIORANDO" if slope > 2 else ("melhorando" if slope < -2 else "estável")
        summary.append({"sensor": s, "condicao_atual_pct": round(cur, 0),
                        "tendencia_pts_por_mes": round(slope, 1), "estado": tend})

    # heatmap sensor × mês
    H = pd.DataFrame(monthly).T
    H = H[sorted(H.columns)]
    fig, ax = plt.subplots(figsize=(13, 4.5))
    im = ax.imshow(H.values, aspect="auto", cmap="RdYlGn_r", vmin=0, vmax=100)
    ax.set_yticks(range(len(H.index))); ax.set_yticklabels(H.index)
    ax.set_xticks(range(len(H.columns)))
    ax.set_xticklabels([c.strftime("%b/%y") for c in H.columns], rotation=45, ha="right")
    for i in range(len(H.index)):
        for j in range(len(H.columns)):
            v = H.values[i, j]
            if not np.isnan(v):
                ax.text(j, i, f"{v:.0f}", ha="center", va="center",
                        color="white" if (v > 65 or v < 20) else "black", fontsize=7)
    ax.set_title("Índice de saúde por sensor (percentil do MAE-condição; verde=normal, vermelho=degradado)")
    fig.colorbar(im, ax=ax, label="condição (0-100)")
    plt.tight_layout(); plt.savefig(f"{OUT}_heatmap.png", dpi=150); plt.close(fig)

    df = pd.DataFrame(summary).set_index("sensor")
    print(df.to_string())
    df.to_csv(f"{OUT}_summary.csv")
    print(f"\nsalvo: {OUT}_heatmap.png  e  {OUT}_summary.csv")


if __name__ == "__main__":
    main()
