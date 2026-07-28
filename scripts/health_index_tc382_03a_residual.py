"""Índice de SAÚDE/condição do TC382_03_A usando o sinal resíduo+CNN1D validado
em 2026-07-27/28 (bate a produção em recall, mas continua sem dar alarme de evento
limpo — teto físico do sensor, elevação crônica de baseline).

Adaptado de `scripts/health_index.py` (commit 77eb7ad, data-free): em vez do alarme
pontual, mede condição LENTA (EWMA de half-life longo) mapeada num percentil 0-100
relativo à própria distribuição saudável do sensor. É o entregável certo pra esse
sensor especificamente — ver memória `op-point-half-life-validado` e o diagnóstico de
2026-07-28 (15/17 incidentes sem pico de erro de reconstrução — não é evento, é nível).

Uso: PYTHONPATH=. python scripts/health_index_tc382_03a_residual.py
"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from clearml import Task

import scripts.eval_per_sensor_level as E

TASK = "11978d260dbf4301838fff35452bf97f"  # resíduo+CNN1D, janela de treino igual à produção
SENSOR = "TC382_03_A"
OUT = "eval_predictive_out/health_index_tc382_03a_residual"
RUN_THR = 0.5  # RUNNING_A (não NGP_A — esta task não rastreia NGP_A, ver ressalva no README gerado)
COND_HALFLIFE_H = 24.0  # half-life longo: condição, não evento


def main():
    task = Task.get_task(task_id=TASK)
    mae_all = E.load_mae_series(task, [SENSOR])
    mae = mae_all[SENSOR]

    arts = task.artifacts
    key = next(k for k in arts if "point_anomalies_all" in k and SENSOR in k)
    pa = pd.read_csv(arts[key].get_local_copy(), usecols=["data_datetime", "operational_state"])
    pa["data_datetime"] = pd.to_datetime(pa["data_datetime"], utc=True, errors="coerce")
    pa = pa.dropna(subset=["data_datetime"]).set_index("data_datetime")
    on_series = (pa["operational_state"] == "on")

    dt_s = pd.Series(mae.index).diff().dt.total_seconds().median() or 300.0
    on = on_series.reindex(mae.index, method="nearest").fillna(False)
    cond = mae.where(on.values).ewm(halflife=max(1, int(round(COND_HALFLIFE_H * 3600 / dt_s)))).mean()
    cond = cond.dropna()
    idx = cond.rank(pct=True) * 100.0
    monthly = idx.resample("MS").median()

    x = np.arange(len(monthly)); y = monthly.values
    ok = ~np.isnan(y)
    slope = np.polyfit(x[ok], y[ok], 1)[0] if ok.sum() >= 2 else 0.0
    cur = float(monthly.dropna().iloc[-1]) if len(monthly.dropna()) else float("nan")
    tend = "PIORANDO" if slope > 2 else ("melhorando" if slope < -2 else "estável")
    print(f"{SENSOR}: condição atual={cur:.0f}/100  tendência={slope:+.1f} pts/mês ({tend})")

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 6), sharex=False)

    ax1.plot(idx.index, idx.values, color="#37474f", linewidth=0.8)
    ax1.axhline(50, color="gray", linestyle=":", linewidth=1)
    ax1.fill_between(idx.index, idx.values, 50, where=(idx.values >= 50),
                      color="#c62828", alpha=0.15)
    ax1.fill_between(idx.index, idx.values, 50, where=(idx.values < 50),
                      color="#2e7d32", alpha=0.15)
    ax1.set_ylabel("Condição (percentil 0-100)")
    ax1.set_title(f"{SENSOR} — índice de saúde (resíduo+CNN1D, EWMA {COND_HALFLIFE_H:.0f}h) — "
                   f"não é alarme de evento, é tendência pra manutenção")
    ax1.grid(True, alpha=0.25)

    ax2.bar(monthly.index, monthly.values, width=20, color="#0f6d63", alpha=0.85)
    ax2.axhline(50, color="gray", linestyle=":", linewidth=1)
    ax2.set_ylabel("Mediana mensal")
    ax2.set_xlabel("Tempo")
    ax2.grid(True, alpha=0.25)

    plt.tight_layout()
    plt.savefig(f"{OUT}.png", dpi=150)
    plt.close(fig)

    pd.DataFrame({"mes": monthly.index, "condicao_pct": monthly.values}).to_csv(
        f"{OUT}_monthly.csv", index=False)
    print(f"Salvo: {OUT}.png e {OUT}_monthly.csv")


if __name__ == "__main__":
    main()
