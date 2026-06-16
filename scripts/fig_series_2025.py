"""Série temporal 2025 (ano de treino) por sensor: sinal bruto (OFF escondido) +
eventos de alarme (verde) + anomalias/alertas do modelo (vermelho), no ponto de
operação DEPLOYÁVEL (threshold absoluto + half-life dos bundles). Usa o MAE de 2025
já salvo nos artefatos do treino (sem re-inferir).

Uso: PYTHONPATH=. python scripts/fig_series_2025.py
"""
import argparse
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import numpy as np
import pandas as pd
from clearml import Task

import scripts.eval_per_sensor_level as E
from src.cnn1d_ae.inference import load_bundle

DS = "/home/thallys/.clearml/cache/storage_manager/datasets/ds_424e5b589e13402d9d95371a317e85c9"
ALARM = "../dados/alarmes_selecionados_turbina_a.csv"
SENS2025 = f"{DS}/sensores_filtrados_Interpolados_2025.csv"
TASK = "58bc393c1d7a4e42815236e8897abc88"
OUT = "eval_predictive_out/fig_series_2025"
RUN_THR, HORIZON, STICKY = 50.0, 8.0, 12.0
PLOT_SENSORS = ["TC382_03_A", "T5_AVG_A"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--q", type=float, default=None,
                    help="quantil override: recalcula threshold no q dado (OFF excl). Default = usa o bundle deployável.")
    args = ap.parse_args()
    task = Task.get_task(task_id=TASK)
    alarms = E.load_alarms_gap(ALARM)
    mae_all = E.load_mae_series(task, PLOT_SENSORS)

    cols = ["data_datetime", "NGP_A"] + PLOT_SENSORS
    raw = pd.read_csv(SENS2025, usecols=lambda c: c in cols)
    raw["data_datetime"] = pd.to_datetime(raw["data_datetime"], utc=True, errors="coerce")
    raw = raw.dropna(subset=["data_datetime"]).set_index("data_datetime").sort_index()
    for c in PLOT_SENSORS + ["NGP_A"]:
        raw[c] = pd.to_numeric(raw[c], errors="coerce")

    for s in PLOT_SENSORS:
        pa = load_bundle(f"production_bundles/{s}_inference_bundle.json")["production_alerting"]
        hl = float(pa["half_life_hours"]); q = pa["threshold_q"]
        mae = mae_all[s]
        dt_s = pd.Series(mae.index).diff().dt.total_seconds().median() or 300.0
        on = raw["NGP_A"].reindex(mae.index, method="nearest") > RUN_THR
        ew = mae.ewm(halflife=max(1, int(round(hl * 3600 / dt_s)))).mean()
        if args.q is not None:  # recalcula threshold no quantil pedido (OFF excl), sobre 2025
            q = args.q
            athr = float(ew[on.values].quantile(q))
        else:
            athr = float(pa["ewma_abs_threshold"])
        alert = pd.Series((ew.to_numpy() >= athr) & on.values, index=mae.index)
        duty = float((alert.values & on.values).sum() / max(on.values.sum(), 1))

        w0, w1 = mae.index.min(), mae.index.max()
        inc = E.cluster_incidents([t for t in alarms.get(s, []) if w0 <= t <= w1], gap_hours=E.GAP_HOURS)
        on_i = raw["NGP_A"].reindex(pd.DatetimeIndex(inc), method="nearest") > RUN_THR if inc else pd.Series([], dtype=bool)
        inc = [t for t, o in zip(inc, on_i.values) if o]
        # recall (sticky + episódio)
        al = alert.copy()
        if al.any():
            idx = al.index; res = al.values.copy(); td = pd.Timedelta(hours=STICKY)
            for p in np.where(res)[0]:
                res[p:idx.searchsorted(idx[p] + td, side="right")] = True
            al = pd.Series(res, index=idx)
        als = np.array([t.timestamp() for t in al.index[al]]); hs = HORIZON * 3600
        inc_s = np.array([t.timestamp() for t in inc])
        nh = sum(1 for ti in inc_s if als.size and np.any((als >= ti - hs) & (als <= ti)))
        rec = nh / len(inc) if inc else float("nan")

        sig = raw[s].where(raw["NGP_A"] > RUN_THR)
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(15, 7), sharex=True)
        ax1.plot(sig.index, sig.values, color="#1565c0", lw=0.5, label="Sensor (ligado)")
        ymin, ymax = float(np.nanmin(sig.values)), float(np.nanmax(sig.values))
        for t in inc:
            ax1.axvline(t, color="#2e7d32", ls="--", lw=1.0, alpha=0.7)
        ax1.scatter(inc, [ymax] * len(inc), marker="v", color="#2e7d32", s=40,
                    label=f"Evento de alarme ({len(inc)})", zorder=5)
        ax1.fill_between(alert.index, ymin, ymax, where=alert.values, color="red", alpha=0.18,
                         step="mid", label="Anomalia/alerta do modelo")
        ax1.set_title(f"2025 (treino) — {s}: recall {rec*100:.0f}% ({nh}/{len(inc)}) · "
                      f"alarme ligado {duty*100:.0f}% do tempo  (q={q}, ponto deployável)")
        ax1.set_ylabel("valor (bruto)"); ax1.legend(loc="upper right", fontsize="small"); ax1.grid(alpha=0.25)

        ax2.plot(ew.index, ew.values, color="#37474f", lw=0.6, label=f"EWMA-MAE (hl={hl}h)")
        ax2.axhline(athr, color="red", ls="--", lw=1.2, label=f"threshold deployável {athr:.4f}")
        ax2.set_ylabel("EWMA do erro"); ax2.set_xlabel("2025"); ax2.legend(loc="upper right", fontsize="small"); ax2.grid(alpha=0.25)
        ax2.xaxis.set_major_formatter(mdates.DateFormatter("%b/%y"))
        qtag = f"_q{int(round(args.q*100))}" if args.q is not None else ""
        out = f"{OUT}_{s}{qtag}.png"
        plt.savefig(out, dpi=150, bbox_inches="tight"); plt.close(fig)
        print(f"salvo: {out}  (recall {rec*100:.0f}%, duty {duty*100:.0f}%, {len(inc)} alarmes)")


if __name__ == "__main__":
    main()
