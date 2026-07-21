"""Estudo de caso de antecipação para os equipamentos Transpetro (B-4064A, B-90001A).

Diferente de Cabiunas (dezenas de incidentes por sensor -> recall/FA estatístico), aqui
há 1 falha catastrófica documentada por equipamento (N=1). A avaliação correta não é
recall%, é: treinar só com dado anterior à degradação, mostrar o health-index subindo
antes da falha e medir o lead time até a detecção formal. Threshold = média + Yσ do
EWMA calculado SÓ no período de treino (evita vazar estatística do teste).

Uso:
  PYTHONPATH=. python scripts/eval_transpetro_case_study.py
"""
import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from clearml import Task

OUT_DIR = "eval_predictive_out/transpetro"
# mesmo half-life usado pelo pipeline (PREDICTIVE_EWMA_HALF_LIFE_HOURS nos configs) —
# necessário p/ o threshold da curva oficial (predictive_curve_H*.csv) ficar na mesma
# escala da linha de health-index plotada aqui.
HALF_LIFE_HOURS = 4.0
DEBOUNCE_HOURS = 2.0
Y_SIGMA = 3.0

EQUIPS = {
    "B-4064A": dict(
        task_id="2ecc70487d3b49c599145253720ef4b3",
        train_end=pd.Timestamp("2024-08-01"),
        failure_ts=pd.Timestamp("2024-08-27 00:00:00"),   # data do roçamento
        detection_ts=pd.Timestamp("2024-08-30 07:58:00"),  # detecção formal
        outage_start=pd.Timestamp("2024-08-26"),
        sensors=["Pressão Sucção", "Pressão Descarga", "Corrente", "Vibração Bomba LNA",
                "Temperatura Bomba LA", "Temperatura Bomba LNA", "Temperatura Motor LA",
                "Temperatura Motor LNA", "Densidade"],
    ),
    "B-90001A": dict(
        task_id="ad0bb221e6bf4e4aa3e2bce678923f0f",
        train_end=pd.Timestamp("2021-07-01"),
        failure_ts=pd.Timestamp("2021-08-28 00:00:00"),
        detection_ts=pd.Timestamp("2021-08-28 00:00:00"),
        outage_start=None,
        sensors=["Pressão Descarga", "Pressão Sucção", "Vibração Motor LNA Y",
                "Vibração Motor LA X", "Vibração Motor LA Y", "Vibração Bomba LA X",
                "Vibração Bomba LA Y", "Vibração Bomba LNA X", "Vibração Bomba LNA Y"],
    ),
}


def load_mae(task: Task, sensor: str) -> pd.Series:
    key = next((k for k in task.artifacts if "sequence_scores_all" in k and k.startswith(sensor)), None)
    if key is None:
        return pd.Series(dtype=float)
    d = pd.read_csv(task.artifacts[key].get_local_copy())
    d["seq_start_time"] = pd.to_datetime(d["seq_start_time"], errors="coerce")
    d = d.dropna(subset=["seq_start_time"]).sort_values("seq_start_time")
    return d.set_index("seq_start_time")["mae_seq"]


def best_point_from_curve(task: Task, sensor: str, horizon_label: str) -> dict | None:
    """Usa a curva sigma-sweep JÁ CALCULADA pelo pipeline (predictive.py::
    compute_predictive_curve, half-life/horizonte de produção) em vez de recalcular
    threshold ad-hoc: pega o y_sigma MAIS ALTO (mais estrito, mais defensável) que
    ainda captura o único incidente (recall==1.0) e reporta FA/lead nesse ponto.
    Com N=1 incidente, FA/dia é descritiva (contra 1 evento), não uma métrica robusta.
    """
    key = next((k for k in task.artifacts
               if f"predictive_curve_{horizon_label}" in k and k.startswith(sensor)), None)
    if key is None:
        return None
    d = pd.read_csv(task.artifacts[key].get_local_copy())
    hit = d[d["recall"] >= 1.0]
    if hit.empty:
        return None
    row = hit.loc[hit["y_sigma"].idxmax()]
    return dict(y_sigma=float(row["y_sigma"]), threshold=float(row["threshold"]),
               fa_per_day=float(row["fa_per_day"]), median_lead_hours=float(row["median_lead_hours"]),
               n_episodes=int(row["n_episodes"]))


def first_sustained_crossing(alert: pd.Series, debounce_hours: float) -> pd.Timestamp | None:
    """Primeiro instante em que o alerta fica ligado continuamente por >= debounce_hours."""
    if not alert.any():
        return None
    idx = alert.index
    on = alert.values
    i = 0
    n = len(on)
    while i < n:
        if on[i]:
            j = i
            while j < n and on[j]:
                j += 1
            if (idx[j - 1] - idx[i]) >= pd.Timedelta(hours=debounce_hours):
                return idx[i]
            i = j
        else:
            i += 1
    return None


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    rows = []
    for equip, cfg in EQUIPS.items():
        task = Task.get_task(task_id=cfg["task_id"])
        print(f"\n===== {equip} =====")
        for sensor in cfg["sensors"]:
            mae = load_mae(task, sensor)
            if mae.empty:
                print(f"  [WARN] {sensor}: sem artefato")
                continue
            hl_pts = max(1, int(round(pd.Timedelta(hours=HALF_LIFE_HOURS) /
                                      (mae.index[1] - mae.index[0]))))
            ew = mae.ewm(halflife=hl_pts).mean()

            train = ew[ew.index < cfg["train_end"]]
            mu, sig = float(train.mean()), float(train.std())
            thr = mu + Y_SIGMA * sig

            test = ew[ew.index >= cfg["train_end"]]
            alert = test >= thr
            cross = first_sustained_crossing(alert, DEBOUNCE_HOURS)

            lead_h = None
            if cross is not None:
                lead_h = (cfg["detection_ts"] - cross).total_seconds() / 3600.0

            best24 = best_point_from_curve(task, sensor, "H24h")
            best72 = best_point_from_curve(task, sensor, "H72h")

            rows.append(dict(
                equip=equip, sensor=sensor,
                adhoc_first_crossing=cross, adhoc_lead_hours=lead_h,
                lead24_hours=best24["median_lead_hours"] if best24 else None,
                fa24_per_day=best24["fa_per_day"] if best24 else None,
                lead72_hours=best72["median_lead_hours"] if best72 else None,
                fa72_per_day=best72["fa_per_day"] if best72 else None,
            ))
            l72 = f"{best72['median_lead_hours']:.1f}h @ FA={best72['fa_per_day']:.2f}/dia" if best72 else "sem captura"
            l24 = f"{best24['median_lead_hours']:.1f}h @ FA={best24['fa_per_day']:.2f}/dia" if best24 else "sem captura"
            print(f"  {sensor:22s} H24h: {l24:28s} | H72h: {l72}")

            fig, axes = plt.subplots(2, 1, figsize=(12, 6), sharex=True)
            axes[0].plot(mae.index, mae.values, lw=0.4, alpha=0.6, color="steelblue")
            axes[0].set_ylabel("MAE (seq)")
            axes[0].set_title(f"{equip} — {sensor}")
            axes[1].plot(ew.index, ew.values, lw=0.8, color="darkorange", label="health (EWMA)")
            thr_label = thr
            if best72 is not None:
                thr_label = best72["threshold"]
                axes[1].axhline(thr_label, color="red", ls="--", lw=1,
                                label=f"threshold curva oficial H72h (y={best72['y_sigma']:.2f}σ)")
            else:
                axes[1].axhline(thr_label, color="red", ls="--", lw=1,
                                label=f"threshold (μ+{Y_SIGMA:g}σ treino, ad-hoc)")
            axes[1].axvline(cfg["train_end"], color="gray", ls=":", lw=1, label="fim do treino")
            axes[1].axvline(cfg["detection_ts"], color="black", ls="-", lw=1.2, label="detecção formal")
            if cross is not None:
                axes[1].axvline(cross, color="green", ls="--", lw=1.2, label="1º cruzamento sustentado")
            if cfg["outage_start"] is not None:
                axes[1].axvspan(cfg["outage_start"], ew.index.max(), color="red", alpha=0.05)
            axes[1].set_ylabel("health index")
            axes[1].legend(fontsize=7, loc="upper left")
            fig.tight_layout()
            safe_sensor = sensor.replace(" ", "_").replace("/", "_")
            fig.savefig(f"{OUT_DIR}/{equip}_{safe_sensor}.png", dpi=110)
            plt.close(fig)

    df = pd.DataFrame(rows)
    out_csv = f"{OUT_DIR}/case_study_summary.csv"
    df.to_csv(out_csv, index=False)
    print(f"\ncsv: {out_csv}")
    pd.set_option("display.width", 160, "display.max_columns", 20)
    out = df.copy()
    for c in ["adhoc_lead_hours", "lead24_hours", "fa24_per_day", "lead72_hours", "fa72_per_day"]:
        out[c] = df[c].map(lambda x: f"{x:.1f}" if pd.notna(x) else "-")
    print(out.to_string(index=False))


if __name__ == "__main__":
    main()
