"""Compara o baseline de B-90001A (7 sensores de vibração, sem feature engineering) contra
o retreino com features de vibração (rolling: RMS/kurtose/std/crest-factor; espectral: FFT
em 3 bandas + freq dominante), pra ver se reduz o falso alarme dos sensores mais ruidosos
sinalizados no teste de sanidade (Vibração Motor LA Y, Vibração Bomba LA X, Vibração Bomba
LA Y). Mesma metodologia rigorosa (sweep de half-life + curva oficial + teto de duty + teste
de sanidade contra trecho distante) usada nos scripts anteriores — só compara 2 tasks lado
a lado em vez de reportar 1.

Uso:
  PYTHONPATH=. python scripts/eval_transpetro_vib_features_compare.py
"""
import pandas as pd
from clearml import Task

from src.cnn1d_ae.predictive import _detect_episodes
from scripts.eval_transpetro_halflife_sweep import (
    EQUIPS, sweep_sensor, load_mae, TIME_STEPS,
)
from scripts.sanity_check_transpetro_healthy_window import (
    fa_per_day_in_window, DEBOUNCE_HOURS, NEAR_WINDOW_DAYS,
)

SENSORS = ["Vibração Motor LNA Y", "Vibração Motor LA X", "Vibração Motor LA Y",
          "Vibração Bomba LA X", "Vibração Bomba LA Y", "Vibração Bomba LNA X",
          "Vibração Bomba LNA Y"]
FLAGGED = {"Vibração Motor LA Y", "Vibração Bomba LA X", "Vibração Bomba LA Y"}

BASE_CFG = EQUIPS["B-90001A"]
VIBFEAT_CFG = dict(BASE_CFG, task_id="56f95613c7074f3fa53c65d4ea8db056")


def eval_one(task_id: str, cfg: dict) -> dict:
    task = Task.get_task(task_id=task_id)
    out = {}
    near_start = cfg["detection_ts"] - pd.Timedelta(days=NEAR_WINDOW_DAYS)
    for sensor in SENSORS:
        best_by_h, best_ew_hl = sweep_sensor(task, cfg, sensor)
        if not best_by_h or 72.0 not in best_by_h:
            out[sensor] = None
            continue
        b72 = best_by_h[72.0]
        hl_used, health = best_ew_hl
        mae = load_mae(task, sensor)
        t_end = mae.index + pd.Timedelta(minutes=TIME_STEPS - 1)
        fa_far, n_far = fa_per_day_in_window(health, t_end, b72["threshold"],
                                             cfg["train_end"], near_start)
        honest_fa = max(b72["fa_per_day"], fa_far) if pd.notna(fa_far) else b72["fa_per_day"]
        out[sensor] = dict(hl=hl_used, lead=b72["median_lead_hours"],
                           fa_reportado=b72["fa_per_day"], fa_distante=fa_far,
                           fa_honesto=honest_fa, duty=b72["duty"])
    return out


def main():
    print("Avaliando baseline (sem feature engineering)...")
    base = eval_one(BASE_CFG["task_id"], BASE_CFG)
    print("Avaliando com features de vibração (rolling+espectral)...")
    vib = eval_one(VIBFEAT_CFG["task_id"], VIBFEAT_CFG)

    rows = []
    for sensor in SENSORS:
        b, v = base.get(sensor), vib.get(sensor)
        rows.append(dict(
            sensor=sensor, sinalizado="*" if sensor in FLAGGED else "",
            base_lead=b["lead"] if b else None, base_fa_honesto=b["fa_honesto"] if b else None,
            base_duty=b["duty"] if b else None,
            vib_lead=v["lead"] if v else None, vib_fa_honesto=v["fa_honesto"] if v else None,
            vib_duty=v["duty"] if v else None,
        ))
    df = pd.DataFrame(rows)
    df["delta_fa"] = df["vib_fa_honesto"] - df["base_fa_honesto"]
    df["delta_lead"] = df["vib_lead"] - df["base_lead"]

    out_csv = "eval_predictive_out/transpetro/vib_features_compare.csv"
    df.to_csv(out_csv, index=False)
    pd.set_option("display.width", 220, "display.max_columns", 20)
    fmt = df.copy()
    for c in ["base_lead", "base_fa_honesto", "base_duty", "vib_lead", "vib_fa_honesto",
             "vib_duty", "delta_fa", "delta_lead"]:
        fmt[c] = df[c].map(lambda x: f"{x:.3f}" if pd.notna(x) else "-")
    print("\n" + fmt.to_string(index=False))
    print(f"\ncsv: {out_csv}")
    print("\n* = sensor sinalizado como suspeito no teste de sanidade anterior")


if __name__ == "__main__":
    main()
