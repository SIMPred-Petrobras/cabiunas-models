"""Sweep de half-life por sensor para os equipamentos Transpetro (B-4064A, B-90001A) —
SEM retreinar. Corrige a causa raiz encontrada após o fix de duty cycle: o pipeline usa
um único PREDICTIVE_EWMA_HALF_LIFE_HOURS fixo (4h, herdado do default de Cabiunas) para
todos os sensores, mas cada sinal (pressão/temperatura/corrente/vibração) tem sua própria
dinâmica — half-life errado faz a curva nunca atingir recall=1.0 sob duty<=25%, mesmo
quando o mae_seq já treinado carrega sinal suficiente.

Reusa as funções JÁ VALIDADAS do pipeline (src/cnn1d_ae/predictive.py):
compute_health_index_ewma + compute_predictive_curve (recall/FA/lead por episódio com
debounce — rigoroso, não um atalho ad-hoc), varrendo a grade de half-life e escolhendo,
por sensor, o ponto com recall=1.0, duty<=0.25 (mesmo teto usado em Cabiunas e na correção
anterior desta sessão), maior y_sigma (mais estrito/defensável) e menor FA no desempate.

Uso:
  PYTHONPATH=. python scripts/eval_transpetro_halflife_sweep.py
"""
import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from clearml import Task

from src.cnn1d_ae.predictive import compute_health_index_ewma, compute_predictive_curve

OUT_DIR = "eval_predictive_out/transpetro"
HL_GRID_HOURS = [0.5, 1.0, 2.0, 4.0, 8.0, 16.0]
HORIZONS = [24.0, 72.0]
DEBOUNCE_HOURS = 8.0
MAX_DUTY = 0.25
TIME_STEPS = 48            # mesmo TIME_STEPS dos configs (48min de contexto, dado a 1min)

EQUIPS = {
    "B-4064A": dict(
        task_id="2ecc70487d3b49c599145253720ef4b3",
        raw_csv="../dados/transpetro/B-4064A_1min.csv",
        running_col="Corrente", running_thr=5.0,
        train_end=pd.Timestamp("2024-08-01"),
        detection_ts=pd.Timestamp("2024-08-30 07:58:00"),
        outage_start=pd.Timestamp("2024-08-26"),
        sensors=["Pressão Sucção", "Pressão Descarga", "Corrente", "Vibração Bomba LNA",
                "Temperatura Bomba LA", "Temperatura Bomba LNA", "Temperatura Motor LA",
                "Temperatura Motor LNA", "Densidade"],
    ),
    "B-90001A": dict(
        task_id="ad0bb221e6bf4e4aa3e2bce678923f0f",
        raw_csv="../dados/transpetro/B-90001A_1min.csv",
        running_col=None, running_thr=None,  # off negligível (~0.006%), tratado como sempre ON
        train_end=pd.Timestamp("2021-07-01"),
        detection_ts=pd.Timestamp("2021-08-28 00:00:00"),
        outage_start=None,
        sensors=["Pressão Descarga", "Pressão Sucção", "Vibração Motor LNA Y",
                "Vibração Motor LA X", "Vibração Motor LA Y", "Vibração Bomba LA X",
                "Vibração Bomba LA Y", "Vibração Bomba LNA X", "Vibração Bomba LNA Y"],
    ),
    "B-8802B": dict(
        task_id="6b6625bc03ba4236b69cd8dcfa3a0b33",
        raw_csv="../dados/transpetro/B-8802B_1min.csv",
        running_col=None, running_thr=None,  # sem coluna de corrente/vazão disponível
        train_end=pd.Timestamp("2022-06-08"),
        detection_ts=pd.Timestamp("2022-07-06 10:00:00"),
        outage_start=None,
        sensors=["Pressão Sucção", "Pressão Descarga", "Vibração Bomba LA",
                "Vibração Bomba LNA", "Temperatura Bomba LA", "Temperatura Bomba LNA",
                "Temperatura Motor LA", "Temperatura Motor LNA"],
    ),
    "B-402E": dict(
        task_id="c5e0ec4603a349cc9d34fa390d9ce7d0",
        raw_csv="../dados/transpetro/B-402E_1min.csv",
        running_col="Corrente", running_thr=5.0,
        train_end=pd.Timestamp("2019-08-01"),
        detection_ts=pd.Timestamp("2019-10-30 11:06:00"),
        outage_start=None,
        sensors=["Pressão Sucção", "Pressão Descarga", "Corrente", "Vazão",
                "Vibração Bomba LA", "Temperatura Estator U", "Temperatura Estator V",
                "Temperatura Estator Wa", "Temperatura Estator Wb",
                "Temperatura Mancal LA Motor", "Temperatura Mancal LNA Motor",
                "Temperatura Mancal Ext. Escora LNA Bomba",
                "Temperatura Mancal Int. Escora LNA Bomba",
                "Temperatura Mancal Radial LA Bomba", "Temperatura Mancal Radial LNA Bomba"],
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


def running_frac_forward(raw_csv: str, running_col: str | None, running_thr: float | None,
                         seq_starts: pd.DatetimeIndex) -> np.ndarray:
    """Fração de tempo ON na janela [start, start+TIME_STEPS-1min] de cada sequência."""
    if running_col is None:
        return np.ones(len(seq_starts), dtype=float)
    raw = pd.read_csv(raw_csv, usecols=["data_datetime", running_col])
    raw["data_datetime"] = pd.to_datetime(raw["data_datetime"], errors="coerce")
    raw = raw.dropna(subset=["data_datetime"]).set_index("data_datetime").sort_index()
    on = (pd.to_numeric(raw[running_col], errors="coerce") > running_thr).astype(float)
    # media movel "para frente": rolling reverso, depois desfaz a reversao
    fwd_mean = on.iloc[::-1].rolling(window=TIME_STEPS, min_periods=1).mean().iloc[::-1]
    return fwd_mean.reindex(seq_starts, method="nearest").fillna(0.0).to_numpy()


def duty_of(ew: np.ndarray, t_end: pd.DatetimeIndex, train_end: pd.Timestamp, thr: float) -> float:
    mask_test = t_end >= train_end
    if not mask_test.any():
        return 1.0
    return float((ew[mask_test] >= thr).mean())


def sweep_sensor(task, cfg, sensor: str):
    mae = load_mae(task, sensor)
    if mae.empty:
        return None, None
    # dt entre sequências consecutivas (= STRIDE * amostragem raw, ex.: 5min a 60s) —
    # necessário p/ compute_health_index_ewma converter half_life_hours em alpha corretamente.
    dt_seconds = float((mae.index[1] - mae.index[0]).total_seconds())
    t_end = mae.index + pd.Timedelta(minutes=TIME_STEPS - 1)
    t_end_seconds = t_end.values.astype("datetime64[s]").astype("int64").astype(float)
    seq_run_frac = running_frac_forward(cfg["raw_csv"], cfg["running_col"], cfg["running_thr"], mae.index)
    seq_run_full = seq_run_frac >= 0.999
    inc_seconds = np.array([cfg["detection_ts"].timestamp()])

    best_by_horizon = {}
    best_ew_hl = None
    for hl in HL_GRID_HOURS:
        health = compute_health_index_ewma(mae.to_numpy(), seq_run_frac,
                                           half_life_hours=hl, dt_seconds=dt_seconds)
        for h in HORIZONS:
            curve = compute_predictive_curve(
                health_ewma=health, seq_running_full=seq_run_full,
                t_end_seconds=t_end_seconds, incident_seconds=inc_seconds,
                horizon_hours=h, debounce_hours=DEBOUNCE_HOURS,
                sigma_y_min=-2.0, sigma_y_max=5.0, n_threshold_steps=60,
            )
            if curve.empty:
                continue
            hit = curve[curve["recall"] >= 1.0].copy()
            if hit.empty:
                continue
            hit["duty"] = hit["threshold"].map(lambda thr: duty_of(health, t_end, cfg["train_end"], thr))
            hit = hit[hit["duty"] <= MAX_DUTY]
            if hit.empty:
                continue
            hit = hit.sort_values(["y_sigma", "fa_per_day"], ascending=[False, True])
            row = hit.iloc[0]
            cand = dict(hl=hl, y_sigma=float(row["y_sigma"]), threshold=float(row["threshold"]),
                       fa_per_day=float(row["fa_per_day"]), median_lead_hours=float(row["median_lead_hours"]),
                       n_episodes=int(row["n_episodes"]), duty=float(row["duty"]))
            prev = best_by_horizon.get(h)
            if prev is None or cand["median_lead_hours"] > prev["median_lead_hours"]:
                best_by_horizon[h] = cand
                if h == 72.0:
                    best_ew_hl = (hl, health)

    return best_by_horizon, best_ew_hl


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    rows = []
    for equip, cfg in EQUIPS.items():
        task = Task.get_task(task_id=cfg["task_id"])
        print(f"\n===== {equip} =====")
        for sensor in cfg["sensors"]:
            best_by_h, best_ew_hl = sweep_sensor(task, cfg, sensor)
            if best_by_h is None:
                print(f"  [WARN] {sensor}: sem artefato")
                continue
            b24, b72 = best_by_h.get(24.0), best_by_h.get(72.0)
            row = dict(equip=equip, sensor=sensor,
                      hl24=b24["hl"] if b24 else None,
                      lead24_hours=b24["median_lead_hours"] if b24 else None,
                      fa24_per_day=b24["fa_per_day"] if b24 else None,
                      duty24=b24["duty"] if b24 else None,
                      hl72=b72["hl"] if b72 else None,
                      lead72_hours=b72["median_lead_hours"] if b72 else None,
                      fa72_per_day=b72["fa_per_day"] if b72 else None,
                      duty72=b72["duty"] if b72 else None)
            rows.append(row)
            l24 = (f"hl={b24['hl']:.1f}h {b24['median_lead_hours']:.1f}h @FA={b24['fa_per_day']:.2f}/d "
                  f"duty={b24['duty']*100:.0f}%") if b24 else "sem sinal"
            l72 = (f"hl={b72['hl']:.1f}h {b72['median_lead_hours']:.1f}h @FA={b72['fa_per_day']:.2f}/d "
                  f"duty={b72['duty']*100:.0f}%") if b72 else "sem sinal"
            print(f"  {sensor:22s} H24h: {l24:46s} | H72h: {l72}")

            if best_ew_hl is not None:
                hl_used, health = best_ew_hl
                mae = load_mae(task, sensor)
                fig, axes = plt.subplots(2, 1, figsize=(12, 6), sharex=True)
                axes[0].plot(mae.index, mae.values, lw=0.4, alpha=0.6, color="steelblue")
                axes[0].set_ylabel("MAE (seq)")
                axes[0].set_title(f"{equip} — {sensor} (half-life vencedor H72h = {hl_used:g}h)")
                axes[1].plot(mae.index, health, lw=0.8, color="darkorange", label=f"health (EWMA hl={hl_used:g}h)")
                if b72:
                    thr = b72["threshold"]
                    alert_mask = health >= thr
                    axes[1].axhline(thr, color="red", ls="--", lw=1,
                                    label=f"threshold (y={b72['y_sigma']:.2f}σ, duty={b72['duty']*100:.0f}%)")
                    axes[1].fill_between(mae.index, thr, health, where=alert_mask,
                                         color="red", alpha=0.15, interpolate=True, label="região em alerta")
                    # pontos discretos de deteccao (onde o alerta esta ativo), marcados nos
                    # dois paineis para correlacionar visualmente com o dado bruto (MAE)
                    axes[1].scatter(mae.index[alert_mask], health[alert_mask], s=10, color="crimson",
                                    zorder=5, label="pontos em alerta")
                    axes[0].scatter(mae.index[alert_mask], mae.values[alert_mask], s=10, color="crimson",
                                    zorder=5, label="pontos em alerta")
                    axes[0].legend(fontsize=7, loc="upper left")
                axes[1].axvline(cfg["train_end"], color="gray", ls=":", lw=1, label="fim do treino")
                axes[1].axvline(cfg["detection_ts"], color="black", ls="-", lw=1.2, label="detecção formal")
                if cfg["outage_start"] is not None:
                    axes[1].axvspan(cfg["outage_start"], mae.index.max(), color="red", alpha=0.05)
                axes[1].set_ylabel("health index")
                axes[1].legend(fontsize=7, loc="upper left")
                fig.tight_layout()
                safe_sensor = sensor.replace(" ", "_").replace("/", "_")
                fig.savefig(f"{OUT_DIR}/{equip}_{safe_sensor}.png", dpi=110)
                plt.close(fig)

    df = pd.DataFrame(rows)
    out_csv = f"{OUT_DIR}/halflife_sweep_summary.csv"
    df.to_csv(out_csv, index=False)
    print(f"\ncsv: {out_csv}")
    pd.set_option("display.width", 200, "display.max_columns", 20)
    out = df.copy()
    for c in ["hl24", "lead24_hours", "fa24_per_day", "duty24", "hl72", "lead72_hours", "fa72_per_day", "duty72"]:
        out[c] = df[c].map(lambda x: f"{x:.2f}" if pd.notna(x) else "-")
    print(out.to_string(index=False))
    n_ok = df["lead72_hours"].notna().sum()
    print(f"\n{n_ok}/{len(df)} sensores com sinal valido sob duty<={MAX_DUTY*100:.0f}% (H72h, antes do sweep: 6/18)")


if __name__ == "__main__":
    main()
