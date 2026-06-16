"""Figura de validação OUT-OF-SAMPLE 2024: modelo de 2025 aplicado em 2024 (nunca
visto), ponto de operação FIXO de 2025. Painel por sensor: sinal bruto + onsets de
alarme (verde) + janelas de detecção/alerta (vermelho) + OFF sombreado; painel da
EWMA-MAE vs threshold absoluto. Mais uma barra-resumo de recall dos 7 sensores.

Uso: PYTHONPATH=. python scripts/fig_oos_2024.py
"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import numpy as np
import pandas as pd
from clearml import Task
from tensorflow import keras

import scripts.eval_per_sensor_level as E
from src.cnn1d_ae.inference import load_bundle, score_dataframe

SENS2024 = "/home/thallys/Downloads/2024.csv"
PREFIX = "bapiha02-"
ALARM = "../dados/alarmes_selecionados_turbina_a.csv"
TASK = "58bc393c1d7a4e42815236e8897abc88"
DEPLOY_CSV = "eval_predictive_out/validate_deployed_2024.csv"
OUT = "eval_predictive_out/fig_oos_2024"
RUN_THR = 50.0
HORIZON, STICKY = 8.0, 12.0
PLOT_SENSORS = ["TC382_03_A", "T5_AVG_A"]
W0, W1 = pd.Timestamp("2024-01-01", tz="UTC"), pd.Timestamp("2024-12-31", tz="UTC")


def main():
    # números deployáveis (ponto de operação q=0.9/0.92 dos bundles), não o piso q=0.5
    ops = pd.read_csv(DEPLOY_CSV).set_index("sensor")
    task = Task.get_task(task_id=TASK)
    alarms = E.load_alarms_gap(ALARM)

    need = list(dict.fromkeys(PLOT_SENSORS + ["NGP_A"]))
    # ler só colunas necessárias (com prefixo)
    head = pd.read_csv(SENS2024, nrows=1)
    tcol = next(c for c in head.columns if "datetime" in c.lower() or c.lower() == "data")
    want = [tcol] + [PREFIX + c for c in need if PREFIX + c in head.columns]
    df = pd.read_csv(SENS2024, usecols=want, low_memory=False)
    df.columns = [c[len(PREFIX):] if c.startswith(PREFIX) else c for c in df.columns]
    df[tcol] = pd.to_datetime(df[tcol], utc=True, errors="coerce")
    df = df.dropna(subset=[tcol]).set_index(tcol).sort_index()
    df = df[(df.index >= W0) & (df.index <= W1)]
    for c in need:
        df[c] = pd.to_numeric(df[c], errors="coerce")

    for s in PLOT_SENSORS:
        rec = float(ops.loc[s, "recall"]); fa = float(ops.loc[s, "fa_per_day"]); n = int(ops.loc[s, "n_inc"])
        duty = float(ops.loc[s, "duty_cycle"])
        # threshold/half-life DEPLOYÁVEIS direto do bundle (production_alerting)
        bundle = load_bundle(f"production_bundles/{s}_inference_bundle.json")
        pa = bundle["production_alerting"]; hl = float(pa["half_life_hours"]); athr = float(pa["ewma_abs_threshold"])
        model = keras.models.load_model(task.artifacts[f"{s}_model_keras"].get_local_copy(), compile=False)

        scored = score_dataframe(model, bundle, df[[s, "NGP_A"]])
        mae = pd.Series(scored["mae_seq"].to_numpy(), index=pd.DatetimeIndex(scored["seq_end_time"]))
        dt_s = pd.Series(mae.index).diff().dt.total_seconds().median() or 300.0
        on = df["NGP_A"].reindex(mae.index, method="nearest") > RUN_THR
        ew = mae.ewm(halflife=max(1, int(round(hl * 3600 / dt_s)))).mean()
        alert = pd.Series((ew.to_numpy() >= athr) & on.values, index=mae.index)

        inc = E.cluster_incidents([t for t in alarms.get(s, []) if W0 <= t <= W1], gap_hours=E.GAP_HOURS)
        on_inc = df["NGP_A"].reindex(pd.DatetimeIndex(inc), method="nearest") > RUN_THR if inc else pd.Series([], dtype=bool)
        inc_on = [t for t, o in zip(inc, on_inc.values) if o]

        sig = df[s].where(df["NGP_A"] > RUN_THR)  # esconde OFF

        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(15, 7), sharex=True)
        ax1.plot(sig.index, sig.values, color="#1565c0", lw=0.5, label="Sensor (ON)")
        ymin, ymax = float(np.nanmin(sig.values)), float(np.nanmax(sig.values))
        for t in inc_on:
            ax1.axvline(t, color="#2e7d32", ls="--", lw=1.0, alpha=0.7)
        ax1.scatter(inc_on, [ymax] * len(inc_on), marker="v", color="#2e7d32", s=40,
                    label=f"Alarme/incidente ({len(inc_on)})", zorder=5)
        # janelas de alerta (detecção)
        ax1.fill_between(alert.index, ymin, ymax, where=alert.values, color="red", alpha=0.18,
                         step="mid", label="Alerta do modelo")
        nh = int(round(rec * n))
        ax1.set_title(f"OOS 2024 — {s}: recall {rec*100:.0f}% ({nh}/{n}) · alarme ligado {duty*100:.0f}% do tempo · "
                      f"FA {fa:.3f}/dia  (q={pa['threshold_q']}, modelo 2025 OOS)")
        ax1.set_ylabel("valor (bruto)"); ax1.legend(loc="upper right", fontsize="small"); ax1.grid(alpha=0.25)

        ax2.plot(ew.index, ew.values, color="#37474f", lw=0.6, label=f"EWMA-MAE (hl={hl}h)")
        ax2.axhline(athr, color="red", ls="--", lw=1.2, label=f"threshold deployável (2025) {athr:.4f}")
        ax2.set_ylabel("EWMA do erro"); ax2.set_xlabel("2024"); ax2.legend(loc="upper right", fontsize="small"); ax2.grid(alpha=0.25)
        ax2.xaxis.set_major_formatter(mdates.DateFormatter("%b/%y"))
        out = f"{OUT}_{s}.png"
        plt.savefig(out, dpi=150, bbox_inches="tight"); plt.close(fig)
        print(f"salvo: {out}  (recall {rec*100:.0f}%, duty {duty*100:.0f}%, {n} incidentes, FA {fa:.3f})")

    # barra-resumo dos 7 (recall + duty)
    m = ops[ops.n_inc > 0].sort_values("n_inc", ascending=False)
    x = np.arange(len(m)); w = 0.38
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.bar(x - w/2, m["recall"] * 100, w, color="#2e7d32", alpha=0.85, label="recall")
    ax.bar(x + w/2, m["duty_cycle"] * 100, w, color="#ef6c00", alpha=0.85, label="duty (tempo-em-alerta)")
    for i, r in enumerate(m["recall"]):
        ax.text(i - w/2, r * 100 + 1, f"{int(m['n_inc'].iloc[i])}", ha="center", fontsize=8)
    ax.set_xticks(x); ax.set_xticklabels(m.index, rotation=30, ha="right")
    ax.set_ylim(0, 108); ax.set_ylabel("%"); ax.legend(loc="upper right", fontsize="small")
    ax.set_title("Validação OOS 2024 (ponto deployável) — recall vs duty por sensor (nº = incidentes)")
    plt.tight_layout()
    plt.savefig(f"{OUT}_resumo.png", dpi=150); plt.close(fig)
    print(f"salvo: {OUT}_resumo.png")


if __name__ == "__main__":
    main()
