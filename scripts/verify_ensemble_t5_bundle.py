"""Verifica o bundle deployável do ensemble T5-via-TC382: roda o caminho de scoring
real (`ensemble_t5_bundle_score.score_t5_proxy_ensemble`, que usa os bundles finalizados
em `production_bundles/t5_ensemble_via_tc382/`) sobre o histórico bruto e mede
recall/FA contra os incidentes reais de T5_AVG_A (HI/HIHI-only).

Nota importante: `score_production` (src/cnn1d_ae/inference.py) NÃO implementa sticky
alert (só EWMA-threshold + debounce) — `sticky_hours` no bundle é só metadado. O sweep
que achou 40,0%/FA=0,104 usava sticky_hours=12 (só existe em avaliação, não no scoring
de produção real) — então o número do bundle deployado é esperado ser diferente do
sweep, não é um bug de paridade.
"""
from __future__ import annotations

import pandas as pd

from scripts.ensemble_t5_bundle_score import score_t5_proxy_ensemble
from scripts.eval_per_sensor_level import ALARM_CSV_DEFAULT, load_alarms_gap, cluster_incidents

RAW_CSV = "../dados/sensores_2024h2_2025_2026_30s.csv"
HORIZON_H = 8.0


def main() -> None:
    print("[1/3] Carregando série bruta...")
    cols = ["data_datetime", "RUNNING_A", "TC382_01_A", "TC382_02_A", "TC382_03_A",
            "TC382_04_A", "TC382_05_A", "TC382_06_A"]
    raw = pd.read_csv(RAW_CSV, low_memory=False, usecols=cols)
    t = pd.to_datetime(raw["data_datetime"], utc=True, errors="coerce")
    raw = raw.loc[t.notna()].copy()
    raw["data_datetime"] = t.loc[t.notna()]
    raw = raw.set_index("data_datetime").sort_index()

    print("[2/3] Rodando os 6 bundles + combinando via OR...")
    combined = score_t5_proxy_ensemble(raw)
    print(f"  duty do proxy combinado: {combined['t5_proxy_alert'].mean():.3f}")

    print("[3/3] Avaliando contra incidentes reais de T5_AVG_A (HI/HIHI-only)...")
    raw_alarms = load_alarms_gap(ALARM_CSV_DEFAULT, ["UNDER", "OVER", "LOLO", "CFN"])
    incidents = cluster_incidents(raw_alarms.get("T5_AVG_A", []))
    print(f"  {len(incidents)} incidentes de T5_AVG_A")

    alert = combined["t5_proxy_alert"].astype(bool)
    alert_times = alert.index[alert].values.astype("datetime64[s]").astype(int)
    inc_s = [int(pd.Timestamp(i).timestamp()) for i in incidents]
    horizon_sec = HORIZON_H * 3600

    n_hit = 0
    for ti in inc_s:
        w = alert_times[(alert_times >= ti - horizon_sec) & (alert_times <= ti)]
        if w.size:
            n_hit += 1
    recall = n_hit / len(incidents) if incidents else 0.0

    # FA: episódios de alerta contíguo que não caem perto de nenhum incidente
    alert_bool = alert.copy()
    groups = (alert_bool != alert_bool.shift()).cumsum()
    episodes = []
    for _, grp in alert_bool.groupby(groups):
        if grp.iloc[0]:
            episodes.append((grp.index[0], grp.index[-1]))
    total_days = (combined.index[-1] - combined.index[0]).total_seconds() / 86400.0
    n_fp = 0
    for s0, s1 in episodes:
        s0s, s1s = s0.timestamp(), s1.timestamp()
        hit_any = any((ti - horizon_sec <= s1s) and (ti >= s0s) for ti in inc_s)
        if not hit_any:
            n_fp += 1
    fa_per_day = n_fp / max(total_days, 1.0)

    print(f"\n=== Bundle deployável do ensemble (score_production real, sem sticky) ===")
    print(f"  recall={recall:.1%} ({n_hit}/{len(incidents)})  FA/dia={fa_per_day:.3f}  "
          f"duty={alert.mean():.3f}  episódios={len(episodes)}")


if __name__ == "__main__":
    main()
