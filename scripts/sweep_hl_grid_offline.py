#!/usr/bin/env python3
"""
sweep_hl_grid_offline.py
Fecha duas lacunas do protocolo da auditoria, SEM retreinar e sem servidor ClearML:

  1) HL_GRID travada em [0.5, 1, 2, 4] (sweep_regime_band_offline.py:64). Os pontos
     de 8/12/24h nunca foram avaliados — e a meia-vida se mostrou a alavanca mais
     eficiente contra falso positivo que existe hoje (~4,3 FP removidos por
     incidente perdido, contra 0,0-0,6 de subir o threshold).

  2) Antecedência CENSURADA. A auditoria mede lead dentro da janela de horizonte
     de 8h, então toda meia-vida longa satura em ~7,9h e fica impossível saber se
     hl=24h atrasa a detecção. Aqui o sweep roda também com horizonte de 24h e 72h,
     que é onde o joelho da curva aparece.

Threshold = mu + y*sigma (regra SPC/Shewhart, a mesma da camada preditiva:
PREDICTIVE_SIGMA_Y_MIN/MAX), com mu/sigma calibrados SÓ na janela de treino do
braço — sem vazamento. Restrição de duty pós-sticky <= 0.25, igual à auditoria.

ATENÇÃO ao comparar entre horizontes: o horizonte entra na definição de FP
(episódio sem incidente em [s0, s1+H]), então FP cai mecanicamente quando H sobe.
Comparação válida é ENTRE meias-vidas DENTRO do mesmo horizonte.

Uso:
    PYTHONPATH=. python scripts/sweep_hl_grid_offline.py
    PYTHONPATH=. python scripts/sweep_hl_grid_offline.py --mae <arquivo.csv> --sensor TC382_03_A
"""
from __future__ import annotations

import argparse
import importlib.util
import os

import numpy as np
import pandas as pd

_HERE = os.path.dirname(os.path.abspath(__file__))


def _load(name: str):
    spec = importlib.util.spec_from_file_location(name, os.path.join(_HERE, f"{name}.py"))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


ev = _load("eval_per_sensor_level")
sw = _load("sweep_regime_band_offline")


def _resolve_dados() -> str:
    for up in ("..", "../..", "../../.."):
        cand = os.path.normpath(os.path.join(_HERE, "..", up, "dados"))
        if os.path.isdir(cand):
            return cand
    raise SystemExit("diretório 'dados/' não encontrado a partir do repo.")


DADOS = _resolve_dados()
sw.RAW_CSV = os.path.join(DADOS, "sensores_2024h2_2025_2026_30s.csv")
sw.ALARM_CSV = os.path.join(DADOS, "alarmes_selecionados_turbina_a.csv")
ev.ALARM_CSV_DEFAULT = sw.ALARM_CSV

CACHE = os.path.expanduser("~/.clearml/cache/storage_manager/global")
# MAE do braço v13 b2024 (task 2e92c618...), o mesmo usado em toda a análise de FP
MAE_DEFAULT = os.path.join(CACHE, "a2981618a3d0bd29b6605f991b2bab95.sequence_scores_all.csv")

HL_GRID = [0.5, 1.0, 2.0, 4.0, 8.0, 12.0, 24.0]   # estendida (auditoria parava em 4.0)
HORIZONS = [8.0, 24.0, 72.0]
MAX_STICKY_DUTY = 0.25
Y_GRID = np.round(np.arange(-1.0, 6.01, 0.05), 3)
TRAIN = (pd.Timestamp("2024-06-01", tz="UTC"), pd.Timestamp("2025-07-01", tz="UTC"))
OUT_CSV = "eval_predictive_out/hl_grid_sweep_{sensor}.csv"


def parse_args():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--mae", default=MAE_DEFAULT, help="CSV de sequence_scores_all em cache")
    p.add_argument("--sensor", default="TC382_03_A")
    p.add_argument("--out", default=None)
    return p.parse_args()


def metrics(sig: pd.Series, thr: float, inc: list, dias: float,
            horizon_h: float) -> dict:
    """recall_raw / FP por episódio / FA/dia / duty / lead mediano, num ponto fixo."""
    hs = horizon_h * 3600.0
    inc_s = np.array([t.timestamp() for t in inc])
    alert = ev.apply_sticky(sig, thr, sw.STICKY)
    eps = ev.detect_episodes_gap(alert)
    cross = np.array([t.timestamp() for t in sig.index[sig >= thr]])
    hits, leads = 0, []
    for ti in inc_s:
        if cross.size:
            m = (cross >= ti - hs) & (cross <= ti)
            if np.any(m):
                hits += 1
                leads.append((ti - cross[m].min()) / 3600.0)
    n_fp = sum(1 for (s0, s1) in eps
               if not np.any((inc_s - hs <= s1.timestamp()) & (inc_s >= s0.timestamp())))
    return {"hit": hits, "fp": n_fp, "n_eps": len(eps), "fa_per_day": n_fp / max(dias, 1.0),
            "duty_sticky": float(alert.mean()),
            "lead_med_h": float(np.median(leads)) if leads else float("nan")}


def main() -> None:
    args = parse_args()
    sensor = args.sensor
    sw.SENSOR = sensor
    out_csv = args.out or OUT_CSV.format(sensor=sensor)

    running, tc03, _t5 = sw.load_raw()
    mae = sw.read_mae(args.mae)
    print(f"MAE: {os.path.basename(args.mae)}  "
          f"{mae.index.min()} → {mae.index.max()}  n={len(mae)}")

    cenarios = [("FULL", None, None), ("BACKCAST_2024", *sw.BACKCAST), ("OOS", *sw.OOS)]
    rows = []
    for hl in HL_GRID:
        sig_full = sw.ewma_on(mae, hl, running)
        tr = sig_full[(sig_full.index >= TRAIN[0]) & (sig_full.index < TRAIN[1])].values
        mu, sg = float(tr.mean()), float(tr.std())
        print(f"[hl={hl:>4}h] mu={mu:.6f} sigma={sg:.6f}")
        for cen, t0, t1 in cenarios:
            s = sig_full
            if t0 is not None:
                s = s[s.index >= t0]
            if t1 is not None:
                s = s[s.index < t1]
            inc = sw.incidents_on(running, tc03, s.index.min(), s.index.max())
            if not inc:
                continue
            dias = (s.index[-1] - s.index[0]).total_seconds() / 86400.0
            for H in HORIZONS:
                for y in Y_GRID:
                    m = metrics(s, mu + y * sg, inc, dias, H)
                    rows.append({"sensor": sensor, "hl": hl, "horizon_h": H, "cenario": cen,
                                 "y": float(y), "mu": mu, "sigma": sg, "thr": mu + y * sg,
                                 "inc_on": len(inc), **m})

    df = pd.DataFrame(rows)
    os.makedirs(os.path.dirname(out_csv), exist_ok=True)
    df.to_csv(out_csv, index=False)
    print(f"\nGravado em {out_csv}  ({len(df)} linhas)")

    # Resumo: melhor ponto por (horizonte, cenário, hl) sob duty <= 0.25
    ok = df[df.duty_sticky <= MAX_STICKY_DUTY]
    for H in HORIZONS:
        for cen in ["FULL", "OOS", "BACKCAST_2024"]:
            d = ok[(ok.horizon_h == H) & (ok.cenario == cen)]
            if d.empty:
                continue
            ninc = int(df[df.cenario == cen]["inc_on"].iloc[0])
            print(f"\n===== horizonte {H:g}h | {cen} (inc={ninc}) | duty<=0.25 =====")
            print(f"  {'hl':>6} {'recall':>13} {'FP':>5} {'FA/dia':>8} {'y':>6} "
                  f"{'duty':>6} {'lead':>8}")
            for hl in HL_GRID:
                g = d[d.hl == hl]
                if g.empty:
                    print(f"  {hl:>5}h  — (duty sempre > 0.25)")
                    continue
                b = g.sort_values(["hit", "fp"], ascending=[False, True]).iloc[0]
                print(f"  {hl:>5}h {int(b.hit):>3}/{ninc} {b.hit/ninc:>7.1%} {int(b.fp):>5} "
                      f"{b.fa_per_day:>8.3f} {b.y:>6.2f} {b.duty_sticky:>6.3f} "
                      f"{b.lead_med_h:>7.1f}h")


if __name__ == "__main__":
    main()
