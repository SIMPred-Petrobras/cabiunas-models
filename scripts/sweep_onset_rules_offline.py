#!/usr/bin/env python3
"""
sweep_onset_rules_offline.py
Ataca os falsos positivos do TC382_03_A SEM retreinar: só muda a REGRA DE DISPARO
sobre o mesmo health (rank da EWMA do MAE) do braço b2024.

Motivação medida na análise de duração: os 71 FPs passam apenas ~23% do tempo acima
do threshold (mediana 3,0h acima em 14,9h de episódio), enquanto os 29 TPs passam 51%
(26,7h em 45,7h). Ou seja: o FP típico TREME em torno do limiar e o sticky de 12h +
debounce de 8h colam os tremores num episódio único. Um disparo que exija excursão
mais forte deveria matar o tremor sem tocar no evento real.

Braços (todos causais — nada olha o futuro):
  base       — protocolo atual: alerta = h >= q, depois sticky 12h.
  hyst       — HISTERESE (Schmitt): arma em q_hi > q, sustenta enquanto h >= q_lo <= q.
               Desarma na parada (gap > 1h no health, i.e. máquina OFF) — ficar armado
               atravessando um desligamento não faz sentido físico.
  kofn       — CONFIRMAÇÃO: só dispara se >= frac das amostras das últimas N horas
               estiverem acima de q. Custa N horas de lead, por construção.
  mindur     — filtro de duração mínima (referência já medida; custa esperar M horas
               ANTES de notificar — inviável com lead mediano de 7,9h, entra só como
               teto de comparação).

Também mede o poder discriminante das features do PRÓPRIO SCORE em janela FIXA
precoce (3h/6h a partir do início do episódio): pico, densidade acima do limiar,
área acima do limiar e inclinação. As features de SENSOR já foram descartadas
(AUC 0,52-0,66, ver memória features-testadas-fp-vs-tp); estas nunca foram medidas.

Protocolo idêntico ao da auditoria: horizonte 8h, sticky 12h, debounce 8h,
incidentes HI/HIHI com maquina ON e filtro de fantasma (<500 C). O ponto de operacao
do braco base e verificado contra fleet_v13_b2024_FULL_hihihi.csv antes de comparar.

Uso:
    PYTHONPATH=. python scripts/sweep_onset_rules_offline.py
"""
from __future__ import annotations

import importlib.util
import os

import numpy as np
import pandas as pd
from clearml import Task

_HERE = os.path.dirname(os.path.abspath(__file__))


def _load(name: str):
    spec = importlib.util.spec_from_file_location(name, os.path.join(_HERE, f"{name}.py"))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


ev = _load("eval_per_sensor_level")
sw = _load("sweep_regime_band_offline")

SENSOR = "TC382_03_A"
TASK_B2024 = "1a15c26d994e44febb77f0bec8c2b378"
FLEET_CSV = "eval_predictive_out/fleet_v13_b2024_FULL_hihihi.csv"
OUT_CSV = "eval_predictive_out/onset_rules_TC382_03_A.csv"
OUT_FEAT = "eval_predictive_out/onset_early_features_TC382_03_A.csv"

HORIZON, STICKY = sw.HORIZON, sw.STICKY
OFF_GAP = pd.Timedelta(hours=1)   # gap no health = maquina parada -> desarma


# ---------------------------------------------------------------------------
# Primitivas vetorizadas (a apply_sticky do ev e O(n_alertas) em loop; aqui o
# sweep chama isso centenas de vezes)
# ---------------------------------------------------------------------------

def sticky_bool(sig: pd.Series, sticky_hours: float) -> pd.Series:
    """Estende cada True por sticky_hours. Equivalente vetorizado de ev.apply_sticky."""
    idx = sig.index
    v = sig.values.astype(bool)
    if sticky_hours <= 0 or not v.any():
        return pd.Series(v, index=idx)
    pos = np.flatnonzero(v)
    ends = idx.searchsorted(idx[pos] + pd.Timedelta(hours=sticky_hours), side="right")
    reach = np.zeros(len(idx), dtype=np.int64)
    reach[pos] = ends
    return pd.Series(np.maximum.accumulate(reach) > np.arange(len(idx)), index=idx)


def schmitt(h: pd.Series, q_hi: float, q_lo: float) -> pd.Series:
    """Arma em h>=q_hi, desarma em h<q_lo, mantem estado no meio. Reseta em gap OFF."""
    v = h.values
    state = np.where(v >= q_hi, 1.0, np.where(v < q_lo, 0.0, np.nan))
    s = pd.Series(state, index=h.index)
    # a maquina desligar zera o estado: o bloco ON seguinte comeca desarmado
    blk = (h.index.to_series().diff() > OFF_GAP).cumsum().values
    s = s.groupby(blk).ffill().fillna(0.0)
    return pd.Series(s.values > 0.5, index=h.index)


def k_of_n(h: pd.Series, q: float, window_h: float, frac: float) -> pd.Series:
    above = (h >= q).astype(float)
    dens = above.rolling(f"{int(window_h * 60)}min", min_periods=1).mean()
    return (dens >= frac) & (h >= q)


# ---------------------------------------------------------------------------
# Metrica end-to-end sobre um sinal de disparo qualquer
# ---------------------------------------------------------------------------

def evaluate(sig: pd.Series, incidents: list, total_days: float) -> dict:
    """recall/FA/duty/lead a partir do sinal BRUTO de disparo (pre-sticky).

    Deteccao = sinal ligado em algum ponto de [t-horizonte, t]. Para o braco base
    (sig = h>=q) isso reproduz exatamente o recall_raw da auditoria.
    """
    alert = sticky_bool(sig, STICKY)
    eps = ev.detect_episodes_gap(alert)
    inc_s = np.array([t.timestamp() for t in incidents], dtype=float)
    hs = HORIZON * 3600.0
    on_s = np.array([t.timestamp() for t in sig.index[sig.values]], dtype=float)

    leads, n_hit = [], 0
    for ti in inc_s:
        w = on_s[(on_s >= ti - hs) & (on_s <= ti)]
        if w.size:
            n_hit += 1
            leads.append((ti - w.min()) / 3600.0)
    n_fp = sum(1 for s0, s1 in eps
               if not (inc_s.size and np.any((inc_s - hs <= s1.timestamp())
                                             & (inc_s >= s0.timestamp()))))
    return dict(recall_raw=n_hit / len(inc_s) if inc_s.size else float("nan"),
                n_hit=n_hit, n_inc=len(inc_s), n_fp=n_fp,
                fa_per_day=n_fp / max(total_days, 1.0),
                duty_sticky=float(alert.mean()),
                n_eps=len(eps),
                lead_med_h=float(np.median(leads)) if leads else float("nan"))


# ---------------------------------------------------------------------------
# Features do proprio score em janela FIXA precoce
# ---------------------------------------------------------------------------

def auc(a: np.ndarray, b: np.ndarray) -> float:
    """Mann-Whitney: P(b > a), com empates valendo meio."""
    if not len(a) or not len(b):
        return float("nan")
    r = pd.Series(np.concatenate([a, b])).rank().values
    return (r[len(a):].sum() - len(b) * (len(b) + 1) / 2) / (len(a) * len(b))


def early_features(h: pd.Series, q: float, eps: list, hours: float) -> pd.DataFrame:
    rows = []
    for s0, s1 in eps:
        w = h[(h.index >= s0) & (h.index <= s0 + pd.Timedelta(hours=hours))]
        if len(w) < 3:
            rows.append({})
            continue
        x = (w.index - w.index[0]).total_seconds().values / 3600.0
        y = w.values
        rows.append(dict(pico=float(y.max()),
                         dens=float((y >= q).mean()),
                         area=float(np.clip(y - q, 0, None).mean()),
                         incl=float(np.polyfit(x, y, 1)[0]) if x.ptp() > 0 else 0.0))
    return pd.DataFrame(rows)


def main() -> None:
    running, tc03, _ = sw.load_raw()
    row = pd.read_csv(FLEET_CSV).set_index("sensor").loc[SENSOR]
    hl, q = float(row["hl"]), float(row["threshold_q"])

    mae = ev.load_mae_series(Task.get_task(task_id=TASK_B2024), [SENSOR])[SENSOR]
    h = sw.ewma_on(mae, hl, running).rank(pct=True)
    inc = sw.incidents_on(running, tc03, mae.index.min(), mae.index.max())
    total_days = (h.index[-1] - h.index[0]).total_seconds() / 86400.0

    base = evaluate(h >= q, inc, total_days)
    print(f"[sanidade] base: recall_raw={base['recall_raw']:.3f} (esp {float(row['recall_raw']):.3f})  "
          f"fa={base['fa_per_day']:.3f} (esp {float(row['fa_per_day']):.3f})  "
          f"duty={base['duty_sticky']:.3f} (esp {float(row['duty_sticky']):.3f})")
    if not (abs(base["recall_raw"] - float(row["recall_raw"])) < 0.01
            and abs(base["fa_per_day"] - float(row["fa_per_day"])) < 0.01
            and abs(base["duty_sticky"] - float(row["duty_sticky"])) < 0.01):
        raise SystemExit("ponto de operacao nao reproduz a auditoria — abortando.")
    print(f"           {base['n_inc']} incidentes ON, {base['n_fp']} FP, "
          f"{base['n_eps']} episodios, lead mediano {base['lead_med_h']:.1f}h, "
          f"{total_days:.0f} dias\n")

    rows = [dict(braco="base", param="q=%.4f" % q, latencia_h=0.0, **base)]

    # ---- histerese: arma mais alto, sustenta no mesmo lugar (ou abaixo)
    print("[hyst] arma em q_hi, sustenta em q_lo")
    for q_hi in [q + d for d in (0.005, 0.010, 0.015, 0.020, 0.030, 0.040)]:
        if q_hi >= 0.9995:
            continue
        for q_lo in (q, q - 0.02, q - 0.05):
            r = evaluate(schmitt(h, q_hi, q_lo), inc, total_days)
            rows.append(dict(braco="hyst", param=f"hi={q_hi:.4f} lo={q_lo:.4f}",
                             latencia_h=0.0, **r))
            print(f"  hi={q_hi:.4f} lo={q_lo:.4f}  recall={r['recall_raw']:.1%} "
                  f"fa={r['fa_per_day']:.3f}  duty={r['duty_sticky']:.3f}  "
                  f"FP={r['n_fp']:>3}  lead={r['lead_med_h']:.1f}h")

    # ---- confirmacao k-de-N (custa N horas de lead)
    print("\n[kofn] >= frac das ultimas N horas acima de q")
    for win in (1.0, 2.0, 3.0, 6.0):
        for frac in (0.5, 0.75, 0.9):
            r = evaluate(k_of_n(h, q, win, frac), inc, total_days)
            rows.append(dict(braco="kofn", param=f"N={win:.0f}h frac={frac:.2f}",
                             latencia_h=win * frac, **r))
            print(f"  N={win:>3.0f}h frac={frac:.2f}  recall={r['recall_raw']:.1%} "
                  f"fa={r['fa_per_day']:.3f}  duty={r['duty_sticky']:.3f}  "
                  f"FP={r['n_fp']:>3}  lead={r['lead_med_h']:.1f}h")

    # ---- duracao minima (referencia: exige esperar M horas para notificar)
    print("\n[mindur] referencia — exige M horas de episodio ANTES de notificar")
    alert0 = sticky_bool(h >= q, STICKY)
    eps0 = ev.detect_episodes_gap(alert0)
    inc_s = np.array([t.timestamp() for t in inc], dtype=float)
    hs = HORIZON * 3600.0
    for m in (6.0, 12.0, 18.0, 24.0):
        a, eps = ev.apply_min_duration(alert0, eps0, m)
        n_fp = sum(1 for s0, s1 in eps
                   if not np.any((inc_s - hs <= s1.timestamp()) & (inc_s >= s0.timestamp())))
        n_hit = sum(1 for ti in inc_s
                    if np.any([(s0.timestamp() - hs) <= ti <= s1.timestamp() for s0, s1 in eps]))
        rows.append(dict(braco="mindur", param=f"M={m:.0f}h", latencia_h=m,
                         recall_raw=n_hit / len(inc_s), n_hit=n_hit, n_inc=len(inc_s),
                         n_fp=n_fp, fa_per_day=n_fp / total_days,
                         duty_sticky=float(a.mean()), n_eps=len(eps),
                         lead_med_h=float("nan")))
        print(f"  M={m:>4.0f}h  recall_ep={n_hit / len(inc_s):.1%} "
              f"fa={n_fp / total_days:.3f}  duty={a.mean():.3f}  FP={n_fp:>3}")

    df = pd.DataFrame(rows)
    os.makedirs(os.path.dirname(OUT_CSV), exist_ok=True)
    df.to_csv(OUT_CSV, index=False)
    print(f"\nGravado: {OUT_CSV}")

    # ---- features do score em janela precoce: FP x TP
    print("\n[features do score em janela FIXA a partir do inicio do episodio]")
    matched, fps = [], []
    for s0, s1 in eps0:
        ok = bool(np.any((inc_s - hs <= s1.timestamp()) & (inc_s >= s0.timestamp())))
        (matched if ok else fps).append((s0, s1))
    print(f"  {len(fps)} FP x {len(matched)} TP")
    frows = []
    for hours in (3.0, 6.0, 12.0):
        ffp = early_features(h, q, fps, hours)
        ftp = early_features(h, q, matched, hours)
        for col in ("pico", "dens", "area", "incl"):
            a = ffp[col].dropna().values if col in ffp else np.array([])
            b = ftp[col].dropna().values if col in ftp else np.array([])
            v = auc(a, b)
            frows.append(dict(janela_h=hours, feature=col, n_fp=len(a), n_tp=len(b),
                              fp_med=np.median(a) if len(a) else np.nan,
                              tp_med=np.median(b) if len(b) else np.nan, auc=v))
            print(f"  {hours:>4.0f}h  {col:<5} FP={np.median(a) if len(a) else float('nan'):.4f} "
                  f"TP={np.median(b) if len(b) else float('nan'):.4f}  AUC={v:.3f}")
    pd.DataFrame(frows).to_csv(OUT_FEAT, index=False)
    print(f"\nGravado: {OUT_FEAT}")


if __name__ == "__main__":
    main()
